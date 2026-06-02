"""
LangGraph workflow nodes for agentic exploration.

Key design:
- Notes are INJECTED into the system prompt at every step
- We control the loop and can add custom logic
- Before refreshing notes, agent decides whether to update them
"""

from typing import Dict, List, Any, Callable, Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from .providers.base import BaseLLMProvider

from .state import AgentState
from .memory import SessionManager
from .tools._tool_message_executor import execute_tool_calls
from .token_logger import log_token_usage
from .tools.context import set_codebase_path, set_session_context


def calculate_tool_output_contributed_tokens(messages: List[BaseMessage]) -> int:
    """
    Estimate the number of input tokens contributed by tool outputs in the messages.
    
    Uses a word-count heuristic: approximate tokens = word_count / 0.75
    This gives a rough estimate of context tokens from prior tool responses being fed
    back to the LLM.
    
    Args:
        messages: List of messages including Tool responses
        
    Returns:
        Approximate token count from tool outputs
    """
    tool_output_tokens = 0
    
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Get the content from ToolMessage
            content = msg.content
            if isinstance(content, str):
                # Simple word count heuristic: tokens ≈ words / 0.75
                word_count = len(content.split())
                estimated_tokens = max(1, int(word_count / 0.75))
                tool_output_tokens += estimated_tokens
    
    return tool_output_tokens


def update_token_tracking(
    state: AgentState,
    call_id: str,
    messages: List[BaseMessage],
    usage: Dict
) -> Dict:
    # Calculate tool-contributed input tokens
    tool_output_contributed_tokens = calculate_tool_output_contributed_tokens(messages)
    
    # Build token usage entry with tool context
    token_entry = {
        "call_id": call_id,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "tool_output_contributed_input_tokens": tool_output_contributed_tokens,
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
    }

    # Update state's token tracking history
    updated_history = state.get("token_usage_history", []) + [token_entry]

    # Recalculate cumulative totals
    cumulative = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "tool_output_contributed_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    for entry in updated_history:
        cumulative["input_tokens"] += entry.get("input_tokens", 0)
        cumulative["output_tokens"] += entry.get("output_tokens", 0)
        cumulative["total_tokens"] += entry.get("total_tokens", 0)
        cumulative["tool_output_contributed_input_tokens"] += entry.get("tool_output_contributed_input_tokens", 0)
        cumulative["cache_read_input_tokens"] += entry.get("cache_read_input_tokens", 0)
        cumulative["cache_creation_input_tokens"] += entry.get("cache_creation_input_tokens", 0)
    
    return {
        "token_usage_history": updated_history,
        "cumulative_tokens": cumulative,
    }


def build_static_system_prompt(state: AgentState) -> str:
    """
    Build the static (cacheable) portion of the system prompt.

    This string must be identical on every invocation for a given agent so
    Anthropic's prompt cache can hit it. No notes, no subagent summary here.
    """
    agent_name = state.get("agent_name", "")

    if agent_name:
        overall_context = state.get("overall_context", "")
        return f"""{overall_context}

You are {agent_name}, a domain-specialist subagent.

RULES:
- Answer from your domain knowledge and tools
- If the question is outside your scope, say so clearly
- Cite specific files and line numbers
- Use update_note to record findings"""
    else:
        return """You are an expert codebase analysis agent and assistant.

You have tools to read files, search code, parse structure, and trace calls.
Answer the user's question accurately with specific file and line citations.

RULES:
- Be efficient: refer to notes before re-reading files
- Be specific: cite files and line numbers
- You must use update_note to record any new findings for any file you have read or discovered to exist
- Stop when you have enough information to answer
- Do not fabricate information, do not guess, do not consider whether something is possible or likely - only use the information you have.
- Always base your answer on facts you have found in the codebase
- Do not write/create any documentation/markdown/notes/files unless instructed or implied to do so. You may assume all communication & knowledge is via your conversation."""


def _repl_namespace_summary(session_id: str) -> str:
    """Summarize REPL namespace variables for injection into context."""
    try:
        from .tools.repl_tool import _namespaces
    except ImportError:
        return ""
    ns = _namespaces.get(session_id)
    if not ns:
        return ""

    # Items to skip — injected by the REPL tool itself, not user-created
    from .tools.context import get_tools
    tool_names = {t.name for t in get_tools()}
    sync_names = {f"{n}_sync" for n in tool_names}
    skip = tool_names | sync_names | {"llm", "asyncio", "__repl_async__"}

    lines = []
    context_dict = ns.get("context", {})
    if context_dict:
        keys_preview = ", ".join(f'"{k}"' for k in list(context_dict.keys())[:10])
        lines.append(f"  context: dict with {len(context_dict)} file(s) — [{keys_preview}]")

    for key, val in ns.items():
        if key in skip or key == "context" or key.startswith("_"):
            continue
        type_name = type(val).__name__
        if isinstance(val, (list, tuple)):
            lines.append(f"  {key}: {type_name} ({len(val)} items)")
        elif isinstance(val, dict):
            lines.append(f"  {key}: {type_name} ({len(val)} keys)")
        elif isinstance(val, str):
            lines.append(f"  {key}: str ({len(val)} chars)")
        elif isinstance(val, (int, float, bool)):
            lines.append(f"  {key}: {type_name} = {val}")
        else:
            lines.append(f"  {key}: {type_name}")

    if not lines:
        return ""
    return "REPL NAMESPACE (persistent variables available in python_repl):\n" + "\n".join(lines)


def build_dynamic_context(state: AgentState) -> str:
    """
    Build the dynamic portion of the system prompt (changes every turn).

    Contains notes and subagent summary — injected as a separate SystemMessage
    after the static one so the static prefix remains cache-eligible.
    """
    notes = state.get("notes", "")
    agent_name = state.get("agent_name", "")
    session_id = state.get("session_id", "")
    repl_summary = _repl_namespace_summary(session_id)

    if agent_name:
        parts = [f"""EXPLORATION NOTES:
{notes if notes else "(No notes yet - use update_note to record learnings)"}"""]
        if repl_summary:
            parts.append(repl_summary)
        return "\n\n".join(parts)
    else:
        subagent_summary = state.get("subagent_summary", "")
        parts = [f"""EXPLORATION NOTES (what you've already learned):
{notes if notes else "(No notes yet - use update_note to record learnings)"}"""]
        if repl_summary:
            parts.append(repl_summary)
        if subagent_summary:
            parts.append(subagent_summary)
        return "\n\n".join(parts)


async def agent_step(state: AgentState, llm: BaseLLMProvider, tool_registry) -> Dict:
    """
    Single agent reasoning step.

    1. Resets cycle tracking (new agent turn)
    2. Injects notes into system prompt
    3. Calls LLM with tools bound
    4. Tracks token usage with tool context breakdown
    5. Returns updated messages and token tracking
    """
    # Re-anchor ContextVars from state so tool calls use the correct codebase/session
    # even when LangGraph resumes this coroutine across internal task boundaries.
    if state.get("codebase_path"):
        set_codebase_path(state["codebase_path"])

    # Reset cycle tools (start of new reasoning cycle)
    cycle_tools = []

    # Read current tools from registry (mutable — picks up dynamically added tools)
    # Tools are read at call time, not captured by partial at graph compile time
    tools = tool_registry.get_all_base_tools()
    llm.bind_tools(tools)

    # Static prompt first — identical every turn, cache-eligible prefix.
    # Prior completed turns follow (cached after first use).
    # Dynamic context injected at the boundary of the current turn so it applies
    # to the entire current question + tool loop. The split index is computed
    # once per turn in ask_stream and stored in state to avoid scanning on every
    # tool-loop re-entry.
    history = state["messages"]
    turn_start = state.get("current_turn_start_idx", 0)
    messages: List[BaseMessage] = [SystemMessage(content=build_static_system_prompt(state))]
    messages.extend(history[:turn_start])
    messages.append(SystemMessage(content=build_dynamic_context(state)))
    messages.extend(history[turn_start:])
    
    # Call LLM with tools
    response, tool_calls, usage = await llm.ainvoke(messages)

    # Update token tracking with tool context analysis
    token_updates = update_token_tracking(state, "agent_step", messages, usage)

    # Log token usage to filesystem
    log_token_usage(state["session_id"], "orchestrator", "agent_step", usage)
    
    return {
        "messages": [*state["messages"], response],
        "cycle_tools_executed": cycle_tools,
        **token_updates,
    }


async def tool_executor(state: AgentState, tool_registry) -> Dict:
    """
    Execute tool calls from the agent and track which tools were called.
    """
    # Re-anchor ContextVars so tools see the correct codebase/session for this agent.
    if state.get("codebase_path"):
        set_codebase_path(state["codebase_path"])
    tools_by_name = tool_registry.get_base_tools_by_name()
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    tool_messages = await execute_tool_calls(tool_calls, tools_by_name)
    
    # Track tool names executed in this cycle
    tools_executed = [call.get("name", "") for call in tool_calls]
    cycle_tools = state.get("cycle_tools_executed", []) + tools_executed

    return {"messages": [*state["messages"], *tool_messages], "cycle_tools_executed": cycle_tools}


async def decide_update_notes(state: AgentState, llm: BaseLLMProvider, session_manager: SessionManager, tool_registry) -> Dict:
    """
    Decide whether to update notes based on recent tool calls.

    This runs after tool execution to let the agent decide if it should:
    1. Update notes with findings (via update_note tool)
    2. Continue to next reasoning step

    Compares findings against current notes to avoid redundant updates.
    Rebinds LLM with ONLY note tools to constrain capabilities.
    Manually executes tool calls (not saved to conversation history).
    Returns updated state with messages from note updates.
    """
    # Get current notes content for comparison
    session_id = state["session_id"]
    current_notes = session_manager.notes_for_prompt(session_id)

    # Pull note tools from the agent's own registry (respects scoping for subagents)
    note_tools = [t for t in tool_registry.get_all_base_tools() if t.name in ("update_note", "delete_note")]
    note_tools_by_name = {t.name: t for t in note_tools}

    llm.bind_tools(note_tools) # only bind with note tools
    
    # Build prompt asking agent whether to update notes, with current notes for comparison
    decision_prompt = f"""Based on the message history, determine if you should update your notes.

CURRENT_NOTES:
{current_notes if current_notes else "(No notes yet)"}

You must update notes if:
- You discovered new information NOT already in the CURRENT_NOTES
- You found important patterns or relationships
- You read or referenced a file for the first time
- You can summarise, or group together information in a more compact way

Compare what you just learned against the CURRENT_NOTES. Only call update_note if there's genuinely new information to add that is not in your CURRENT_NOTES.
Then respond with your decision.

IMPORTANT: YOU MAY ONLY CALL NOTE RELATED TOOLS. YOUR JOB IS ONLY TO UPDATE NOTES IF NEEDED. YOU CANNOT GATHER MORE INFORMATION.
"""

    messages = [
        *state["messages"],
        SystemMessage(content=decision_prompt),
    ]
    
    # Call LLM to decide and execute note updates
    response, tool_calls, usage = await llm.ainvoke(messages)
    print("Decision response:", response)

    # Update token tracking with tool context analysis
    token_updates = update_token_tracking(state, "decide_update_notes", messages, usage)

    # Log token usage to filesystem
    log_token_usage(state["session_id"], "orchestrator", "decide_update_notes", usage)
    
    # Manually execute any tool calls from the response
    tool_messages = await execute_tool_calls(
        tool_calls,
        note_tools_by_name,
        unknown_tool_prefix="Unknown note tool",
    )

    print(tool_messages)
    return token_updates


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Decide whether to continue tool calling or end.
    """
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # Otherwise, we're done
    return "end"

def might_update_notes(state: AgentState) -> Literal["decide_update_notes", "end"]:
    """
    Decide whether to update notes at the end of the workflow.

    Triggers if read_file was called (new file content was observed) or
    if any subagent was invoked (new information returned from a specialist).
    Skipped entirely when skip_notes is set (e.g. eval sessions).
    """
    if state.get("skip_notes", False):
        return "end"
    cycle_tools = state.get("cycle_tools_executed", [])
    if "read_file" in cycle_tools:
        return "decide_update_notes"
    if "consult_agents" in cycle_tools:
        return "decide_update_notes"
    return "end"


def refresh_notes(state: AgentState, session_manager: SessionManager, subagent_registry=None) -> Dict:
    """
    Refresh notes and subagent summary from session storage.
    Called before each agent step to get latest state.
    """
    session_id = state["session_id"]
    notes = session_manager.notes_for_prompt(session_id)
    updates: Dict = {"notes": notes}
    if subagent_registry is not None:
        updates["subagent_summary"] = subagent_registry.summary_for_prompt()
    return updates
