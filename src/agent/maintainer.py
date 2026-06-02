"""
Codebase Maintainer Agent - Custom LangGraph workflow.

Key features:
- Notes injected into system prompt at every step
- LLM decides tools and when to stop
- Session persistence with exploration memory
- Token tracking with styled console output
"""

from typing import Dict, Optional, List
from functools import partial

from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from .state import AgentState
from .memory import SessionManager
from .nodes import agent_step, refresh_notes, tool_executor, decide_update_notes, should_continue, might_update_notes
from .tools import (
    get_all_tools, get_core_tools, get_analysis_tools, get_repl_only_tools,
    set_codebase_path, set_session_context,
    set_llm_provider, set_tools,
    reset_repl_llm_token_usage, get_repl_llm_token_usage,
)
from .tool_registry import ToolRegistry
from .providers import get_provider
from .token_logger import read_and_purge, purge_stale_logs


class CodebaseAgent:
    """
    Fully agentic codebase analysis agent.
    
    Uses a custom LangGraph workflow so we can:
    - Inject notes into the system prompt at every step
    - Control the loop with custom logic
    - Refresh dynamic context between steps
    
    Capabilities:
    - Read, search, and parse code files
    - Search codebase with ripgrep
    - Automatic exploration notes in context
    - Spawn and consult scoped folder agents
    """
    
    def __init__(
        self,
        model: str,
        codebase_path: str,
        session_id: Optional[str] = None,
        tool_set: str = "all",
        provider: str = "anthropic",
        scoped_folders: Optional[List[str]] = None,
        overall_context: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            model: LLM model name
            codebase_path: Path to the codebase to analyze
            session_id: Optional session ID to resume
            tool_set: Which tools to expose - "core", "analysis", or "all"
            provider: LLM provider name ("anthropic", "openai", "openrouter", "api")
            scoped_folders: If set, agent runs in subagent mode scoped to these folders
            overall_context: Optional context string for subagent mode
            agent_name: Optional name for this agent (used in subagent mode)
        """
        self.codebase_path = codebase_path
        self.tool_set = tool_set
        self.scoped_folders = scoped_folders
        self.overall_context = overall_context or ""
        self.agent_name = agent_name or ""
        self.model = model
        self.provider = provider

        if scoped_folders:
            # --- Subagent mode ---
            # Do NOT touch module-level globals (codebase_path, session_context, tool_registry).
            # The orchestrator already set them; overwriting would break the orchestrator's tools.
            from .tools.scoped_tools import (
                make_scoped_read_file, make_scoped_list_files, make_scoped_search,
                make_scoped_update_note, make_scoped_delete_note,
            )

            # Namespaced session under the orchestrator's session dir
            from pathlib import Path
            subagent_db_dir = Path(SessionManager().db_dir) / "subagents" / (agent_name or "unknown")
            self.session_manager = SessionManager(db_dir=subagent_db_dir)
            self.session = self.session_manager.load_or_create_session(codebase_path=codebase_path, session_id=session_id)

            session_dir = self.session_manager.sessions_dir / self.session["session_id"]
            self.tool_registry = ToolRegistry(str(session_dir))
            self.tool_registry.register_base_tool(make_scoped_read_file(codebase_path, scoped_folders))
            self.tool_registry.register_base_tool(make_scoped_list_files(codebase_path, scoped_folders))
            self.tool_registry.register_base_tool(make_scoped_search(codebase_path, scoped_folders))
            self.tool_registry.register_base_tool(make_scoped_update_note(self.session_manager, self.session["session_id"]))
            self.tool_registry.register_base_tool(make_scoped_delete_note(self.session_manager, self.session["session_id"]))
        else:
            # --- Orchestrator mode ---
            set_codebase_path(codebase_path)

            self.session_manager = SessionManager()
            self.session = self.session_manager.load_or_create_session(
                codebase_path=codebase_path,
                session_id=session_id
            )
            set_session_context(self.session_manager, self.session["session_id"])

            session_dir = self.session_manager.sessions_dir / self.session["session_id"]
            self.tool_registry = ToolRegistry(str(session_dir))

            # Register core tools
            core_tools = self._get_tools(tool_set)
            for t in core_tools:
                if isinstance(t, BaseTool):
                    self.tool_registry.register_base_tool(t)

            # Load subagent registry and AgentBuilder
            from .subagent_registry import SubAgentRegistry
            from .agent_builder import AgentBuilder
            from .tools.agent_tools import set_agent_builder

            self.subagent_registry = SubAgentRegistry(str(session_dir), codebase_path)

            self.agent_builder = AgentBuilder(
                llm=get_provider(provider, model),
                codebase_path=codebase_path,
                subagent_registry=self.subagent_registry,
                tool_registry=self.tool_registry,
                model=model,
                provider=provider,
            )
            set_agent_builder(self.agent_builder)

            self._load_subagent_handles()

        self.llm = get_provider(provider, model)

        # Make LLM and tools available to REPL tool
        set_llm_provider(self.llm)
        set_tools(self.tool_registry.get_all_base_tools())

        # Build the graph
        self.graph = self._build_graph()
    

    
    def _build_tokens_usage(self, token_usage_history: List[Dict], log_data: Dict) -> Dict:
        """
        Build a structured namespace breakdown of token usage.

        Args:
            token_usage_history: List of token usage dicts from LangGraph state (orchestrator calls)
            log_data: Dict of namespace -> list of entries from read_and_purge

        Returns:
            Dict with orchestrator, agent_builder, subagents, and total buckets
        """
        def _sum_entries(entries: List[Dict]) -> Dict:
            bucket = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "num_calls": len(entries),
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
            for e in entries:
                bucket["input_tokens"] += e.get("input_tokens", 0)
                bucket["output_tokens"] += e.get("output_tokens", 0)
                bucket["total_tokens"] += e.get("total_tokens", 0)
                bucket["cache_read_input_tokens"] += e.get("cache_read_input_tokens", 0)
                bucket["cache_creation_input_tokens"] += e.get("cache_creation_input_tokens", 0)
            return bucket

        # Orchestrator bucket — from LangGraph state (authoritative)
        orchestrator_bucket: Dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "tool_output_contributed_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "num_calls": len(token_usage_history),
        }
        for entry in token_usage_history:
            orchestrator_bucket["input_tokens"] += entry.get("input_tokens", 0)
            orchestrator_bucket["output_tokens"] += entry.get("output_tokens", 0)
            orchestrator_bucket["total_tokens"] += entry.get("total_tokens", 0)
            orchestrator_bucket["tool_output_contributed_input_tokens"] += entry.get("tool_output_contributed_input_tokens", 0)
            orchestrator_bucket["cache_read_input_tokens"] += entry.get("cache_read_input_tokens", 0)
            orchestrator_bucket["cache_creation_input_tokens"] += entry.get("cache_creation_input_tokens", 0)

        # Agent builder bucket — from log_data
        agent_builder_bucket = _sum_entries(log_data.get("agent_builder", []))

        # Subagents bucket — sum all subagent_* namespaces
        subagents_bucket: Dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "num_calls": 0,
                                   "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        for namespace, entries in log_data.items():
            if namespace.startswith("subagent_"):
                partial_bucket = _sum_entries(entries)
                subagents_bucket["input_tokens"] += partial_bucket["input_tokens"]
                subagents_bucket["output_tokens"] += partial_bucket["output_tokens"]
                subagents_bucket["total_tokens"] += partial_bucket["total_tokens"]
                subagents_bucket["num_calls"] += partial_bucket["num_calls"]
                subagents_bucket["cache_read_input_tokens"] += partial_bucket["cache_read_input_tokens"]
                subagents_bucket["cache_creation_input_tokens"] += partial_bucket["cache_creation_input_tokens"]

        # Total bucket — sum of all three
        total_bucket = {
            "input_tokens": orchestrator_bucket["input_tokens"] + agent_builder_bucket["input_tokens"] + subagents_bucket["input_tokens"],
            "output_tokens": orchestrator_bucket["output_tokens"] + agent_builder_bucket["output_tokens"] + subagents_bucket["output_tokens"],
            "total_tokens": orchestrator_bucket["total_tokens"] + agent_builder_bucket["total_tokens"] + subagents_bucket["total_tokens"],
            "num_calls": orchestrator_bucket["num_calls"] + agent_builder_bucket["num_calls"] + subagents_bucket["num_calls"],
            "cache_read_input_tokens": orchestrator_bucket["cache_read_input_tokens"] + agent_builder_bucket["cache_read_input_tokens"] + subagents_bucket["cache_read_input_tokens"],
            "cache_creation_input_tokens": orchestrator_bucket["cache_creation_input_tokens"] + agent_builder_bucket["cache_creation_input_tokens"] + subagents_bucket["cache_creation_input_tokens"],
        }

        return {
            "orchestrator": orchestrator_bucket,
            "agent_builder": agent_builder_bucket,
            "subagents": subagents_bucket,
            "total": total_bucket,
        }
    
    def _get_tools(self, tool_set: str) -> list:
        """Get tools based on the specified set."""
        if tool_set == "core":
            return get_core_tools()
        elif tool_set == "analysis":
            return get_analysis_tools()
        elif tool_set == "repl":
            return get_repl_only_tools()
        else:
            return get_all_tools()
    
    def _load_subagent_handles(self) -> None:
        """Load existing subagent definitions from registry and create handles."""
        from .subagent_handle import SubAgentHandle

        for agent_def in self.subagent_registry.list_agents():
            scoped_agent = CodebaseAgent(
                model=self.model,
                codebase_path=self.codebase_path,
                tool_set="core",
                provider=self.provider,
                scoped_folders=agent_def["folders"],
                overall_context=agent_def.get("overall_context", ""),
                agent_name=agent_def["name"],
                session_id=agent_def.get("session_id") or None,
            )
            handle = SubAgentHandle(agent_def, scoped_agent=scoped_agent, orchestrator_session_id=self.session["session_id"])
            self.agent_builder.register_subagent_handle(agent_def, handle)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Node: Refresh notes from storage and subagent summary (if orchestrator)
        subagent_reg = getattr(self, "subagent_registry", None)
        workflow.add_node(
            "refresh_notes",
            partial(refresh_notes, session_manager=self.session_manager, subagent_registry=subagent_reg)
        )
        
        # Node: Agent reasoning step
        workflow.add_node(
            "agent",
            partial(agent_step, llm=self.llm, tool_registry=self.tool_registry)
        )

        # Node: Tool execution
        workflow.add_node(
            "tools",
            partial(tool_executor, tool_registry=self.tool_registry)
        )
        
        # Node: Decide whether to update notes (tool calls here are not saved to history)
        # Rebinds LLM with ONLY note tools for this node
        workflow.add_node(
            "decide_update_notes",
            partial(decide_update_notes, llm=self.llm, session_manager=self.session_manager, tool_registry=self.tool_registry)
        )
        
        # Node: Router for deciding whether to update notes (pass-through for routing logic)
        workflow.add_node("might_update_notes", lambda state: state)
        
        # Flow:
        # refresh_notes -> agent -> (tools -> agent) OR (might_update_notes check)
        workflow.set_entry_point("refresh_notes")
        workflow.add_edge("refresh_notes", "agent")
        
        # Agent routes: continue with tools or check if notes need updating
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": "might_update_notes"}
        )
        
        # After tools, go back to agent for next reasoning step
        workflow.add_edge("tools", "agent")
        
        # At end of agent work, check if notes should be updated
        workflow.add_conditional_edges(
            "might_update_notes",
            might_update_notes,
            {"decide_update_notes": "decide_update_notes", "end": END}
        )
        
        # After deciding on note updates, end the workflow
        workflow.add_edge("decide_update_notes", END)

        # Compile with checkpointer
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory) #type: ignore
    
    async def ask_stream(self, question: str, skip_notes: bool = False):
        """
        Ask a question and stream the response.

        Yields:
            dict with 'type' and 'content':
            - {"type": "thinking", "content": "..."} for agent thinking
            - {"type": "tool_call", "content": "..."} for tool calls
            - {"type": "tool_done", "content": "..."} for tool results
            - {"type": "done", "answer": "...", "citations": [...]} at the end
        """
        purge_stale_logs(self.session["session_id"])
        reset_repl_llm_token_usage()

        # Get current notes (seed value — refresh_notes node will update before first agent_step)
        notes = self.session_manager.notes_for_prompt(self.session["session_id"])
        subagent_summary = ""
        if hasattr(self, "subagent_registry"):
            subagent_summary = self.subagent_registry.summary_for_prompt()

        # Initial state
        state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "current_turn_start_idx": 0,
            "notes": notes,
            "subagent_summary": subagent_summary,
            "overall_context": self.overall_context,
            "agent_name": self.agent_name,
            "model": self.model,
            "session_id": self.session["session_id"],
            "codebase_path": self.codebase_path,
            "cycle_tools_executed": [],
            "token_usage_history": [],
            "cumulative_tokens": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "tool_output_contributed_input_tokens": 0,
            },
            "skip_notes": skip_notes,
        }
        
        config = {"configurable": {"thread_id": self.session["session_id"]}, "recursion_limit": 100}
        full_answer = ""
        context_window_used = 0

        # Use astream to get node-by-node updates
        async for mode, chunk in self.graph.astream(state, config, stream_mode=["updates", "messages"]):
            if mode == "updates":
                # Store the latest complete state
                node, state = next(iter(chunk.items()))

                if node == "tools":
                    # Emit tool_call + tool_done for each call/result pair
                    messages = state.get("messages", [])
                    # ToolMessages are at the end; AI message with tool_calls precedes them
                    from langchain_core.messages import AIMessage, ToolMessage as LCToolMessage
                    tool_messages = [m for m in messages if isinstance(m, LCToolMessage)]
                    ai_messages = [m for m in messages if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)]
                    if ai_messages and tool_messages:
                        last_ai = ai_messages[-1]
                        # Match tool_calls to tool_messages by position (same order guaranteed)
                        tc_list = last_ai.tool_calls
                        # Only the newly added ones (same count as tc_list)
                        new_tool_msgs = tool_messages[-len(tc_list):]
                        for tc, tm in zip(tc_list, new_tool_msgs):
                            yield {"type": "tool_call", "name": tc.get("name", ""), "args": tc.get("args", {})}
                            # Emit structured repl events for python_repl calls
                            if tc.get("name") == "python_repl":
                                yield {"type": "repl_code", "content": tc.get("args", {}).get("code", "")}
                                yield {"type": "repl_output", "content": tm.content}
                            yield {"type": "tool_done", "name": tc.get("name", ""), "result": tm.content}

                # Yield token metrics after LLM-invoking nodes
                if node in ["agent", "decide_update_notes"] and state.get("cumulative_tokens"):
                    cumulative = state.get("cumulative_tokens", {})
                    # Track peak input tokens across orchestrator calls
                    history = state.get("token_usage_history", [])
                    if history:
                        latest_input = history[-1].get("input_tokens", 0)
                        context_window_used = max(context_window_used, latest_input)
                    yield {"type": "tokens_usage", "node": node, "tokens": cumulative, "context_window_used": context_window_used}

                yield {"type": "update", "content": chunk}
            
            elif mode == "messages":
                message, metadata = chunk
                for content in message.content_blocks:
                    if content['type'] == "text":
                        yield {"type": "token", "content": content['text']}
        
        # Aggregate final token metrics
        token_history = state.get("token_usage_history", [])
        log_data = read_and_purge(self.session["session_id"])
        final_token_metrics = self._build_tokens_usage(token_history, log_data)
        final_token_metrics["context_window_used"] = context_window_used

        # Merge REPL inline-LLM token usage into the total
        repl_tokens = get_repl_llm_token_usage()
        if repl_tokens.get("total_tokens", 0) > 0:
            if "total" in final_token_metrics:
                for k in ("input_tokens", "output_tokens", "total_tokens"):
                    final_token_metrics["total"][k] = final_token_metrics["total"].get(k, 0) + repl_tokens.get(k, 0)
            final_token_metrics["repl_llm_input_tokens"] = repl_tokens["input_tokens"]
            final_token_metrics["repl_llm_output_tokens"] = repl_tokens["output_tokens"]
            final_token_metrics["repl_llm_total_tokens"] = repl_tokens["total_tokens"]
        
        # Extract final answer from the last AI message in the final state
        final_response = state["messages"][-1]
        full_answer = final_response.content
        
        if isinstance(full_answer, list):
            full_answer = "".join(
                part['text'] for part in full_answer if part['type'] == 'text' #type: ignore for anthropic messages
            )

        # Extract citations and update session
        citations = self._extract_citations(full_answer)
        self._update_session(question, full_answer, citations, final_token_metrics)
        
        yield {"type": "done", "answer": full_answer, "citations": citations, "tokens_usage": final_token_metrics}
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract file:line citations from text."""
        import re
        # Match patterns like file.py:10, file.py:10-20, path/to/file.py:5
        pattern = r'[\w/.-]+\.\w+:\d+(?:-\d+)?'
        return list(set(re.findall(pattern, text)))
    
    def _update_session(self, question: str, answer: str, citations: list, token_metrics: Optional[Dict] = None):
        """Update session with new conversation and token metrics."""
        from datetime import datetime
        
        self.session["conversation_history"].append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat(),
        })
        self.session["conversation_history"].append({
            "role": "assistant", 
            "content": answer,
            "timestamp": datetime.now().isoformat(),
            "citations": citations
        })
        self.session["last_active"] = datetime.now().isoformat()
        
        # Track tokens in workflow invocation history
        if token_metrics:
            invocation_entry = {
                "timestamp": datetime.now().isoformat(),
                "tokens_usage": token_metrics,
            }
            if "workflow_invocation_history" not in self.session:
                self.session["workflow_invocation_history"] = []
            self.session["workflow_invocation_history"].append(invocation_entry)

            # Update cumulative token count
            if "total_workflow_tokens" not in self.session:
                self.session["total_workflow_tokens"] = 0
            self.session["total_workflow_tokens"] += token_metrics.get("total", {}).get("total_tokens", 0)
        
        self.session_manager._save_session(self.session)
    
    def get_session_info(self) -> Dict:
        """Get current session info."""
        notes_stats = self.session_manager.notes_stats(self.session["session_id"])
        
        return {
            "session_id": self.session["session_id"],
            "codebase_path": self.session["codebase_path"],
            "created_at": self.session.get("created_at", ""),
            "last_active": self.session.get("last_active", ""),
            "conversation_count": len(self.session.get("conversation_history", [])) // 2,
            "notes_count": notes_stats["entries"],
        }
    
    @classmethod
    def resume_session(cls, session_id: str) -> "CodebaseAgent":
        """Resume an existing session."""
        session_manager = SessionManager()
        session = session_manager.load_session(session_id)

        if not session:
            raise ValueError(f"Session not found: {session_id}")

        return cls(
            codebase_path=session["codebase_path"],
            session_id=session_id
        )

    def inject_repl_vars(self, **kwargs) -> None:
        """
        Inject variables directly into this session's REPL namespace.

        They are immediately available in any subsequent python_repl call
        without appearing in the LLM context unless the REPL code explicitly
        references and prints them.
        """
        from .tools.repl_tool import _get_or_create_namespace
        ns = _get_or_create_namespace(self.session["session_id"])
        ns.update(kwargs)
