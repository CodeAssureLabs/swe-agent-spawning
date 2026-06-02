"""
State schema for LangGraph workflow.
"""

from typing import TypedDict, List, Annotated, Optional, Dict
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """
    State for the agentic exploration workflow.
    
    Uses LangGraph's message accumulation pattern.
    Notes are stored separately for injection into system prompt.
    """
    # Message history (accumulated by LangGraph)
    messages: Annotated[List[AIMessage | ToolMessage | SystemMessage | HumanMessage], add_messages]
    
    # Dynamic context - injected into system prompt at every step
    notes: str  # Current exploration notes
    subagent_summary: str  # Injected subagent registry summary (empty if no subagents)
    overall_context: str  # Onboarding brief for subagents (empty for orchestrator)
    agent_name: str  # Subagent name (empty for orchestrator)
    
    # Track tools executed in current agent+tool cycle (reset after each cycle)
    cycle_tools_executed: List[str]  # Names of tools called in this cycle

    # Index into messages[] where the current user turn starts (the HumanMessage).
    # Set once per ask_stream call so agent_step doesn't scan backwards each loop.
    current_turn_start_idx: int
    
    # Token usage tracking per LLM invocation
    token_usage_history: List[Dict]  # List of {call_id, input_tokens, output_tokens, total_tokens, tool_output_contributed_input_tokens}
    cumulative_tokens: Dict  # Rolling totals: {input_tokens, output_tokens, total_tokens, tool_output_contributed_input_tokens}
    
    # Session info
    model: str
    session_id: str
    codebase_path: str

    # When True, skip the decide_update_notes LLM call (eval/throwaway sessions)
    skip_notes: bool


class SessionMemory(TypedDict):
    """
    Persistent session memory stored in JSON.
    """
    session_id: str
    created_at: str
    last_active: str
    codebase_path: str
    
    visited_entities: dict  # {"files": [...], "functions": [...]}
    entity_cache: dict  # Cached entity details
    conversation_history: list  # Past Q&A pairs
    
    # Token tracking across invocations
    total_workflow_tokens: int  # Cumulative tokens across all invocations
    workflow_invocation_history: list  # List of {timestamp, input_tokens, output_tokens, total_tokens, tool_output_contributed_input_tokens}


class ExplorationState(TypedDict):
    """Legacy compatibility - maps to AgentState."""
    user_question: str
    conversation_history: list
    explored_entities: list
    findings: list
    tool_calls: list
    current_thought: str
    should_respond: bool
    final_answer: str
    citations: list
    session_id: str
    codebase_path: str
    session_memory: SessionMemory
    tokens_used: int
    token_budget: int
