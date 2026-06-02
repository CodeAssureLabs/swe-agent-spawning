# src/agent/subagent_handle.py
"""
SubAgentHandle — wraps a scoped CodebaseAgent as an internal callable handle.

The orchestrator LLM calls consult_agents, not these handles directly. The
handle forwards the question to the scoped agent, runs its full graph, and
returns the free-text answer as the tool result.
"""

from typing import Dict, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from .token_logger import log_token_usage


class SubAgentInput(BaseModel):
    question: str = Field(description="The question to ask this subagent")


class SubAgentHandle(BaseTool):
    """
    A tool that wraps a scoped CodebaseAgent instance.

    When invoked, it forwards the question to the scoped agent,
    runs its exploration graph, and returns the final answer.
    """

    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = SubAgentInput
    agent_def: Dict = {}
    scoped_agent: object = None  # CodebaseAgent instance (typed loosely to avoid circular imports)
    orchestrator_session_id: str = ""

    def __init__(self, agent_def: Dict, scoped_agent: object, orchestrator_session_id: str = ""):
        super().__init__(
            name=f"invoke_subagent_{agent_def['name']}",
            description=agent_def.get("job_description", "(no description)"),
            agent_def=agent_def,
            scoped_agent=scoped_agent,
            orchestrator_session_id=orchestrator_session_id,
        )

    def _run(self, question: str) -> str:
        raise NotImplementedError("Use _arun for async execution")

    async def _arun(self, question: str) -> str:
        try:
            answer = ""
            async for event in self.scoped_agent.ask_stream(question):
                if event.get("type") == "done":
                    answer = event.get("answer", "")

                    # Extract and log tokens if orchestrator_session_id is set
                    if self.orchestrator_session_id:
                        tokens = event.get("tokens_usage", {})
                        total = tokens.get("total", tokens)  # handles both old flat and new breakdown shapes
                        log_token_usage(
                            self.orchestrator_session_id,
                            f"subagent_{self.agent_def['name']}",
                            "invocation",
                            total,
                        )

                    break
            return answer if answer else "(subagent returned no answer)"
        except Exception as e:
            return f"Error from subagent {self.agent_def['name']}: {e}"
