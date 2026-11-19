"""Per-issue log of every tool call made by any agent (coordinator, domain agents, and domain
agents during registry initialization), with its arguments and output.

run_eval.py starts a log for each issue (and for registry initialization) and stores it in the
report, so that an audit can see what every agent did, including the domain agents, whose
actions are otherwise visible only through their answers.
"""
from contextvars import ContextVar

_ACTIONS: ContextVar = ContextVar("agent_action_log", default=None)
_AGENT: ContextVar = ContextVar("agent_action_agent", default="coordinator")
MAX_OUTPUT_CHARS = 200_000


def start() -> list:
    log: list = []
    _ACTIONS.set(log)
    return log


def set_agent(name: str):
    return _AGENT.set(name)


def reset_agent(token) -> None:
    _AGENT.reset(token)


def record(tool: str, args, output: str) -> None:
    log = _ACTIONS.get()
    if log is None:
        return
    log.append({"agent": _AGENT.get(), "tool": tool, "args": args,
                "output": output[:MAX_OUTPUT_CHARS], "output_chars": len(output)})
