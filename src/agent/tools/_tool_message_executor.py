"""Shared helper for executing tool calls and creating tool messages."""

from typing import Any, Mapping

from langchain_core.messages import ToolMessage

from .. import action_log


async def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    tools_by_name: Mapping[str, Any],
    *,
    unknown_tool_prefix: str = "Unknown tool",
) -> list[ToolMessage]:
    """Execute tool calls and return ToolMessages for graph state updates."""
    tool_messages: list[ToolMessage] = []

    for call in tool_calls:
        tool_name = call.get("name", "")
        tool_args = call.get("args", call.get("input", {}))
        tool_call_id = call.get("id", "")

        if tool_name in tools_by_name:
            try:
                result = await tools_by_name[tool_name].ainvoke(tool_args)
            except Exception as e:
                result = f"Error: {e}"
        else:
            result = f"{unknown_tool_prefix}: {tool_name}"

        # Files with invalid UTF-8 can leave lone surrogates in the text, which the API request
        # cannot encode; replace them so one odd file does not end the issue run.
        content = str(result).encode("utf-8", "replace").decode("utf-8")
        action_log.record(tool_name, tool_args, content)
        tool_messages.append(
            ToolMessage(content=content, tool_call_id=tool_call_id)
        )

    return tool_messages
