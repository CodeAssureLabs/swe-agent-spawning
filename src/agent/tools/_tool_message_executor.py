"""Shared helper for executing tool calls and creating tool messages."""

from typing import Any, Mapping

from langchain_core.messages import ToolMessage


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

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call_id)
        )

    return tool_messages
