"""
Tool-calling conversation format for Pebble.

Defines how conversations (with and without tool calls) are serialized
into the token format that Pebble learns during SFT.

Each conversation is a list of messages with roles: system, user, assistant.
Assistant messages may contain tool calls. Tool results follow tool calls.
"""

import json


# Token markers (match the special tokens in our tokenizer)
BOS = "<" + "|bos|" + ">"
EOS = "<" + "|eos|" + ">"
END = "<" + "|end|" + ">"
SYSTEM = "<" + "|system|" + ">"
USER = "<" + "|user|" + ">"
ASSISTANT = "<" + "|assistant|" + ">"
TOOL_CALL_START = "<" + "|tool_call_start|" + ">"
TOOL_CALL_END = "<" + "|tool_call_end|" + ">"
TOOL_RESULT_START = "<" + "|tool_result_start|" + ">"
TOOL_RESULT_END = "<" + "|tool_result_end|" + ">"
FUNC_NAME = "<" + "|function_name|" + ">"
PARAMS = "<" + "|parameters|" + ">"


def format_tool_schema(tools: list[dict]) -> str:
    """
    Format tool definitions for the system prompt.

    Args:
        tools: List of tool dicts with keys: name, description, parameters.
               parameters is a list of dicts with keys: name, type, description.

    Returns:
        Formatted string describing the available tools.
    """
    lines = ["You have access to the following tools:\n"]
    for tool in tools:
        # Build signature: tool_name(param1: type1, param2: type2) -> return_type
        params_str = ", ".join(
            f"{p['name']}: {p['type']}" for p in tool["parameters"]
        )
        ret_type = tool.get("return_type", "str")
        lines.append(f"- {tool['name']}({params_str}) -> {ret_type}")
        lines.append(f"  {tool['description']}")
    lines.append("\nTo call a tool, use the tool_call format. "
                 "Only call a tool when the user's request requires it. "
                 "For normal conversation, respond directly.")
    return "\n".join(lines)


def format_conversation(messages: list[dict], tools: list[dict] | None = None) -> str:
    """
    Format a full conversation into Pebble's token format.

    Args:
        messages: List of message dicts with keys:
            - role: "system", "user", "assistant", "tool_call", "tool_result"
            - content: str (for system/user/assistant)
            - name: str (for tool_call — the function name)
            - arguments: dict (for tool_call)
            - result: str (for tool_result)
        tools: Optional list of tool schemas. If provided and no system message
               exists, a system message is prepended.

    Returns:
        Formatted string ready for tokenization.
    """
    parts = [BOS]

    # If tools provided and first message isn't system, prepend system
    if tools and (not messages or messages[0]["role"] != "system"):
        parts.append(SYSTEM + format_tool_schema(tools) + END)

    for msg in messages:
        role = msg["role"]

        if role == "system":
            parts.append(SYSTEM + msg["content"] + END)

        elif role == "user":
            parts.append(USER + msg["content"] + END)

        elif role == "assistant":
            parts.append(ASSISTANT + msg["content"] + END)

        elif role == "tool_call":
            # Structured tool call
            call_json = json.dumps({
                "name": msg["name"],
                "arguments": msg["arguments"]
            }, separators=(",", ":"))
            parts.append(ASSISTANT + TOOL_CALL_START + call_json + TOOL_CALL_END + END)

        elif role == "tool_result":
            result_str = msg["result"] if isinstance(msg["result"], str) else json.dumps(msg["result"])
            parts.append(TOOL_RESULT_START + result_str + TOOL_RESULT_END)

    parts.append(EOS)
    return "".join(parts)


def format_plain_chat(messages: list[dict]) -> str:
    """
    Format a plain conversation (no tools) into Pebble's token format.
    Used for mixing in non-tool-calling examples during SFT.
    """
    parts = [BOS]
    for msg in messages:
        role = msg["role"]
        if role == "system":
            parts.append(SYSTEM + msg["content"] + END)
        elif role == "user":
            parts.append(USER + msg["content"] + END)
        elif role == "assistant":
            parts.append(ASSISTANT + msg["content"] + END)
    parts.append(EOS)
    return "".join(parts)
