"""
Agent tools the LLM can invoke via function calling.
Groq uses the OpenAI-compatible tool schema (different shape from Anthropic's).
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Summarize the retrieved document content into a concise summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What to focus the summary on, e.g. 'key obligations' or 'overall content'",
                    }
                },
                "required": ["focus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_key_dates",
            "description": "Extract important dates, deadlines, or time periods mentioned in the retrieved content.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_risk_clauses",
            "description": "Identify clauses or statements in the retrieved content that could represent risk, liability, or obligations worth flagging.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_tool(tool_name: str, tool_input: dict, context_text: str) -> str:
    """
    Execute a 'tool' — for a hackathon build, each tool just returns instructions
    the LLM will follow, combined with the retrieved context. The actual reasoning
    happens in the follow-up LLM call in main.py.
    """
    if tool_name == "summarize_document":
        focus = tool_input.get("focus", "overall content")
        return f"Summarize the following content, focusing on {focus}:\n\n{context_text}"

    if tool_name == "extract_key_dates":
        return f"Extract all dates, deadlines, and time periods from this content, listed clearly:\n\n{context_text}"

    if tool_name == "flag_risk_clauses":
        return f"Identify any clauses in this content that represent risk, liability, or firm obligations:\n\n{context_text}"

    return f"Unknown tool: {tool_name}"
