"""Clarify tool — allows the agent to ask the user multi-choice or open-ended questions.

When the agent encounters ambiguity, insufficient information, or needs user input,
it can call this tool to pause and present a question. The tool's result is included
in the agent's response, and the agent waits for the user's next message as the answer.

This module provides the tool definition and executor. The agent loop handles
interleaving the question with the response text.
"""

from __future__ import annotations

import re
from typing import Any


async def _exec_clarify(args: dict) -> str:
    """Ask the user a question to resolve ambiguity.

    The agent uses this when it needs clarification before proceeding.
    The question is presented to the user, and the user's next message
    is treated as the answer.
    """
    question = args.get("question", "")
    if not question:
        return "[error: 'question' is required]"

    choices = args.get("choices")

    if choices:
        if not isinstance(choices, list) or len(choices) < 2:
            return "[error: 'choices' must be a list with at least 2 items]"
        if len(choices) > 6:
            choices = choices[:6]

        lines = [f"❓ {question}", ""]
        for i, choice in enumerate(choices, 1):
            lines.append(f"{i}. {choice}")
        lines.append("")
        lines.append("(Reply with a number, the text of your choice, or type your own answer)")
        return "\n".join(lines)
    else:
        return f"❓ {question}"


# ── Tool definition ───────────────────────────────────────────────────

CLARIFY_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "clarify",
        "description": (
            "Ask the user a question when you need clarification, feedback, "
            "or a decision before proceeding. Use this when information is "
            "ambiguous, you have multiple valid approaches, or the user's "
            "intent is unclear. Optionally provide up to 6 answer choices. "
            "The tool returns the question which you should present in your response. "
            "Then wait for the user's next message as their answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to present to the user.",
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 6,
                    "description": (
                        "Up to 6 answer choices to offer. When provided, the user "
                        "can pick one or type their own answer. Omit for open-ended questions."
                    ),
                },
            },
            "required": ["question"],
        },
    },
}
