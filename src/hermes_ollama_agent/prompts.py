from __future__ import annotations


def build_system_prompt(available_skills: str) -> str:
    return (
        "You are HermesLocal, a practical software agent.\n"
        "Operating rules:\n"
        "- Be concise and deterministic.\n"
        "- Use tools when they reduce uncertainty.\n"
        "- Keep responses actionable and test-focused.\n"
        "- Respect persistent memory and user profile context.\n"
        "- If you need a missing capability, clearly state what to add.\n\n"
        "Loaded skills:\n"
        f"{available_skills}\n"
    )


def build_user_turn(user_input: str, memory_block: str) -> str:
    return (
        f"{memory_block}\n\n"
        "User request:\n"
        f"{user_input}\n\n"
        "Respond directly and include a short execution plan when needed."
    )
