from __future__ import annotations

from typing import List, Optional

from .cards import cards_to_text


SYSTEM_PROMPT = (
    "You are a poker agent. Output ONLY valid JSON that matches the schema. "
    "Use only legal actions. action_amount must be 0 for fold/check/call, and within "
    "min/max for bet/raise. memory_summary must be 2-4 short bullets. "
    "Do not add extra keys or commentary."
)


def render_history(actions: List[dict]) -> str:
    if not actions:
        return "(no actions yet)"

    lines: List[str] = []
    current_street = None
    for action in actions:
        street = action["street"]
        if street != current_street:
            lines.append(f"{street}:")
            current_street = street
        lines.append(f"- {action['text']}")
    return "\n".join(lines)


def build_messages(
    *,
    hand_id: str,
    street: str,
    button_seat: int,
    blinds: tuple[int, int],
    starting_stacks: List[int],
    current_stacks: List[int],
    pot: int,
    board_cards: List,
    hole_cards: List,
    history_actions: List[dict],
    to_call: int,
    min_raise: Optional[int],
    max_raise: Optional[int],
    legal_actions: List[str],
    memory_summary: Optional[List[str]] = None,
    correction: Optional[str] = None,
) -> List[dict]:
    board_text = (
        ", ".join(cards_to_text(board_cards)) if board_cards else "none"
    )
    hole_text = ", ".join(cards_to_text(hole_cards)) if hole_cards else "none"

    lines = [
        "STATE:",
        f"hand_id: {hand_id}",
        f"street: {street}",
        f"button_seat: {button_seat} (SB), other seat is BB",
        f"blinds: {blinds[0]}/{blinds[1]}",
        f"starting_stacks: {starting_stacks}",
        f"current_stacks: {current_stacks}",
        f"pot: {pot}",
        f"board: {board_text}",
        f"hole: {hole_text}",
        "",
        "HISTORY:",
        render_history(history_actions),
        "",
        "DECISION:",
        f"to_call: {to_call}",
        f"min_raise: {min_raise}",
        f"max_raise: {max_raise}",
        f"legal_actions: {legal_actions}",
    ]

    if memory_summary is not None:
        lines.extend(["", "MEMORY:"])
        if memory_summary:
            for bullet in memory_summary:
                lines.append(f"- {bullet}")
        else:
            lines.append("(none)")

    if correction:
        lines.extend(["", "CORRECTION:", correction])

    content = "\n".join(lines)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
