from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    parse_error: str | None = None


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        if self.tool_calls:
            return {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": canonical_json(c.arguments)},
                    }
                    for c in self.tool_calls
                ],
            }
        if self.role == "tool":
            return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}
        return {"role": self.role, "content": self.content}

    def render(self) -> str:
        out = f"<|{self.role}|>\n{self.content or ''}\n"
        if self.tool_calls:
            payload = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
            out += f"<|tool_calls|>{canonical_json(payload)}\n"
        if self.tool_call_id:
            out += f"<|tool_call_id|>{self.tool_call_id}\n"
        return out


class History:
    """Append-only message log. Rendered once per append and cached.

    Deliberately exposes no __setitem__, __delitem__, insert, pop, clear or mutable access."""

    def __init__(self) -> None:
        self._msgs: list[Message] = []
        self._rendered: list[str] = []

    def append(self, msg: Message) -> None:
        self._msgs.append(msg)
        self._rendered.append(msg.render())

    def __len__(self) -> int:
        return len(self._msgs)

    def __getitem__(self, i):
        return tuple(self._msgs)[i]

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._msgs)

    def render(self) -> str:
        return "".join(self._rendered)

    def __repr__(self) -> str:
        return f"History({len(self._msgs)} messages)"
