"""Public, condition-independent action parsing and outcome normalization."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import asdict, dataclass
from typing import List, Optional


ACTION_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*\[(.*)\]\s*$", re.DOTALL)
FAILURE_MARKERS = (
    "too far",
    "occluded",
    "failed to",
    "unexpected failure",
    "cannot",
    "can't",
    "not found",
    "not present",
    "wrong use of api",
    "invalid",
    "timed out",
    "used all",
    "no messages remaining",
    "message blocked",
    "no valid recipients",
    "not delivered",
    "you can only message",
    "you can only send messages",
    "none of your specified recipients are allowed",
)
DEFINITE_NO_EFFECT_MARKERS = (
    "too far", "occluded", "not found", "not present", "wrong use of api",
    "invalid", "message blocked", "no messages remaining", "used all", "timed out",
    "no valid recipients", "not delivered", "you can only message",
    "you can only send messages",
    "none of your specified recipients are allowed",
)


@dataclass(frozen=True)
class CanonicalAction:
    name: str
    arguments: List[str]
    serialized: str
    valid: bool
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ActionOutcome:
    status: str
    success: bool
    had_physical_effect: Optional[bool]
    feedback: str

    def to_dict(self):
        return asdict(self)


def _split_args(raw: str) -> List[str]:
    if not raw.strip():
        return []
    try:
        return [item.strip() for item in next(csv.reader(io.StringIO(raw), skipinitialspace=True))]
    except (csv.Error, StopIteration):
        return [item.strip() for item in raw.split(",")]


def parse_action(action: str) -> CanonicalAction:
    """Parse an action without inventing or resolving any simulator entity ID."""
    match = ACTION_RE.match(str(action or ""))
    if not match:
        return CanonicalAction("", [], str(action or ""), False, "expected Name[arg,...]")
    name, raw = match.groups()
    arguments = _split_args(raw)
    if name == "Place" and len(arguments) not in (3, 5):
        return CanonicalAction(name, arguments, str(action), False, "Place requires 3 or 5 arguments")
    if name == "Communicate":
        if len(arguments) < 2:
            return CanonicalAction(name, arguments, str(action), False, "Communicate requires message and recipient")
        recipient_start = None
        for index in range(1, len(arguments)):
            candidate = arguments[index].strip().lower()
            if candidate == "all" or re.fullmatch(r"agent_\d+", candidate):
                if all(
                    item.strip().lower() == "all"
                    or re.fullmatch(r"agent_\d+", item.strip().lower())
                    for item in arguments[index:]
                ):
                    recipient_start = index
                    break
        if recipient_start is None:
            return CanonicalAction(
                name, arguments, str(action), False,
                "Communicate requires explicit trailing recipient(s): agent_N or all",
            )
        # LLMs sometimes omit the required quotes around a message containing
        # commas.  Rejoin only the fields before explicit trailing recipients;
        # this repairs syntax without inventing a recipient or hidden ID.
        message = ", ".join(arguments[:recipient_start]).strip().strip('"')
        recipients = [item.strip().lower() for item in arguments[recipient_start:]]
        if not message:
            return CanonicalAction(name, arguments, str(action), False, "Communicate message is empty")
        arguments = [message] + recipients
    if name not in {"Wait", "Done"} and not arguments:
        return CanonicalAction(name, arguments, str(action), False, f"{name} requires an argument")
    if name in {"Navigate", "Pick", "Open", "Close"} and len(arguments) != 1:
        return CanonicalAction(name, arguments, str(action), False, f"{name} requires exactly one argument")

    # EnactToM's Place tool requires five fields.  Expanding an already supplied
    # object/relation/receptacle triple is serialization only; no target is guessed.
    if name == "Place" and len(arguments) == 3:
        arguments = arguments + ["None", "None"]
    if name == "Communicate":
        encoded = ", ".join([json.dumps(arguments[0], ensure_ascii=False)] + arguments[1:])
    else:
        encoded = ", ".join(arguments)
    return CanonicalAction(name, arguments, f"{name}[{encoded}]", True)


def classify_outcome(feedback: str, *, action: Optional[CanonicalAction] = None) -> ActionOutcome:
    text = str(feedback or "")
    lower = text.lower()
    if action is not None and not action.valid:
        return ActionOutcome("invalid", False, False, text or action.error or "invalid action")
    if any(marker in lower for marker in FAILURE_MARKERS):
        effect = False if any(marker in lower for marker in DEFINITE_NO_EFFECT_MARKERS) else None
        return ActionOutcome("failed", False, effect, text)
    if not text.strip():
        return ActionOutcome("unknown", False, None, text)
    # Perception, waiting, and communication have no physical effect even when valid.
    nonphysical = bool(action and (
        action.name.startswith("Find") or action.name in {"Wait", "Done", "Communicate"}
    ))
    return ActionOutcome("succeeded", True, False if nonphysical else True, text)


def communication_payload(action: CanonicalAction) -> tuple[str, List[str]]:
    if not action.valid or action.name != "Communicate":
        raise ValueError("not a valid Communicate action")
    message = action.arguments[0].strip().strip('"')
    recipients = [item.strip() for item in action.arguments[1:] if item.strip()]
    return message, recipients
