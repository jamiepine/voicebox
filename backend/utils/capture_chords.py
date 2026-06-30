"""Platform defaults for capture hotkey chords."""

from __future__ import annotations

import sys


MAC_PUSH_TO_TALK = ["MetaRight", "AltGr"]
MAC_TOGGLE_TO_TALK = ["MetaRight", "AltGr", "Space"]
NON_MAC_PUSH_TO_TALK = ["ControlRight", "ShiftRight"]
NON_MAC_TOGGLE_TO_TALK = ["ControlRight", "ShiftRight", "Space"]


def default_push_to_talk_chord() -> list[str]:
    if sys.platform == "darwin":
        return MAC_PUSH_TO_TALK.copy()
    return NON_MAC_PUSH_TO_TALK.copy()


def default_toggle_to_talk_chord() -> list[str]:
    if sys.platform == "darwin":
        return MAC_TOGGLE_TO_TALK.copy()
    return NON_MAC_TOGGLE_TO_TALK.copy()
