"""Prompt construction and output parsing shared by baselines, training, and serving.

Keeping this in one module guarantees the property called out in the build
plan: training and inference prompts stay structurally identical, and a
malformed model output is always treated as incorrect rather than silently
coerced into a valid label.
"""
from __future__ import annotations

import re
from typing import Sequence

SYSTEM_INSTRUCTION = (
    "You are a banking customer support intent classifier. "
    "Read the customer message and respond with exactly one intent label "
    "from the list below. Respond with the label text only -- no punctuation, "
    "no explanation, nothing else."
)


def format_label_list(labels: Sequence[str]) -> str:
    return "\n".join(f"- {label}" for label in labels)


def build_prompt(
    text: str,
    labels: Sequence[str],
    examples: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Build a classification prompt.

    Zero-shot: examples=None or empty.
    Few-shot: examples is a fixed list of (text, label) pairs, identical
    across every run that claims to be "the few-shot baseline".
    """
    parts = [SYSTEM_INSTRUCTION, "", "Intent labels:", format_label_list(labels), ""]

    if examples:
        parts.append("Examples:")
        for ex_text, ex_label in examples:
            parts.append(f'Message: "{ex_text}"\nIntent: {ex_label}')
        parts.append("")

    parts.append(f'Message: "{text}"')
    parts.append("Intent:")
    return "\n".join(parts)


def build_training_example(text: str, label: str) -> str:
    """Same structural shape as build_prompt(), with the answer appended.

    Used to construct supervised fine-tuning targets so the model is trained
    on exactly the format it will be prompted with at inference time.
    """
    return f'Message: "{text}"\nIntent: {label}'


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize(s: str) -> str:
    return _NORMALIZE_RE.sub("_", s.strip().lower()).strip("_")


def parse_label(raw_output: str, valid_labels: Sequence[str]) -> str | None:
    """Parse a generated string into one of valid_labels, or None if invalid.

    Handles minor formatting noise (case, punctuation, surrounding
    whitespace/quotes) but does not guess: an output that doesn't map
    unambiguously to exactly one valid label is invalid, counted in the
    invalid-output rate rather than scored as a guess.
    """
    if raw_output is None:
        return None

    candidate = raw_output.strip().splitlines()[0] if raw_output.strip() else ""
    candidate = candidate.strip(" \t\"'.:-")
    if not candidate:
        return None

    normalized_candidate = _normalize(candidate)
    normalized_map = {_normalize(label): label for label in valid_labels}

    if normalized_candidate in normalized_map:
        return normalized_map[normalized_candidate]

    # Allow an exact label to appear as a standalone token inside a longer
    # response (e.g. "Intent: card_not_working, thanks") without allowing
    # arbitrary substring matches that could match multiple labels.
    matches = {
        label
        for norm_label, label in normalized_map.items()
        if norm_label and re.search(rf"(?:^|_){re.escape(norm_label)}(?:_|$)", normalized_candidate)
    }
    if len(matches) == 1:
        return next(iter(matches))

    return None
