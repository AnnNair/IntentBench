"""Output parser tests: unknown, punctuated, and case-varied outputs are
handled correctly (build plan section 9.1)."""
from src.prompts import build_prompt, build_training_example, parse_label

LABELS = ["card_not_working", "balance_inquiry", "transfer_money"]


def test_parse_exact_label():
    assert parse_label("card_not_working", LABELS) == "card_not_working"


def test_parse_is_case_insensitive():
    assert parse_label("Card_Not_Working", LABELS) == "card_not_working"


def test_parse_strips_punctuation_and_whitespace():
    assert parse_label('  "balance_inquiry."  ', LABELS) == "balance_inquiry"


def test_parse_takes_only_first_line():
    assert parse_label("transfer_money\nthanks for asking", LABELS) == "transfer_money"


def test_parse_handles_spaced_label():
    assert parse_label("balance inquiry", LABELS) == "balance_inquiry"


def test_parse_unknown_label_is_invalid():
    assert parse_label("not_a_real_label", LABELS) is None


def test_parse_empty_string_is_invalid():
    assert parse_label("", LABELS) is None
    assert parse_label(None, LABELS) is None


def test_parse_ambiguous_output_is_invalid():
    # Contains no clean single-label match -- must not guess.
    assert parse_label("either card_not_working or balance_inquiry", LABELS) is None


def test_prompt_contains_all_labels_and_message():
    prompt = build_prompt("My card isn't working", LABELS)
    for label in LABELS:
        assert label in prompt
    assert "My card isn't working" in prompt


def test_few_shot_prompt_includes_examples():
    examples = [("I need my balance", "balance_inquiry")]
    prompt = build_prompt("send money", LABELS, examples=examples)
    assert "I need my balance" in prompt
    assert "balance_inquiry" in prompt


def test_training_example_matches_prompt_structure():
    training_text = build_training_example("send money", "transfer_money")
    prompt = build_prompt("send money", LABELS)
    # The training target's message line must appear verbatim in the
    # zero-shot prompt structure, so train/inference prompts don't drift.
    assert 'Message: "send money"' in training_text
    assert 'Message: "send money"' in prompt
