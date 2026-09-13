"""Metric tests: known toy predictions -> expected precision/recall/F1
(build plan section 9.1). No model, no GPU."""
from src.metrics import compute_classification_metrics, compute_latency_percentiles, top_confusion_pairs


LABELS = ["card_lost", "balance_inquiry", "transfer_money"]


def test_perfect_predictions_score_one():
    y_true = ["card_lost", "balance_inquiry", "transfer_money"]
    y_pred = ["card_lost", "balance_inquiry", "transfer_money"]

    metrics = compute_classification_metrics(y_true, y_pred, LABELS)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["invalid_output_rate"] == 0.0


def test_all_wrong_predictions_score_zero_accuracy():
    y_true = ["card_lost", "balance_inquiry"]
    y_pred = ["transfer_money", "transfer_money"]

    metrics = compute_classification_metrics(y_true, y_pred, LABELS)

    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] < 1.0


def test_invalid_outputs_are_scored_as_incorrect_and_counted():
    y_true = ["card_lost", "balance_inquiry", "transfer_money"]
    y_pred = ["card_lost", None, "not_a_real_label"]

    metrics = compute_classification_metrics(y_true, y_pred, LABELS)

    assert metrics["invalid_output_rate"] == 2 / 3
    assert metrics["accuracy"] == 1 / 3


def test_known_toy_confusion_matches_hand_computed_precision_recall():
    # card_lost: 2 true, 1 correctly predicted, 1 predicted as balance_inquiry
    # -> precision(card_lost) = 1/1 = 1.0, recall(card_lost) = 1/2 = 0.5
    y_true = ["card_lost", "card_lost", "balance_inquiry"]
    y_pred = ["card_lost", "balance_inquiry", "balance_inquiry"]

    metrics = compute_classification_metrics(y_true, y_pred, LABELS)

    assert metrics["per_class"]["card_lost"]["precision"] == 1.0
    assert metrics["per_class"]["card_lost"]["recall"] == 0.5
    assert metrics["per_class"]["balance_inquiry"]["precision"] == 0.5
    assert metrics["per_class"]["balance_inquiry"]["recall"] == 1.0


def test_top_confusion_pairs_ranks_by_frequency():
    y_true = ["card_lost"] * 3 + ["balance_inquiry"] * 1
    y_pred = ["balance_inquiry"] * 3 + ["transfer_money"] * 1

    pairs = top_confusion_pairs(y_true, y_pred, LABELS, top_n=5)

    assert pairs[0] == {"true": "card_lost", "predicted": "balance_inquiry", "count": 3}


def test_latency_percentiles_on_known_values():
    latencies = [10.0, 20.0, 30.0, 40.0, 100.0]
    result = compute_latency_percentiles(latencies)

    assert result["p50"] == 30.0
    assert result["p95"] > result["p50"]


def test_latency_percentiles_empty_input():
    assert compute_latency_percentiles([]) == {"p50": 0.0, "p95": 0.0}
