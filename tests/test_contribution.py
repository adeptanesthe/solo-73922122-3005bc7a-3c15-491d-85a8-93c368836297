"""Tests for exact-Shapley validation contribution analysis."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contribution import (  # noqa: E402
    MAX_VALIDATION_PAIRS,
    SHAPLEY_SUM_TOLERANCE,
    analyze_contribution,
    candidate_weight,
    parse_validation_pairs,
    shapley_by_permutations,
    shapley_by_weighted_subsets,
    validation_cross_entropy,
)
from app.engine import (  # noqa: E402
    IGNORE_INDEX,
    ValidationError,
    make_weight,
    parse_payload,
)


def _payload(microbatches, validation, seed=20240517, scale=1.0, vocab=256):
    return {
        "seed": seed,
        "scale": scale,
        "vocab_size": vocab,
        "microbatches": microbatches,
        "validation_tokens": validation,
    }


BENIGN_HARMFUL = _payload(
    microbatches=[
        {"name": "align", "tokens": [[4, 4], [7, 7], [9, 9], [4, 4]]},
        {"name": "conflict", "tokens": [[4, 8], [7, 8], [9, 3], [4, 3]]},
    ],
    validation=[[4, 4], [7, 7], [9, 9], [4, 7], [7, 4]],
)


def test_basic_report_structure_and_baseline():
    report = analyze_contribution(BENIGN_HARMFUL)
    assert report["ok"]
    cfg = report["config"]
    assert cfg["microbatches"] == 2
    assert cfg["permutations"] == 2
    assert cfg["subset_runs"] == 4
    assert cfg["validation_pairs"] == 5
    assert cfg["validation_valid_tokens"] == 5
    # One positive and one negative contribution.
    effects = {c["effect"] for c in report["contributions"]}
    assert "beneficial" in effects and "harmful" in effects
    labels = {c["effect_label"] for c in report["contributions"]}
    assert {"有益", "有害"} <= labels


def test_shapley_sum_equals_total_improvement_exactly():
    report = analyze_contribution(BENIGN_HARMFUL)
    assert report["residual_abs"] <= SHAPLEY_SUM_TOLERANCE
    assert report["checks"]["shapley_sum_matches_improvement"]
    # Total improvement is defined against the untouched-W baseline.
    expected = (
        report["baseline"]["validation_loss"]
        - report["full_training"]["validation_loss"]
    )
    assert report["total_improvement"] == pytest.approx(expected, abs=1e-14)
    assert report["total_improvement"] == pytest.approx(
        report["sum_of_contributions"], abs=1e-12
    )


def test_independent_permutation_enumeration_matches_report():
    """Recompute Shapley values from the subset losses with a fresh loop."""
    report = analyze_contribution(BENIGN_HARMFUL)
    names = [c["microbatch"] for c in report["contributions"]]
    # Report keys are micro-batch names; translate index coalitions to names.
    losses = {}
    for row in report["subset_losses"]:
        idx = tuple(sorted(names.index(nm) for nm in row["subset"]))
        losses[idx] = row["validation_loss"]
    n = len(names)
    phi = [0.0] * n
    for perm in itertools.permutations(range(n)):
        prev: tuple = ()
        for i in perm:
            cur = tuple(sorted(prev + (i,)))
            phi[i] += losses[prev] - losses[cur]
            prev = cur
    phi = [v / math.factorial(n) for v in phi]
    for i, c in enumerate(report["contributions"]):
        assert c["shapley_contribution"] == pytest.approx(phi[i], abs=1e-12)


def test_weighted_subset_formula_cross_check():
    report = analyze_contribution(BENIGN_HARMFUL)
    assert report["checks"]["shapley_formula_cross_check"]
    for c in report["contributions"]:
        assert c["shapley_contribution"] == pytest.approx(
            c["weighted_subset_cross_check"], abs=1e-12
        )
    assert report["errors"]["shapley_formula_max_abs_diff"] <= 1e-12


def test_permutation_helper_equals_weighted_subset_helper():
    losses = {0: 5.0, 1: 4.0, 2: 4.5, 3: 3.2}
    p = shapley_by_permutations(losses, 2)
    w = shapley_by_weighted_subsets(losses, 2)
    assert p == pytest.approx(w, abs=1e-14)
    assert sum(p) == pytest.approx(losses[0] - losses[3], abs=1e-14)


def test_every_candidate_starts_from_same_initial_weight():
    spec = parse_payload(
        {k: v for k, v in BENIGN_HARMFUL.items() if k != "validation_tokens"}
    )
    base = make_weight(spec.vocab_size, spec.seed, spec.scale)
    for mask in range(1 << len(spec.microbatches)):
        indices = [i for i in range(len(spec.microbatches)) if mask & (1 << i)]
        w = candidate_weight(base, spec, indices)
        assert w.shape == base.shape and w.dtype == torch.float64
        # Candidates are deterministically reproducible.
        again = candidate_weight(base.clone(), spec, indices)
        assert torch.equal(w, again)


def test_empty_and_all_ignored_subsets_keep_original_weight():
    payload = _payload(
        microbatches=[
            {"name": "real", "tokens": [[1, 2], [3, 4]]},
            {"name": "ignored", "tokens": [[5, -100], [6, -100]]},
        ],
        validation=[[1, 2], [3, 4], [5, 6]],
    )
    report = analyze_contribution(payload)
    # The all-ignored micro-batch has exactly zero Shapley contribution.
    by_name = {c["microbatch"]: c for c in report["contributions"]}
    assert by_name["ignored"]["shapley_contribution"] == 0.0
    assert by_name["ignored"]["effect"] == "neutral"
    # Subsets {ignored} and the empty set both keep W -> identical val loss.
    losses = {
        tuple(row["subset"]): row for row in report["subset_losses"]
    }
    assert losses[()]["updated"] is False
    assert losses[("ignored",)]["updated"] is False
    assert (
        losses[()]["validation_loss"]
        == losses[("ignored",)]["validation_loss"]
    )
    assert report["ok"]


def test_validation_loss_ignores_minus_100_targets():
    weight = make_weight(256, 1, 1.0)
    a = validation_cross_entropy(weight, [[1, 2], [3, 4]])
    b = validation_cross_entropy(
        weight, [[1, 2], [9, IGNORE_INDEX], [3, 4], [8, IGNORE_INDEX]]
    )
    # Same valid rows -> same mean up to float64 summation-order noise.
    assert a == pytest.approx(b, abs=1e-12)


def test_all_validation_targets_ignored_is_rejected():
    payload = _payload(
        microbatches=[{"name": "mb", "tokens": [[1, 2], [3, 4]]}],
        validation=[[1, IGNORE_INDEX], [2, IGNORE_INDEX]],
    )
    with pytest.raises(ValidationError, match="验证目标全部为 -100"):
        analyze_contribution(payload)
    # Parser level with explicit message about clearing stale contributions.
    with pytest.raises(ValidationError, match="清除旧贡献"):
        parse_validation_pairs(
            {"validation_tokens": [[1, -100]]}, vocab_size=256
        )


def test_training_config_is_reused_and_dense_update_matches():
    report = analyze_contribution(BENIGN_HARMFUL)
    assert report["config"]["seed"] == BENIGN_HARMFUL["seed"]
    assert report["config"]["learning_rate"] == 0.01
    assert report["checks"]["full_update_vs_dense"]
    assert report["errors"]["full_update_vs_dense_max_abs_diff"] <= 1e-8


def test_validation_pair_count_bounds():
    base = dict(BENIGN_HARMFUL)
    with pytest.raises(ValidationError):
        parse_validation_pairs({"validation_tokens": []}, vocab_size=256)
    too_many = {"validation_tokens": [[i % 256, (i + 1) % 256]
                                      for i in range(MAX_VALIDATION_PAIRS + 1)]}
    with pytest.raises(ValidationError, match="不能超过"):
        parse_validation_pairs(too_many, vocab_size=256)


def test_validation_pair_range_validation():
    bad_cases = [
        {"validation_tokens": [[256, 1]]},
        {"validation_tokens": [[1, 256]]},
        {"validation_tokens": [[-1, 1]]},
        {"validation_tokens": [[1, -99]]},
        {"validation_tokens": [[1]]},
        {"validation_tokens": "nope"},
    ]
    for payload in bad_cases:
        with pytest.raises(ValidationError):
            parse_validation_pairs(payload, vocab_size=256)


def test_full_training_rejected_when_all_training_targets_ignored():
    payload = _payload(
        microbatches=[{"name": "mb", "tokens": [[1, -100], [2, -100]]}],
        validation=[[1, 2]],
    )
    with pytest.raises(ValidationError):
        analyze_contribution(payload)


def test_four_microbatches_sixteen_subsets_twenty_four_permutations():
    payload = _payload(
        microbatches=[
            {"name": f"mb{i}", "tokens": [[i, i + 1], [i + 4, i + 8]]}
            for i in range(4)
        ],
        validation=[[0, 1], [1, 2], [2, 3], [3, 4]],
        seed=424242,
        vocab=512,
    )
    report = analyze_contribution(payload)
    assert report["ok"]
    assert report["config"]["subset_runs"] == 16
    assert report["config"]["permutations"] == 24
    assert len(report["subset_losses"]) == 16
    assert len(report["contributions"]) == 4
    assert report["residual_abs"] <= SHAPLEY_SUM_TOLERANCE


def test_ignored_validation_tokens_still_count_toward_pair_limit():
    pairs = [[i % 256, IGNORE_INDEX] for i in range(5)] + [[1, 2]]
    parsed, valid = parse_validation_pairs(
        {"validation_tokens": pairs}, vocab_size=256
    )
    assert len(parsed) == 6 and valid == 1


def test_large_scores_stay_accurate_in_contribution():
    payload = _payload(
        microbatches=[
            {"name": "a", "tokens": [[1, 4095], [2048, 0], [4095, 4095]]},
            {"name": "b", "tokens": [[255, 256], [256, 255], [1024, 33]]},
        ],
        validation=[[1, 4095], [255, 256], [4095, 4095]],
        seed=424242,
        scale=100.0,
        vocab=4096,
    )
    report = analyze_contribution(payload)
    assert report["ok"]
    assert all(
        math.isfinite(c["shapley_contribution"])
        for c in report["contributions"]
    )
    assert report["errors"]["full_update_vs_dense_max_abs_diff"] <= 1e-8


def test_unknown_field_still_rejected():
    payload = dict(BENIGN_HARMFUL)
    payload["bogus"] = 1
    with pytest.raises(ValidationError):
        analyze_contribution(payload)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
