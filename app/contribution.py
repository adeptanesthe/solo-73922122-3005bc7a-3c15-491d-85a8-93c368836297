"""Exact-Shapley validation contribution analysis (CPU, float64).

For every subset of the training micro-batches this module performs the same
single SGD update the training engine would perform, but every candidate run
starts from the *same* initial shared weight ``W`` (the seed/scale/vocab are
part of the reused training configuration, so ``make_weight`` reproduces the
identical matrix bit-for-bit).  An empty subset, or a subset whose micro-batches
all have every target ignored (``-100``), keeps the original ``W`` untouched.

Each candidate weight is scored on the validation token pairs: the mean cross
entropy over the valid validation targets.  The validation loss of the original
``W`` is the baseline.  A micro-batch's exact Shapley contribution is the mean
of its marginal improvement ``loss(S) - loss(S + {i})`` over every insertion
order (all ``n!`` permutations).  The contributions therefore sum exactly to
the full-training improvement ``loss(empty) - loss(all)``.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Tuple

import torch

from .engine import (
    EMBED_DIM,
    IGNORE_INDEX,
    LEARNING_RATE,
    MAX_MICROBATCHES,
    MicroBatch,
    RunSpec,
    TOLERANCE,
    _as_int,
    _flatten,
    _max_abs_diff,
    _require,
    dense_reference,
    make_weight,
    parse_payload,
    run_chunked,
)

MIN_VALIDATION_PAIRS = 1
MAX_VALIDATION_PAIRS = 32
SHAPLEY_SUM_TOLERANCE = 1e-10
SHAPLEY_FORMULA_TOLERANCE = 1e-10
NEUTRAL_TOLERANCE = 0.0  # only exactly-zero contributions (empty/ignored batch)


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------
def parse_validation_pairs(
    payload: dict, vocab_size: int
) -> Tuple[Tuple[Tuple[int, int], ...], int]:
    """Validate the ``validation_tokens`` field against the training vocab."""
    raw = payload.get("validation_tokens")
    _require(
        isinstance(raw, list) and len(raw) >= MIN_VALIDATION_PAIRS,
        "至少需要 1 组验证 token 对（validation_tokens 不能为空）",
    )
    _require(
        len(raw) <= MAX_VALIDATION_PAIRS,
        f"验证 token 对数量不能超过 {MAX_VALIDATION_PAIRS} 组",
    )
    pairs: List[Tuple[int, int]] = []
    for idx, pair in enumerate(raw):
        _require(
            isinstance(pair, (list, tuple)) and len(pair) == 2,
            f"第 {idx + 1} 组验证 token 必须是 [输入号, 目标号]",
        )
        inp = _as_int("验证输入 token 号", pair[0])
        tgt = _as_int("验证目标 token 号", pair[1])
        _require(
            0 <= inp < vocab_size,
            f"验证第 {idx + 1} 组: 输入号 {inp} 超出词表 [0, {vocab_size})",
        )
        _require(
            tgt == IGNORE_INDEX or 0 <= tgt < vocab_size,
            f"验证第 {idx + 1} 组: 目标号 {tgt} 必须为 {IGNORE_INDEX} "
            f"或位于 [0, {vocab_size})",
        )
        pairs.append((inp, tgt))

    valid_count = sum(1 for _, tgt in pairs if tgt != IGNORE_INDEX)
    _require(
        valid_count >= 1,
        "验证目标全部为 -100：验证损失无定义，已拒绝本次贡献分析并清除旧贡献"
        "（原训练对照仍可运行）",
    )
    return tuple(pairs), valid_count


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def validation_cross_entropy(
    weight: torch.Tensor, pairs: Sequence[Tuple[int, int]]
) -> float:
    """Mean cross entropy of ``weight`` over the valid validation token pairs."""
    idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    tgt = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    valid = tgt != IGNORE_INDEX
    hidden = weight.index_select(0, idx)
    logits = hidden @ weight.t()  # dense [pairs, vocab]; validation set is tiny
    log_probs = torch.log_softmax(logits, dim=1)
    picked = log_probs.gather(1, tgt.clamp_min(0).unsqueeze(1)).squeeze(1)
    return float((-picked[valid].mean()).item())


def _subset_has_valid_target(batches: Sequence[MicroBatch]) -> bool:
    return any(
        tgt != IGNORE_INDEX for mb in batches for tgt in mb.target_ids
    )


def candidate_weight(
    base_weight: torch.Tensor, spec: RunSpec, indices: Sequence[int]
) -> torch.Tensor:
    """Weight obtained by one original SGD step on the chosen micro-batches.

    Empty subsets and fully-ignored subsets return the original ``W``.  Every
    non-trivial run starts from the same initial shared weight because the
    configuration reuses the same seed/scale/vocab.
    """
    if not indices:
        return base_weight.detach().clone()
    batches = [spec.microbatches[i] for i in indices]
    if not _subset_has_valid_target(batches):
        return base_weight.detach().clone()
    sub_spec = RunSpec(
        microbatches=list(batches),
        vocab_size=spec.vocab_size,
        seed=spec.seed,
        scale=spec.scale,
    )
    result = run_chunked(sub_spec)
    # The candidate run must start from the identical shared initial W.
    if not torch.equal(result.base_weight, base_weight):
        raise RuntimeError("内部错误：候选 SGD 更新的初始权重与共享 W 不一致")
    return result.updated_weight


def shapley_by_permutations(
    losses: Dict[int, float], n: int
) -> List[float]:
    """Mean marginal improvement over all ``n!`` insertion orders."""
    sums = [0.0] * n
    for perm in itertools.permutations(range(n)):
        prev_mask = 0
        for i in perm:
            new_mask = prev_mask | (1 << i)
            # Improvement = loss before - loss after adding micro-batch i.
            sums[i] += losses[prev_mask] - losses[new_mask]
            prev_mask = new_mask
    return [s / math.factorial(n) for s in sums]


def shapley_by_weighted_subsets(
    losses: Dict[int, float], n: int
) -> List[float]:
    """Independent Shapley formula via weighted coalition values (cross-check)."""
    values: List[float] = []
    for i in range(n):
        total = 0.0
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            coalition_size = mask.bit_count()
            weight = (
                math.factorial(coalition_size)
                * math.factorial(n - coalition_size - 1)
                / math.factorial(n)
            )
            total += weight * (losses[mask] - losses[mask | (1 << i)])
        values.append(total)
    return values


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------
def analyze_contribution(payload: dict) -> dict:
    _require(isinstance(payload, dict), "请求体必须是 JSON 对象")

    # Reuse the existing training-config validation unchanged.  The only extra
    # field allowed on top of the training payload is validation_tokens.
    training_payload = {
        key: value for key, value in payload.items() if key != "validation_tokens"
    }
    spec = parse_payload(training_payload)
    validation_pairs, validation_valid = parse_validation_pairs(
        payload, spec.vocab_size
    )

    n = len(spec.microbatches)
    base_weight = make_weight(spec.vocab_size, spec.seed, spec.scale)

    # One candidate W per subset (2^n), then validation loss of each candidate.
    losses: Dict[int, float] = {}
    subset_rows: List[dict] = []
    full_weight: torch.Tensor | None = None
    full_training_loss: float | None = None
    full_mask = (1 << n) - 1
    for mask in range(1 << n):
        indices = [i for i in range(n) if mask & (1 << i)]
        chosen = [spec.microbatches[i] for i in indices]
        if mask == full_mask:
            # The full coalition is exactly the original training run; record
            # its training loss while we are at it.
            full_result = run_chunked(spec)
            weight = full_result.updated_weight
            full_weight = weight
            full_training_loss = full_result.loss
            if not torch.equal(full_result.base_weight, base_weight):
                raise RuntimeError("完整训练的初始权重与共享 W 不一致")
        else:
            weight = candidate_weight(base_weight, spec, indices)
        losses[mask] = validation_cross_entropy(weight, validation_pairs)
        subset_rows.append(
            {
                "subset": [mb.name for mb in chosen],
                "updated": bool(indices) and _subset_has_valid_target(chosen),
                "validation_loss": losses[mask],
            }
        )

    baseline_loss = losses[0]
    full_loss = losses[(1 << n) - 1]
    total_improvement = baseline_loss - full_loss

    phi_perm = shapley_by_permutations(losses, n)
    phi_formula = shapley_by_weighted_subsets(losses, n)
    formula_diff = max(
        abs(a - b) for a, b in zip(phi_perm, phi_formula)
    ) if n else 0.0

    sum_contributions = sum(phi_perm)
    residual = abs(sum_contributions - total_improvement)

    # Independent dense reference for the full-training SGD update.
    all_inputs, all_targets = _flatten(spec)
    ref_weight = make_weight(spec.vocab_size, spec.seed, spec.scale)
    _, ref_grad = dense_reference(ref_weight, all_inputs, all_targets)
    ref_updated = ref_weight - LEARNING_RATE * ref_grad
    assert full_weight is not None
    dense_update_diff = _max_abs_diff(full_weight, ref_updated)

    contributions = []
    for i, mb in enumerate(spec.microbatches):
        value = phi_perm[i]
        if value > NEUTRAL_TOLERANCE:
            effect = "beneficial"
            effect_label = "有益"
        elif value < -NEUTRAL_TOLERANCE:
            effect = "harmful"
            effect_label = "有害"
        else:
            effect = "neutral"
            effect_label = "无影响（空/全忽略）"
        contributions.append(
            {
                "microbatch": mb.name,
                "training_valid_tokens": sum(
                    1 for tgt in mb.target_ids if tgt != IGNORE_INDEX
                ),
                "shapley_contribution": value,
                "weighted_subset_cross_check": phi_formula[i],
                "average_marginal_improvement": value,
                "effect": effect,
                "effect_label": effect_label,
            }
        )

    checks = {
        "shapley_sum_matches_improvement": residual <= SHAPLEY_SUM_TOLERANCE,
        "shapley_formula_cross_check": formula_diff <= SHAPLEY_FORMULA_TOLERANCE,
        "full_update_vs_dense": dense_update_diff <= TOLERANCE,
    }
    ok = all(checks.values())

    return {
        "ok": ok,
        "message": (
            "贡献分析完成：各微批 Shapley 贡献之和与完整训练总改善一致"
            if ok
            else "贡献分析未通过内部一致性校验"
        ),
        "tolerance": {
            "shapley_sum": SHAPLEY_SUM_TOLERANCE,
            "shapley_formula": SHAPLEY_FORMULA_TOLERANCE,
            "dense_update": TOLERANCE,
        },
        "config": {
            "vocab_size": spec.vocab_size,
            "embed_dim": EMBED_DIM,
            "seed": spec.seed,
            "scale": spec.scale,
            "learning_rate": LEARNING_RATE,
            "microbatches": n,
            "max_microbatches": MAX_MICROBATCHES,
            "training_total_tokens": len(all_inputs),
            "training_valid_tokens": sum(
                1 for tgt in all_targets if tgt != IGNORE_INDEX
            ),
            "validation_pairs": len(validation_pairs),
            "validation_valid_tokens": validation_valid,
            "permutations": math.factorial(n),
            "subset_runs": 1 << n,
        },
        "baseline": {
            "validation_loss": baseline_loss,
            "weight": "原始共享权重 W（未做 SGD 更新）",
        },
        "full_training": {
            "validation_loss": full_loss,
            "training_loss": full_training_loss,
            "improvement_vs_baseline": total_improvement,
        },
        "total_improvement": total_improvement,
        "sum_of_contributions": sum_contributions,
        "residual_abs": residual,
        "contributions": contributions,
        "subset_losses": subset_rows,
        "checks": checks,
        "errors": {
            "shapley_formula_max_abs_diff": formula_diff,
            "full_update_vs_dense_max_abs_diff": dense_update_diff,
        },
        "method": {
            "game": "验证集有效 token 平均交叉熵（越低越好），改善量 = 损失下降",
            "candidate_update": (
                "每个微批子集从同一初始共享 W 做一次原有 SGD 更新"
                "（lr=0.01，子集内按全部有效 token 平均）"
            ),
            "empty_or_ignored_subset": "保持原始 W，不做更新",
            "shapley": "所有微批加入顺序（全排列）的平均边际改善量",
        },
    }
