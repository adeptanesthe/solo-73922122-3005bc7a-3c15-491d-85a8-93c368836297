"""HTTP-level tests for the contribution-analysis endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402
from app.samples import CONTRIBUTION_SAMPLES, SAMPLES  # noqa: E402

client = TestClient(app)


def test_samples_endpoint_exposes_contribution_samples():
    body = client.get("/api/samples").json()
    assert len(body["samples"]) == len(SAMPLES) == 3
    assert len(body["contribution_samples"]) == len(CONTRIBUTION_SAMPLES) == 3
    for sample in body["contribution_samples"]:
        assert 1 <= len(sample["payload"]["validation_tokens"]) <= 32


def test_contribution_samples_all_pass_and_balance():
    effects_seen = set()
    for sample in CONTRIBUTION_SAMPLES:
        resp = client.post("/api/contribution", json=sample["payload"])
        assert resp.status_code == 200, (sample["id"], resp.text)
        report = resp.json()
        assert report["ok"], (sample["id"], report["checks"])
        assert report["residual_abs"] <= 1e-10
        for c in report["contributions"]:
            effects_seen.add(c["effect"])
        # Sum equals total improvement numerically.
        s = sum(c["shapley_contribution"] for c in report["contributions"])
        assert abs(s - report["total_improvement"]) <= 1e-10
    # The bundled samples demonstrate both directions plus a neutral batch.
    assert {"beneficial", "harmful", "neutral"} <= effects_seen


def test_contribution_endpoint_basic_flow():
    payload = {
        "seed": 20240517,
        "scale": 1.0,
        "vocab_size": 256,
        "microbatches": [
            {"name": "align", "tokens": [[4, 4], [7, 7], [9, 9], [4, 4]]},
            {"name": "conflict", "tokens": [[4, 8], [7, 8], [9, 3], [4, 3]]},
        ],
        "validation_tokens": [[4, 4], [7, 7], [9, 9], [4, 7], [7, 4]],
    }
    resp = client.post("/api/contribution", json=payload)
    assert resp.status_code == 200
    report = resp.json()
    assert report["config"]["subset_runs"] == 4
    assert report["config"]["permutations"] == 2
    assert len(report["subset_losses"]) == 4
    # Empty subset must be the baseline and stay unupdated.
    empty = next(r for r in report["subset_losses"] if not r["subset"])
    assert empty["updated"] is False
    assert empty["validation_loss"] == report["baseline"]["validation_loss"]


def test_all_ignored_validation_returns_422_but_training_still_runs():
    payload = {
        "seed": 1,
        "scale": 1.0,
        "vocab_size": 256,
        "microbatches": [{"name": "mb", "tokens": [[1, 2], [3, 4]]}],
        "validation_tokens": [[1, -100], [2, -100]],
    }
    resp = client.post("/api/contribution", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "-100" in detail and "清除旧贡献" in detail

    # The original training comparison endpoint is unaffected.
    run_payload = {k: v for k, v in payload.items() if k != "validation_tokens"}
    run_resp = client.post("/api/run", json=run_payload)
    assert run_resp.status_code == 200
    assert run_resp.json()["ok"]


def test_validation_pair_limit_enforced_over_http():
    payload = {
        "seed": 1,
        "scale": 1.0,
        "vocab_size": 256,
        "microbatches": [{"name": "mb", "tokens": [[1, 2]]}],
        "validation_tokens": [[i % 256, (i + 1) % 256] for i in range(33)],
    }
    resp = client.post("/api/contribution", json=payload)
    assert resp.status_code == 422


def test_missing_validation_tokens_rejected():
    resp = client.post(
        "/api/contribution",
        json={
            "seed": 1, "scale": 1.0, "vocab_size": 256,
            "microbatches": [{"name": "mb", "tokens": [[1, 2]]}],
        },
    )
    assert resp.status_code == 422


def test_index_page_contains_contribution_controls():
    html = client.get("/").text
    for token in (
        "运行贡献分析",
        "验证 token 对",
        "/api/contribution",
        "contrib-card",
        "微批验证贡献",
    ):
        assert token in html, token


def test_run_endpoint_unchanged_without_validation_field():
    payload = {
        "seed": 7, "scale": 1.0, "vocab_size": 256,
        "microbatches": [{"name": "mb", "tokens": [[1, 2], [3, -100]]}],
    }
    resp = client.post("/api/run", json=payload)
    assert resp.status_code == 200 and resp.json()["ok"]
