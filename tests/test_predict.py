import hashlib
import json
import pickle
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn")

from semif_phase1 import predict  # noqa: E402

CALLS = []


def stub_embed(text):
    """Deterministic 'model': the signal lives in whether the text mentions 'crash'."""
    CALLS.append(text)
    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
    return np.concatenate([[1.0 if "crash" in text else -1.0], rng.normal(size=7)]).astype(np.float32)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    crash = rng.random(n) < 0.5
    texts = [f"issue {i} {'crash' if c else 'question'}" for i, c in enumerate(crash)]
    noise = rng.normal(size=(n, 2))
    y = ((crash.astype(float) + 0.3 * rng.normal(size=n)) > 0.5).astype(int)
    return texts, noise, y


def test_encoder_caches_by_text(tmp_path):
    CALLS.clear()
    enc = predict.StateEncoder(embed_fn=stub_embed, cache_dir=str(tmp_path))
    first = enc.transform(["a crash", "a question"])
    second = enc.transform(["a crash", "a question"])
    assert first.shape == (2, 8)
    assert np.array_equal(first, second)
    assert CALLS == ["a crash", "a question"]


def test_encoder_rejects_non_finite():
    enc = predict.StateEncoder(embed_fn=lambda text: np.array([np.nan, 1.0]))
    with pytest.raises(ValueError):
        enc.transform(["x"])


def test_encoder_needs_a_model():
    with pytest.raises(ValueError):
        predict.StateEncoder().transform(["x"])


def test_conformal_quantile_finite_sample():
    scores = np.arange(1, 11, dtype=float)
    assert predict.conformal_quantile(scores, 0.1) == 10.0
    assert predict.conformal_quantile(scores, 0.5) == 6.0
    with pytest.raises(ValueError):
        predict.conformal_quantile(scores, 0.05)


def test_encoder_beats_noise_numerics():
    texts, noise, y = _data()
    tr, te = slice(0, 300), slice(300, None)
    enc = predict.StateEncoder(embed_fn=stub_embed)
    with_state = predict.SemIfPredictor(encoder=enc).fit(texts[tr], None, y[tr])
    noise_only = predict.SemIfPredictor().fit(None, noise[tr], y[tr])
    a = with_state.evaluate(texts[te], None, y[te])
    b = noise_only.evaluate(None, noise[te], y[te])
    assert a["auc"] > 0.8 > b["auc"]
    assert 0.0 <= a["ece"] <= 1.0


def test_classify_conformal_coverage_on_average():
    coverages = []
    for seed in range(8):
        texts, noise, y = _data(n=1200, seed=seed)
        model = predict.SemIfPredictor(encoder=predict.StateEncoder(embed_fn=stub_embed), alpha=0.1)
        model.fit(texts[:800], noise[:800], y[:800])
        coverages.append(model.evaluate(texts[800:], noise[800:], y[800:])["coverage"])
    assert 0.86 <= np.mean(coverages) <= 0.95
    sets = model.predict_set(texts[800:810], noise[800:810])
    assert sets.shape == (10, 2) and sets.any(axis=1).all()


def test_regress_interval_coverage_on_average():
    coverages = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(1000, 3))
        y = x @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.5, size=1000)
        model = predict.SemIfPredictor("regress", alpha=0.2).fit(None, x[:700], y[:700])
        report = model.evaluate(None, x[700:], y[700:])
        coverages.append(report["coverage"])
        assert report["mae"] < 0.6
    assert 0.76 <= np.mean(coverages) <= 0.85
    with pytest.raises(ValueError):
        model.predict_proba(None, x[:2])


def test_bad_inputs():
    with pytest.raises(ValueError):
        predict.SemIfPredictor(task="rank")
    with pytest.raises(ValueError):
        predict.SemIfPredictor().fit(None, None, [0, 1])
    with pytest.raises(ValueError):
        predict.SemIfPredictor().fit(None, np.zeros((50, 1)), np.full(50, 2))


def test_predictor_pickles_without_model_handle(tmp_path):
    texts, noise, y = _data(n=200)
    enc = predict.StateEncoder(embed_fn=stub_embed, cache_dir=str(tmp_path))
    model = predict.SemIfPredictor(encoder=enc).fit(texts, noise, y)
    clone = pickle.loads(pickle.dumps(model))
    assert np.allclose(clone.predict_proba(texts[:5], noise[:5]), model.predict_proba(texts[:5], noise[:5]))


def test_cli_eval_numeric_only_time_split(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(3)
    path = tmp_path / "rows.jsonl"
    with path.open("w") as handle:
        for i in range(300):
            a = float(rng.normal())
            handle.write(json.dumps({"id": i, "as_of": f"2026-01-{1 + i // 12:02d}T{i % 12:02d}:00:00Z",
                                     "a": a, "label": int(a + 0.5 * rng.normal() > 0)}) + "\n")
    monkeypatch.setattr(sys, "argv", ["semif-predict", "eval", "--data", str(path), "--numeric-fields", "a"])
    predict.main()
    report = json.loads(capsys.readouterr().out)
    assert report["train"]["to"] <= report["test"]["from"]
    assert set(report["models"]) == {"base_rate", "numeric"}
    assert report["models"]["numeric"]["auc"] > 0.75
