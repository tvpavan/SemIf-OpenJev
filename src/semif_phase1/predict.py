"""Prediction on top of a frozen decision model: hidden state in, calibrated forecast out.

A decision readout asks the model a question and keeps 16 letter probabilities. Prediction
needs something else: the label arrives later, from the world, and history is the teacher.
So this module keeps the representation the model built and hands it to ordinary ML:

    StateEncoder     record text -> last-token hidden state of a frozen GGUF model (cached)
    SemIfPredictor   [state | numerics] -> scikit-learn head -> calibration -> split conformal

Rows are assumed to be in time order. Every split here is chronological: the calibration set
is the tail of training, never a shuffle. Requires the ``predict`` extra (scikit-learn) and,
for a real encoder, the ``llamacpp`` extra.

Installable as ``semif-predict``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class StateEncoder(BaseEstimator, TransformerMixin):
    """Frozen model as a scikit-learn transformer: one prefill per record, no decode.

    ``fit`` does nothing. ``transform`` returns the final-norm hidden state of the last
    token, which is the vector the lm_head would have turned into letter logits. Vectors are
    cached on disk by a hash of (model file, max_tokens, text), so many targets over the same
    records cost one forward pass.

    The context is opened in logits mode with one sequence and switched to embeddings with
    ``llama_set_embeddings``. llama-cpp-python's ``embedding=True`` reserves 256 sequences,
    and on a hybrid DeltaNet model each carries its own recurrent state (about 50 MiB for
    Qwen3.5-4B), which exceeds the Metal working set. ``embed_fn`` replaces the model for tests.
    """

    def __init__(self, model_path: str | None = None, max_tokens: int = 512, n_gpu_layers: int = -1,
                 cache_dir: str | None = None, embed_fn: Callable[[str], np.ndarray] | None = None):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.n_gpu_layers = n_gpu_layers
        self.cache_dir = cache_dir
        self.embed_fn = embed_fn

    def fit(self, X, y=None):
        return self

    def _model_key(self) -> str:
        if self.embed_fn is not None:
            return getattr(self.embed_fn, "__name__", "embed_fn")
        path = Path(self.model_path)
        return f"{path.name}:{path.stat().st_size}"

    def _load(self) -> Callable[[str], np.ndarray]:
        if self.embed_fn is not None:
            return self.embed_fn
        if getattr(self, "_embed", None) is not None:
            return self._embed
        if not self.model_path:
            raise ValueError("StateEncoder needs model_path or embed_fn")
        import llama_cpp

        llm = llama_cpp.Llama(model_path=str(self.model_path), n_ctx=self.max_tokens, n_batch=self.max_tokens,
                              n_ubatch=self.max_tokens, n_gpu_layers=self.n_gpu_layers, verbose=False)
        llama_cpp.llama_set_embeddings(llm.ctx, True)
        width = llm.n_embd()

        def embed(text: str) -> np.ndarray:
            tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)[: self.max_tokens]
            llm.reset()
            llm.eval(tokens)
            pointer = llama_cpp.llama_get_embeddings_ith(llm.ctx, -1)
            if not pointer:
                raise RuntimeError("llama.cpp returned no embedding for the last token")
            return np.ctypeslib.as_array(pointer, shape=(width,)).astype(np.float32).copy()

        self._llm, self._embed = llm, embed
        return embed

    def _cache_path(self, text: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(f"{self._model_key()}|{self.max_tokens}|{text}".encode("utf-8")).hexdigest()
        return Path(self.cache_dir) / digest[:2] / f"{digest}.npy"

    def transform(self, X: Sequence[str]) -> np.ndarray:
        rows = []
        for text in X:
            path = self._cache_path(text)
            if path is not None and path.exists():
                rows.append(np.load(path))
                continue
            vector = np.asarray(self._load()(text), dtype=np.float32)
            if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                raise ValueError("encoder returned a non-finite or non-vector embedding")
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, vector)
            rows.append(vector)
        return np.vstack(rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_llm", None)
        state.pop("_embed", None)
        return state


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold with the finite-sample (n + 1) correction."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise ValueError(f"{n} calibration rows are too few for alpha={alpha}")
    return float(np.sort(scores)[rank - 1])


def binary_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of P(y=1), equal-width bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += abs(y[mask].mean() - p[mask].mean()) * mask.sum() / len(y)
    return float(total)


class SemIfPredictor:
    """[encoder state | numerics] -> tuned linear head -> calibration -> split conformal.

    ``fit`` takes time-ordered rows. The last ``calib_fraction`` of them is held out and the
    head is tuned with TimeSeriesSplit on the rest. For classify, the first half of the tail
    fits a sigmoid calibrator and the second half sets the conformal threshold; for regress,
    the whole tail sets it. Pass ``encoder=None`` for a numeric-only model
    and ``numerics=None`` everywhere for an encoder-only model.
    """

    def __init__(self, task: str = "classify", encoder: StateEncoder | None = None, alpha: float = 0.1,
                 calib_fraction: float = 0.2, folds: int = 5):
        if task not in ("classify", "regress"):
            raise ValueError("task must be 'classify' or 'regress'")
        self.task, self.encoder, self.alpha = task, encoder, alpha
        self.calib_fraction, self.folds = calib_fraction, folds

    def features(self, texts: Sequence[str] | None, numerics) -> np.ndarray:
        parts = []
        if self.encoder is not None:
            if texts is None:
                raise ValueError("this predictor has an encoder and needs texts")
            parts.append(self.encoder.transform(texts))
        if numerics is not None:
            parts.append(np.asarray(numerics, dtype=np.float32).reshape(len(numerics), -1))
        if not parts:
            raise ValueError("no features: give an encoder, numerics, or both")
        return np.hstack(parts)

    def fit(self, texts, numerics, y) -> "SemIfPredictor":
        X, y = self.features(texts, numerics), np.asarray(y)
        cut = int(len(y) * (1 - self.calib_fraction))
        if cut < self.folds + 1 or len(y) - cut < 4:
            raise ValueError("not enough rows for a time-ordered train/calibration split")
        X_fit, y_fit, X_cal, y_cal = X[:cut], y[:cut], X[cut:], y[cut:]
        cv = TimeSeriesSplit(n_splits=self.folds)
        if self.task == "classify":
            if set(np.unique(y)) - {0, 1}:
                raise ValueError("classify expects binary 0/1 labels")
            head = make_pipeline(StandardScaler(), LogisticRegressionCV(
                Cs=np.logspace(-4, 1, 11), cv=cv, scoring="neg_log_loss", max_iter=5000))
            head.fit(X_fit, y_fit)
            # Calibrator and conformal threshold must not share rows, or the scores are optimistic.
            mid = len(y_cal) // 2
            self.model_ = CalibratedClassifierCV(FrozenEstimator(head), method="sigmoid").fit(X_cal[:mid], y_cal[:mid])
            X_conf, y_conf = X_cal[mid:], y_cal[mid:].astype(int)
            p_conf = self.model_.predict_proba(X_conf)
            self.qhat_ = conformal_quantile(1 - p_conf[np.arange(len(y_conf)), y_conf], self.alpha)
        else:
            head = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15), cv=cv))
            self.model_ = head.fit(X_fit, y_fit.astype(float))
            self.qhat_ = conformal_quantile(np.abs(y_cal - self.model_.predict(X_cal)), self.alpha)
        return self

    def predict_proba(self, texts, numerics) -> np.ndarray:
        if self.task != "classify":
            raise ValueError("predict_proba is for task='classify'")
        return self.model_.predict_proba(self.features(texts, numerics))

    def predict(self, texts, numerics) -> np.ndarray:
        return self.model_.predict(self.features(texts, numerics))

    def predict_set(self, texts, numerics) -> np.ndarray:
        """Boolean (n, classes): labels kept at 1 - alpha marginal coverage."""
        return self.predict_proba(texts, numerics) >= 1 - self.qhat_

    def predict_interval(self, texts, numerics) -> np.ndarray:
        if self.task != "regress":
            raise ValueError("predict_interval is for task='regress'")
        center = self.predict(texts, numerics)
        return np.column_stack([center - self.qhat_, center + self.qhat_])

    def evaluate(self, texts, numerics, y) -> dict:
        y = np.asarray(y)
        if self.task == "classify":
            p = self.predict_proba(texts, numerics)
            sets = p >= 1 - self.qhat_
            return {
                "n": int(len(y)), "auc": float(roc_auc_score(y, p[:, 1])),
                "brier": float(brier_score_loss(y, p[:, 1])), "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "ece": binary_ece(y, p[:, 1]),
                "coverage": float(sets[np.arange(len(y)), y.astype(int)].mean()),
                "mean_set_size": float(sets.sum(1).mean()),
            }
        interval = self.predict_interval(texts, numerics)
        return {
            "n": int(len(y)), "mae": float(mean_absolute_error(y, interval.mean(1))),
            "coverage": float(((y >= interval[:, 0]) & (y <= interval[:, 1])).mean()),
            "interval_width": float((interval[:, 1] - interval[:, 0]).mean()),
        }


def base_rate_report(y_train, y_test) -> dict:
    """The forecast to beat: predict the training base rate for everyone."""
    y_test = np.asarray(y_test)
    p = np.full(len(y_test), float(np.mean(y_train)))
    return {"n": int(len(y_test)), "auc": 0.5, "brier": float(brier_score_loss(y_test, p)),
            "log_loss": float(log_loss(y_test, np.column_stack([1 - p, p]), labels=[0, 1])),
            "ece": binary_ece(y_test, p)}


def load_rows(path: Path, time_field: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(time_field not in row for row in rows):
        raise ValueError(f"every row needs '{time_field}'")
    return sorted(rows, key=lambda row: row[time_field])


def _columns(rows, args):
    texts = [row[args.text_field] for row in rows] if args.text_field else None
    fields = [f for f in (args.numeric_fields or "").split(",") if f]
    numerics = np.array([[float(row[f]) for f in fields] for row in rows]) if fields else None
    labels = np.array([row[args.label_field] for row in rows]) if args.label_field in rows[0] else None
    return texts, numerics, labels


def _encoder(args) -> StateEncoder:
    return StateEncoder(model_path=str(args.gguf), max_tokens=args.max_tokens,
                        n_gpu_layers=args.gpu_layers, cache_dir=str(args.cache_dir) if args.cache_dir else None)


def run_eval(args) -> dict:
    rows = load_rows(args.data, args.time_field)
    cut = int(len(rows) * (1 - args.test_fraction))
    train, test = rows[:cut], rows[cut:]
    t_tr, n_tr, y_tr = _columns(train, args)
    t_te, n_te, y_te = _columns(test, args)
    report = {"train": {"n": len(train), "from": train[0][args.time_field], "to": train[-1][args.time_field]},
              "test": {"n": len(test), "from": test[0][args.time_field], "to": test[-1][args.time_field]},
              "models": {}}
    if args.task == "classify":
        report["models"]["base_rate"] = base_rate_report(y_tr, y_te)
    variants = {}
    if n_tr is not None:
        variants["numeric"] = (None, True)
    if args.gguf and t_tr is not None:
        variants["encoder"] = (_encoder(args), False)
        if n_tr is not None:
            variants["encoder+numeric"] = (variants["encoder"][0], True)
    for name, (encoder, use_numeric) in variants.items():
        model = SemIfPredictor(args.task, encoder=encoder, alpha=args.alpha).fit(t_tr, n_tr if use_numeric else None, y_tr)
        report["models"][name] = model.evaluate(t_te, n_te if use_numeric else None, y_te)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "predict", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True, help="JSONL rows")
        p.add_argument("--text-field", help="Field rendered as the record's state text")
        p.add_argument("--numeric-fields", help="Comma-separated numeric fields")
        p.add_argument("--label-field", default="label")
        p.add_argument("--time-field", default="as_of", help="Rows are sorted by this field")
        p.add_argument("--gguf", type=Path, help="Frozen encoder; omit for a numeric-only model")
        p.add_argument("--max-tokens", type=int, default=512)
        p.add_argument("--gpu-layers", type=int, default=-1)
        p.add_argument("--cache-dir", type=Path)
        p.add_argument("--task", choices=("classify", "regress"), default="classify")
        p.add_argument("--alpha", type=float, default=0.1)
        p.add_argument("--model", type=Path, help="Pickle written by fit, read by predict")
        if name == "eval":
            p.add_argument("--test-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if args.command == "eval":
        print(json.dumps(run_eval(args), indent=2))
        return
    if not args.model:
        parser.error("--model is required for fit and predict")
    rows = load_rows(args.data, args.time_field)
    texts, numerics, labels = _columns(rows, args)
    if args.command == "fit":
        encoder = _encoder(args) if args.gguf else None
        model = SemIfPredictor(args.task, encoder=encoder, alpha=args.alpha).fit(texts, numerics, labels)
        args.model.write_bytes(pickle.dumps(model))
        return
    model = pickle.loads(args.model.read_bytes())
    if model.task == "classify":
        probs, sets = model.predict_proba(texts, numerics), model.predict_set(texts, numerics)
        for row, p, s in zip(rows, probs, sets):
            print(json.dumps({"id": row.get("id"), "probabilities": p.tolist(), "set": np.flatnonzero(s).tolist()}))
    else:
        for row, (lo, hi) in zip(rows, model.predict_interval(texts, numerics)):
            print(json.dumps({"id": row.get("id"), "interval": [float(lo), float(hi)]}))


if __name__ == "__main__":
    main()
