"""Leakage-safe paired trimodal survival benchmark.

The benchmark compares each modality with two risk-level fusion baselines on
the exact same cohort and outer 80/20 splits. Every imputer, scaler, PCA and
Cox model is fitted inside the training partition. Fusion operates on
cross-fitted risk percentiles, avoiding the 2307-dimensional raw
concatenation used by the original Phase 5 baseline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
MODALITIES = ("tabular", "text", "vision")


def cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index(time, -risk, event))


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map risk to [0, 1] using only a model's training-risk distribution."""
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    return np.searchsorted(reference, values, side="right") / (len(reference) + 1.0)


def load_vision_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "case_id" not in frame:
        raise ValueError("Vision CSV requires case_id")
    z_cols = [f"z{i:03d}" for i in range(768)]
    if not set(z_cols).issubset(frame.columns):
        # Accept the raw 512D export from the Colab notebook as well.
        f_cols = sorted(c for c in frame.columns if c.startswith("f") and c[1:].isdigit())
        if len(f_cols) < 2:
            raise ValueError("Vision CSV needs z000..z767 or numeric f* feature columns")
        value_cols = f_cols
    else:
        value_cols = z_cols
    frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    if frame["case_id"].duplicated().any():
        raise ValueError("Vision CSV contains duplicate case_id values")
    result = frame.set_index("case_id")[value_cols].astype(np.float32)
    finite = np.isfinite(result.to_numpy()).all(axis=1)
    nonzero = np.linalg.norm(result.to_numpy(), axis=1) > 0
    return result.loc[finite & nonzero]


def load_text_npz(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as cache:
        ids = [str(value).strip().upper() for value in cache["case_ids"]]
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        confidence = np.asarray(cache["confidence"], dtype=np.float32)
    if len(ids) != len(embeddings) or len(confidence) != len(embeddings):
        raise ValueError("Text NPZ arrays are not aligned")
    finite = np.isfinite(embeddings).all(axis=1)
    valid = finite & (np.linalg.norm(embeddings, axis=1) > 0) & (confidence > 0)
    frame = pd.DataFrame(embeddings[valid], index=np.asarray(ids)[valid])
    if frame.index.duplicated().any():
        raise ValueError("Text NPZ contains duplicate case IDs")
    return frame


def load_indexed_csv(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    if frame["case_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate case IDs")
    return frame.set_index("case_id")


class TabularTransform:
    def fit(self, values: np.ndarray) -> "TabularTransform":
        values = np.asarray(values, dtype=np.float64)
        self.keep = ~np.isnan(values).all(axis=0)
        kept = values[:, self.keep]
        self.medians = np.nanmedian(kept, axis=0)
        self.indicator_cols = np.isnan(kept).any(axis=0)
        transformed = self._impute_and_indicate(kept)
        self.scaler = StandardScaler().fit(transformed)
        return self

    def _impute_and_indicate(self, kept: np.ndarray) -> np.ndarray:
        missing = np.isnan(kept)
        imputed = np.where(missing, self.medians[None, :], kept)
        if self.indicator_cols.any():
            imputed = np.concatenate(
                [imputed, missing[:, self.indicator_cols].astype(np.float64)], axis=1
            )
        return imputed

    def transform(self, values: np.ndarray) -> np.ndarray:
        kept = np.asarray(values, dtype=np.float64)[:, self.keep]
        return self.scaler.transform(self._impute_and_indicate(kept))


class EmbeddingTransform:
    def __init__(self, n_components: int, seed: int):
        self.n_components = int(n_components)
        self.seed = int(seed)

    def fit(self, values: np.ndarray) -> "EmbeddingTransform":
        values = np.asarray(values, dtype=np.float64)
        self.nonconstant = np.nanstd(values, axis=0) > 1e-10
        kept = values[:, self.nonconstant]
        self.scaler = StandardScaler().fit(kept)
        scaled = self.scaler.transform(kept)
        n_components = min(self.n_components, len(values) - 1, scaled.shape[1])
        if n_components < 1:
            raise ValueError("No nonconstant embedding dimensions remain")
        self.pca = PCA(
            n_components=n_components, svd_solver="randomized", random_state=self.seed
        ).fit(scaled)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        kept = np.asarray(values, dtype=np.float64)[:, self.nonconstant]
        return self.pca.transform(self.scaler.transform(kept))


@dataclass(frozen=True)
class ModelSpec:
    pca_dim: int | None
    penalizer: float


def make_transform(kind: str, spec: ModelSpec, seed: int):
    if kind == "tabular":
        return TabularTransform()
    if spec.pca_dim is None:
        raise ValueError("Embedding models require pca_dim")
    return EmbeddingTransform(spec.pca_dim, seed)


def fit_cox(x: np.ndarray, time: np.ndarray, event: np.ndarray, penalizer: float):
    columns = [f"x{i:03d}" for i in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=columns)
    frame["time"] = time
    frame["event"] = event
    model = CoxPHFitter(penalizer=float(penalizer), l1_ratio=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, duration_col="time", event_col="event", show_progress=False)
    return model, columns


def predict_cox(model, columns: list[str], values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values, columns=columns)
    return model.predict_log_partial_hazard(frame).to_numpy(dtype=np.float64)


def fit_predict(
    values: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    kind: str,
    spec: ModelSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    transform = make_transform(kind, spec, seed).fit(values[train_idx])
    x_train = transform.transform(values[train_idx])
    x_test = transform.transform(values[test_idx])
    keep = np.isfinite(x_train).all(axis=0) & (np.std(x_train, axis=0) > 1e-8)
    if not keep.any():
        raise ValueError("No finite nonconstant model features remain")
    x_train = x_train[:, keep]
    x_test = x_test[:, keep]
    model, columns = fit_cox(x_train, time[train_idx], event[train_idx], spec.penalizer)
    train_risk = predict_cox(model, columns, x_train)
    test_risk = predict_cox(model, columns, x_test)
    return train_risk, test_risk


def candidate_specs(kind: str, pca_dims: list[int], penalties: list[float]):
    dims = [None] if kind == "tabular" else pca_dims
    return [ModelSpec(dim, penalty) for dim in dims for penalty in penalties]


def select_spec_and_oof(
    values: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    kind: str,
    seed: int,
    pca_dims: list[int],
    penalties: list[float],
    n_folds: int,
) -> tuple[ModelSpec, np.ndarray, float]:
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(event)), event))
    candidates = []
    errors = []
    for spec in candidate_specs(kind, pca_dims, penalties):
        oof = np.full(len(event), np.nan, dtype=np.float64)
        fold_scores = []
        failed = False
        for fold, (train_idx, val_idx) in enumerate(splits):
            try:
                train_risk, val_risk = fit_predict(
                    values, time, event, train_idx, val_idx, kind, spec,
                    seed * 100 + fold,
                )
                oof[val_idx] = empirical_percentile(train_risk, val_risk)
                fold_scores.append(cindex(time[val_idx], event[val_idx], oof[val_idx]))
            except Exception as exc:
                errors.append(f"{spec} fold={fold}: {type(exc).__name__}: {exc}")
                failed = True
                break
        if failed or np.isnan(oof).any():
            continue
        candidates.append((float(np.mean(fold_scores)), spec, oof))
    if not candidates:
        detail = errors[-1] if errors else "no candidate diagnostics"
        raise RuntimeError(f"All {kind} model candidates failed; last error: {detail}")
    # Prefer a smaller PCA and stronger regularization when scores tie.
    candidates.sort(
        key=lambda item: (
            item[0],
            -(item[1].pca_dim or 0),
            item[1].penalizer,
        ),
        reverse=True,
    )
    best_score, best_spec, best_oof = candidates[0]
    return best_spec, best_oof, best_score


def simplex_weights(step: float):
    units = int(round(1.0 / step))
    for a in range(units + 1):
        for b in range(units + 1 - a):
            c = units - a - b
            yield np.asarray([a, b, c], dtype=np.float64) / units


def select_fusion_weights(
    oof_risks: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    step: float,
) -> tuple[np.ndarray, float]:
    rows = []
    for weights in simplex_weights(step):
        score = cindex(time, event, oof_risks @ weights)
        nonzero = int((weights > 0).sum())
        # Clinical preference is only a deterministic tie-break, not a score.
        rows.append((score, -nonzero, weights[0], weights))
    rows.sort(key=lambda item: item[:3], reverse=True)
    return rows[0][3], float(rows[0][0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--vision-embeddings", type=Path, required=True)
    parser.add_argument("--vision-label", default="vision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument(
        "--penalizers", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0]
    )
    parser.add_argument("--weight-step", type=float, default=0.1)
    args = parser.parse_args()

    features = load_indexed_csv(args.features, {"case_id"}).astype(np.float64)
    targets = load_indexed_csv(
        args.targets, {"case_id", "survival_days", "event"}
    )[["survival_days", "event"]]
    text = load_text_npz(args.text_embeddings)
    vision = load_vision_csv(args.vision_embeddings)
    common = sorted(set(features.index) & set(targets.index) & set(text.index) & set(vision.index))
    cohort = targets.loc[common].copy()
    cohort["survival_days"] = pd.to_numeric(cohort["survival_days"], errors="coerce")
    cohort["event"] = pd.to_numeric(cohort["event"], errors="coerce")
    valid = (
        cohort["survival_days"].notna()
        & (cohort["survival_days"] > 0)
        & cohort["event"].isin([0, 1])
    )
    cohort = cohort.loc[valid].sort_index()
    ids = cohort.index.tolist()
    if len(ids) < 50:
        raise ValueError(f"Only {len(ids)} common patients have valid outcomes")
    time = cohort["survival_days"].to_numpy(dtype=np.float64)
    event = cohort["event"].to_numpy(dtype=np.int64)
    values = {
        "tabular": features.loc[ids].to_numpy(dtype=np.float64),
        "text": text.loc[ids].to_numpy(dtype=np.float64),
        "vision": vision.loc[ids].to_numpy(dtype=np.float64),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.reset_index().to_csv(args.output_dir / "cohort_common.csv", index=False)
    split_rows = []
    prediction_rows = []
    seed_rows = []
    all_idx = np.arange(len(ids))

    for seed in args.seeds:
        train_idx, heldout_idx = train_test_split(
            all_idx, test_size=0.20, stratify=event, random_state=int(seed)
        )
        split_rows.extend(
            {
                "seed": int(seed), "case_id": ids[idx],
                "partition": "heldout" if idx in set(heldout_idx) else "train",
                "event": int(event[idx]), "survival_days": float(time[idx]),
            }
            for idx in all_idx
        )
        oof_columns = []
        heldout_columns = []
        specs = {}
        inner_scores = {}
        for modality in MODALITIES:
            spec, oof, inner_score = select_spec_and_oof(
                values[modality][train_idx], time[train_idx], event[train_idx],
                modality, int(seed), args.pca_dims, args.penalizers, args.inner_folds,
            )
            train_risk, heldout_risk = fit_predict(
                values[modality], time, event, train_idx, heldout_idx,
                modality, spec, int(seed) * 1000 + 7,
            )
            heldout_percentile = empirical_percentile(train_risk, heldout_risk)
            oof_columns.append(oof)
            heldout_columns.append(heldout_percentile)
            specs[modality] = spec
            inner_scores[modality] = inner_score

        oof_matrix = np.column_stack(oof_columns)
        heldout_matrix = np.column_stack(heldout_columns)
        tuned_weights, fusion_inner_ci = select_fusion_weights(
            oof_matrix, time[train_idx], event[train_idx], args.weight_step
        )
        equal_weights = np.repeat(1.0 / 3.0, 3)
        risks = {
            "tabular": heldout_matrix[:, 0],
            "text": heldout_matrix[:, 1],
            "vision": heldout_matrix[:, 2],
            "late_equal": heldout_matrix @ equal_weights,
            "late_tuned_simplex": heldout_matrix @ tuned_weights,
        }
        metrics = {
            name: cindex(time[heldout_idx], event[heldout_idx], risk)
            for name, risk in risks.items()
        }
        seed_rows.append({
            "seed": int(seed), "n_train": int(len(train_idx)),
            "n_heldout": int(len(heldout_idx)),
            "events_train": int(event[train_idx].sum()),
            "events_heldout": int(event[heldout_idx].sum()),
            **{f"cindex_{name}": score for name, score in metrics.items()},
            "weight_tabular": float(tuned_weights[0]),
            "weight_text": float(tuned_weights[1]),
            "weight_vision": float(tuned_weights[2]),
            "fusion_inner_cindex": fusion_inner_ci,
            **{
                f"{modality}_pca_dim": specs[modality].pca_dim
                for modality in MODALITIES
            },
            **{
                f"{modality}_penalizer": specs[modality].penalizer
                for modality in MODALITIES
            },
            **{
                f"{modality}_inner_cindex": inner_scores[modality]
                for modality in MODALITIES
            },
        })
        for local, idx in enumerate(heldout_idx):
            prediction_rows.append({
                "seed": int(seed), "case_id": ids[idx],
                "survival_days": float(time[idx]), "event": int(event[idx]),
                **{f"risk_{name}": float(risk[local]) for name, risk in risks.items()},
            })
        print(
            f"seed={seed} tab={metrics['tabular']:.3f} text={metrics['text']:.3f} "
            f"vision={metrics['vision']:.3f} equal={metrics['late_equal']:.3f} "
            f"tuned={metrics['late_tuned_simplex']:.3f} weights={tuned_weights.tolist()}",
            flush=True,
        )

    per_seed = pd.DataFrame(seed_rows)
    predictions = pd.DataFrame(prediction_rows)
    splits = pd.DataFrame(split_rows)
    per_seed.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    splits.to_csv(args.output_dir / "splits.csv", index=False)

    result_names = ["tabular", "text", "vision", "late_equal", "late_tuned_simplex"]
    summary_results = {}
    for name in result_names:
        column = f"cindex_{name}"
        summary_results[name] = {
            "cindex_mean": float(per_seed[column].mean()),
            "cindex_std_across_seeds": float(per_seed[column].std(ddof=1)),
            "cindex_per_seed": per_seed[column].tolist(),
        }
    for fusion in ("late_equal", "late_tuned_simplex"):
        delta = per_seed[f"cindex_{fusion}"] - per_seed["cindex_tabular"]
        summary_results[f"{fusion}_minus_tabular"] = {
            "mean": float(delta.mean()),
            "std_across_seeds": float(delta.std(ddof=1)),
            "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()),
            "per_seed": delta.tolist(),
        }
    summary = {
        "status": "paired_trimodal_fusion_benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "vision_label": args.vision_label,
        "inputs": {
            "features": str(args.features.resolve()),
            "targets": str(args.targets.resolve()),
            "text_embeddings": str(args.text_embeddings.resolve()),
            "vision_embeddings": str(args.vision_embeddings.resolve()),
        },
        "cohort": {"n_cases": len(ids), "n_events": int(event.sum())},
        "protocol": {
            "outer_split": "80/20 event-stratified holdout",
            "seeds": [int(seed) for seed in args.seeds],
            "inner_folds": args.inner_folds,
            "train_only_preprocessing": True,
            "fusion": "cross-fitted empirical-risk percentiles",
            "simplex_weight_step": args.weight_step,
            "pca_dims": args.pca_dims,
            "penalizers": args.penalizers,
        },
        "results": summary_results,
        "limitations": [
            "Repeated holdouts overlap and are not independent replicates.",
            "Fusion weights are selected on outer-train cross-fitted predictions only.",
            "A result with the local ResNet cache is diagnostic; rerun with the 160-patient STU-Net export for the target comparison.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary_results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
