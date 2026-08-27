"""Nested paired evaluation of frozen STU-Net volumetric pooling variants.

The primary contrast is historical regional means (768D) versus the predeclared
renal ROI mean+standard-deviation representation (512D). Both embeddings come
from the same inference and use the same outcome-independent CT cohort. Every
outer split uses train-only PCA and inner CV to select a ridge Cox head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored


DEFAULT_COMPONENTS = [4, 8]
DEFAULT_ALPHAS = [100.0, 10.0, 1.0]
MODEL_FILES = {
    "stunet_mean_768": "stunet_s_fp32_embeddings_768.csv",
    "stunet_renal_moments_512": "stunet_s_fp32_renal_moments_512.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def survival_array(events: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.array(
        list(zip(events.astype(bool), times.astype(float))),
        dtype=[("event", "?"), ("time", "<f8")],
    )


def cindex(events: np.ndarray, times: np.ndarray, risks: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            events.astype(bool), times.astype(float), risks.astype(float)
        )[0]
    )


def load_embeddings(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "case_id" not in frame:
        raise ValueError(f"{path} has no case_id column")
    feature_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("z") and column[1:].isdigit()
    )
    if not feature_columns:
        raise ValueError(f"{path} has no embedding columns")
    frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    if frame["case_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate case_id values")
    values = frame[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError(f"{path} contains zero-norm embeddings")
    values /= norms[:, None]
    return pd.DataFrame(values, index=frame["case_id"], columns=feature_columns)


def fit_transform_cox(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    components: int,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_components = min(components, x_train.shape[0] - 1, x_train.shape[1])
    if n_components < 1:
        raise ValueError("Not enough training samples for PCA")
    pca = PCA(n_components=n_components, svd_solver="full")
    train_pcs = pca.fit_transform(x_train)
    test_pcs = pca.transform(x_test)
    scaler = StandardScaler().fit(train_pcs)
    train_scaled = scaler.transform(train_pcs)
    test_scaled = scaler.transform(test_pcs)
    model = CoxPHSurvivalAnalysis(alpha=float(alpha), ties="breslow", n_iter=200)
    model.fit(train_scaled, y_train)
    return model.predict(test_scaled), {
        "pca_components": int(n_components),
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "alpha": float(alpha),
    }


def select_hyperparameters(
    x: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    train_indices: np.ndarray,
    components_grid: list[int],
    alpha_grid: list[float],
    inner_splits: int,
    seed: int,
) -> tuple[int, float, float]:
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    candidates: list[tuple[float, int, float]] = []
    for components in sorted(components_grid):
        for alpha in sorted(alpha_grid, reverse=True):
            scores = []
            for inner_train_local, inner_valid_local in splitter.split(
                train_indices, events[train_indices]
            ):
                inner_train = train_indices[inner_train_local]
                inner_valid = train_indices[inner_valid_local]
                try:
                    risk, _ = fit_transform_cox(
                        x[inner_train],
                        survival_array(events[inner_train], times[inner_train]),
                        x[inner_valid],
                        components,
                        alpha,
                    )
                    scores.append(cindex(events[inner_valid], times[inner_valid], risk))
                except (ArithmeticError, ValueError, np.linalg.LinAlgError):
                    scores = []
                    break
            if scores:
                candidates.append((float(np.mean(scores)), components, alpha))
    if not candidates:
        raise RuntimeError("All inner-CV Cox configurations failed")
    # Stable sort means ties retain the predeclared simpler order: fewer PCs,
    # then stronger ridge regularization.
    best = max(candidates, key=lambda item: item[0])
    return int(best[1]), float(best[2]), float(best[0])


def rank_within_fold(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["risk_rank"] = ranked.groupby(
        ["model", "repeat", "outer_fold"], sort=False
    )["risk"].rank(method="average", pct=True)
    return ranked


def paired_patient_bootstrap(
    averaged: pd.DataFrame, iterations: int, seed: int
) -> dict[str, Any]:
    pivot = averaged.pivot(
        index=["case_id", "survival_days", "event"],
        columns="model",
        values="risk_rank",
    ).reset_index()
    baseline = "stunet_mean_768"
    candidate = "stunet_renal_moments_512"
    rng = np.random.default_rng(seed)
    observed = {
        baseline: cindex(
            pivot.event.to_numpy(),
            pivot.survival_days.to_numpy(),
            pivot[baseline].to_numpy(),
        ),
        candidate: cindex(
            pivot.event.to_numpy(),
            pivot.survival_days.to_numpy(),
            pivot[candidate].to_numpy(),
        ),
    }
    samples = {baseline: [], candidate: [], "delta": []}
    for _ in range(iterations):
        indices = rng.integers(0, len(pivot), size=len(pivot))
        boot = pivot.iloc[indices]
        try:
            values = {
                baseline: cindex(
                    boot.event.to_numpy(),
                    boot.survival_days.to_numpy(),
                    boot[baseline].to_numpy(),
                ),
                candidate: cindex(
                    boot.event.to_numpy(),
                    boot.survival_days.to_numpy(),
                    boot[candidate].to_numpy(),
                ),
            }
        except ValueError:
            continue
        samples[baseline].append(values[baseline])
        samples[candidate].append(values[candidate])
        samples["delta"].append(values[candidate] - values[baseline])
    result: dict[str, Any] = {
        "observed_cindex": observed,
        "observed_delta_candidate_minus_historical": (
            observed[candidate] - observed[baseline]
        ),
        "bootstrap_valid_iterations": len(samples["delta"]),
    }
    for key, values in samples.items():
        result[f"{key}_ci95"] = np.quantile(values, [0.025, 0.975]).astype(float).tolist()
    result["probability_delta_gt_zero"] = float(
        np.mean(np.asarray(samples["delta"]) > 0)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260827)
    parser.add_argument("--components", nargs="+", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    embeddings = {
        name: load_embeddings(args.embedding_dir / filename)
        for name, filename in MODEL_FILES.items()
    }
    targets = pd.read_csv(args.targets)
    cohort = pd.read_csv(args.cohort)
    for frame, label in ((targets, "targets"), (cohort, "cohort")):
        if "case_id" not in frame:
            raise ValueError(f"{label} has no case_id column")
        frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    required_targets = {"survival_days", "event"}
    if missing := required_targets - set(targets.columns):
        raise ValueError(f"targets missing columns: {sorted(missing)}")
    target_index = targets.drop_duplicates("case_id").set_index("case_id")
    requested_ids = set(cohort["case_id"])
    common_ids = sorted(
        requested_ids
        & set(target_index.index)
        & set.intersection(*(set(frame.index) for frame in embeddings.values()))
    )
    outcomes = target_index.loc[common_ids, ["survival_days", "event"]].copy()
    outcomes["survival_days"] = pd.to_numeric(outcomes["survival_days"], errors="coerce")
    outcomes["event"] = pd.to_numeric(outcomes["event"], errors="coerce")
    outcomes = outcomes[
        outcomes.survival_days.notna()
        & (outcomes.survival_days > 0)
        & outcomes.event.isin([0, 1])
    ].sort_index()
    common_ids = outcomes.index.tolist()
    if len(common_ids) < args.outer_splits * 2:
        raise ValueError(f"Too few complete cases: {len(common_ids)}")
    times = outcomes.survival_days.to_numpy(dtype=float)
    events = outcomes.event.to_numpy(dtype=int)
    if int(events.sum()) < args.outer_splits:
        raise ValueError("Fewer events than outer folds")
    x_by_model = {
        name: frame.loc[common_ids].to_numpy(dtype=float)
        for name, frame in embeddings.items()
    }

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    all_indices = np.arange(len(common_ids))
    for repeat in range(args.repeats):
        outer_seed = args.base_seed + repeat
        outer = StratifiedKFold(
            n_splits=args.outer_splits, shuffle=True, random_state=outer_seed
        )
        for outer_fold, (train_indices, heldout_indices) in enumerate(
            outer.split(all_indices, events)
        ):
            for model_name, x in x_by_model.items():
                components, alpha, inner_cindex = select_hyperparameters(
                    x,
                    times,
                    events,
                    train_indices,
                    args.components,
                    args.alphas,
                    args.inner_splits,
                    seed=outer_seed * 100 + outer_fold,
                )
                risk, metadata = fit_transform_cox(
                    x[train_indices],
                    survival_array(events[train_indices], times[train_indices]),
                    x[heldout_indices],
                    components,
                    alpha,
                )
                heldout_cindex = cindex(
                    events[heldout_indices], times[heldout_indices], risk
                )
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "n_train": len(train_indices),
                        "n_heldout": len(heldout_indices),
                        "events_train": int(events[train_indices].sum()),
                        "events_heldout": int(events[heldout_indices].sum()),
                        "inner_mean_cindex": inner_cindex,
                        "heldout_cindex": heldout_cindex,
                        **metadata,
                    }
                )
                prediction_rows.extend(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "case_id": common_ids[index],
                        "survival_days": float(times[index]),
                        "event": int(events[index]),
                        "risk": float(case_risk),
                    }
                    for index, case_risk in zip(heldout_indices, risk)
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = rank_within_fold(pd.DataFrame(prediction_rows))
    folds = pd.DataFrame(fold_rows)
    predictions.to_csv(args.output_dir / "outer_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    outcomes.reset_index().to_csv(args.output_dir / "cohort_complete.csv", index=False)
    averaged = (
        predictions.groupby(
            ["case_id", "survival_days", "event", "model"], as_index=False
        )
        .risk_rank.mean()
    )
    bootstrap = paired_patient_bootstrap(
        averaged, args.bootstrap_iterations, args.base_seed + 999_999
    )
    fold_summary = (
        folds.groupby("model")
        .heldout_cindex.agg(["mean", "std", "median", "min", "max"])
        .to_dict(orient="index")
    )
    paired_folds = folds.pivot(
        index=["repeat", "outer_fold"], columns="model", values="heldout_cindex"
    )
    fold_deltas = (
        paired_folds["stunet_renal_moments_512"] - paired_folds["stunet_mean_768"]
    )
    summary = {
        "schema_version": 1,
        "design": (
            f"{args.repeats}x repeated nested stratified CV with "
            f"{args.outer_splits} outer folds; train-only PCA; "
            "inner-selected ridge Cox"
        ),
        "n_cases": len(common_ids),
        "n_events": int(events.sum()),
        "outer_splits": args.outer_splits,
        "repeats": args.repeats,
        "inner_splits": args.inner_splits,
        "components_grid": args.components,
        "alpha_grid": args.alphas,
        "fold_cindex": fold_summary,
        "paired_fold_delta": {
            "mean": float(fold_deltas.mean()),
            "std": float(fold_deltas.std(ddof=1)),
            "median": float(fold_deltas.median()),
            "min": float(fold_deltas.min()),
            "max": float(fold_deltas.max()),
            "positive_folds": int((fold_deltas > 0).sum()),
            "n_folds": int(len(fold_deltas)),
        },
        "patient_clustered_rank_bootstrap": bootstrap,
        "interpretation_guardrail": (
            "Exploratory representation comparison; not an external validation or a "
            "claim of clinical utility. Bootstrap ranks normalize fold-specific risk scales."
        ),
        "input_sha256": {
            "targets": sha256_file(args.targets),
            "cohort": sha256_file(args.cohort),
            **{
                name: sha256_file(args.embedding_dir / filename)
                for name, filename in MODEL_FILES.items()
            },
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
