"""Portable, deterministic export and reload of a frozen Cox model."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any

import lifelines
import numpy as np
import pandas as pd
import sklearn
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

from components.adapters.ingestion.tabular.utils.imputation_benchmark import (
    MeanMedianImputer,
    TabularPreprocessor,
)


BUNDLE_FORMAT = "clinical-core-portable-cox-v1"
PROTOCOL = "stage_model"
DROP_FEATURES = ("pathologic_T", "pathologic_N", "pathologic_M")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: dict[str, Any]) -> bytes:
    """Return stable, standards-compliant JSON bytes."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _float_list(values) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def _named_values(names, values) -> dict[str, float]:
    return {str(name): float(value) for name, value in zip(names, values)}


def _fit_cox(frame: pd.DataFrame, targets: pd.DataFrame):
    fit_frame = frame.copy()
    fit_frame["T"] = targets["survival_days"].to_numpy(dtype=float)
    fit_frame["E"] = targets["event"].to_numpy(dtype=int)
    for penalizer in (0.5, 1.0, 5.0, 20.0):
        try:
            model = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
            model.fit(fit_frame, duration_col="T", event_col="E", show_progress=False)
            return model, float(penalizer)
        except Exception:
            continue
    raise RuntimeError("CoxPHFitter failed for every approved penalizer")


def fit_frozen_stage_model(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    *,
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the frozen full-development stage model and return its portable bundle."""
    valid = df_targets["survival_days"].notna() & (df_targets["survival_days"] > 0)
    case_order = sorted(df_targets.index[valid].astype(str))
    raw = df_features.reindex(case_order).drop(
        columns=[column for column in DROP_FEATURES if column in df_features.columns],
        errors="ignore",
    )
    targets = df_targets.reindex(case_order)

    preprocessor = TabularPreprocessor(onehot_columns=["race"], onehot_drop_first=True)
    prepared, _, _ = preprocessor.fit_transform(raw, MeanMedianImputer())
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    if prepared.isna().any(axis=None):
        raise ValueError("Unexpected missing values after frozen preprocessing")

    keep_columns = prepared.var(axis=0)
    keep_columns = keep_columns[keep_columns > 1e-8].index.tolist()
    if not keep_columns:
        raise ValueError("Frozen model has no non-constant features")

    final_scaler = StandardScaler()
    design = pd.DataFrame(
        final_scaler.fit_transform(prepared[keep_columns]),
        columns=keep_columns,
        index=prepared.index,
    )
    model, penalizer = _fit_cox(design, targets)
    risk = model.predict_partial_hazard(design).to_numpy(dtype=float)
    baseline = model.baseline_cumulative_hazard_

    numeric_statistics = _named_values(
        preprocessor.numeric_cols,
        preprocessor.imputer.imputer_num.statistics_,
    )
    categorical_statistics = _named_values(
        preprocessor.categorical_cols,
        preprocessor.imputer.imputer_cat.statistics_,
    )
    imputed_raw = preprocessor.imputer.transform(raw)
    onehot_categories = {
        column: sorted(float(value) for value in imputed_raw[column].unique())
        for column in preprocessor.onehot_columns
        if column in imputed_raw.columns
    }
    initial_scaler = {
        "columns": list(preprocessor.scaler_cols),
        "mean": _named_values(preprocessor.scaler_cols, preprocessor.scaler.mean_),
        "scale": _named_values(preprocessor.scaler_cols, preprocessor.scaler.scale_),
    }
    cox_center = {
        str(name): float(value) for name, value in model._norm_mean.items()
    }
    coefficients = {
        str(name): float(value) for name, value in model.params_.items()
    }

    bundle = {
        "format": BUNDLE_FORMAT,
        "model": {
            "family": "Cox proportional hazards",
            "protocol": PROTOCOL,
            "penalizer": penalizer,
            "l1_ratio": 0.0,
            "feature_order": keep_columns,
            "coefficients": coefficients,
            "cox_center": cox_center,
            "baseline_cumulative_hazard": {
                "time_days": _float_list(baseline.index.to_numpy(dtype=float)),
                "values": _float_list(baseline.iloc[:, 0].to_numpy(dtype=float)),
            },
        },
        "preprocessing": {
            "raw_feature_order": list(raw.columns),
            "dropped_features": list(DROP_FEATURES),
            "imputation": {
                "strategy": "median_numeric_most_frequent_categorical",
                "numeric_values": numeric_statistics,
                "categorical_values": categorical_statistics,
            },
            "initial_scaler": initial_scaler,
            "onehot": {
                "columns": ["race"],
                "drop_first": True,
                "categories": onehot_categories,
            },
            "encoded_feature_order": list(preprocessor.output_cols),
            "final_scaler": {
                "columns": keep_columns,
                "mean": _named_values(keep_columns, final_scaler.mean_),
                "scale": _named_values(keep_columns, final_scaler.scale_),
            },
        },
        "training_summary": {
            "cases": int(len(targets)),
            "events": int(targets["event"].sum()),
            "censored": int(len(targets) - targets["event"].sum()),
            "in_sample_cindex_descriptive_only": float(
                concordance_index(
                    targets["survival_days"].to_numpy(dtype=float),
                    -risk,
                    targets["event"].to_numpy(dtype=int),
                )
            ),
        },
        "endpoint_contract": {
            "endpoint": "overall_survival",
            "time_origin": "initial_pathologic_diagnosis",
            "claim_scope": "retrospective_association_not_postoperative_landmark",
        },
        "provenance": provenance,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lifelines": lifelines.__version__,
        },
        "privacy": {
            "contains_patient_rows": False,
            "contains_patient_identifiers": False,
            "contains_patient_predictions": False,
        },
    }
    reloaded = predict_partial_hazard(bundle, raw)
    if not np.allclose(reloaded, risk, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Portable reload does not reproduce fitted Cox risk")
    return bundle, design


def transform_features(bundle: dict[str, Any], raw: pd.DataFrame) -> pd.DataFrame:
    """Apply only parameters stored in a bundle to raw mapped features."""
    prep = bundle["preprocessing"]
    frame = raw.reindex(columns=prep["raw_feature_order"]).copy()
    for column, value in prep["imputation"]["numeric_values"].items():
        frame[column] = frame[column].fillna(value)
    for column, value in prep["imputation"]["categorical_values"].items():
        frame[column] = frame[column].fillna(value)
    for column in prep["initial_scaler"]["columns"]:
        frame[column] = (
            frame[column] - prep["initial_scaler"]["mean"][column]
        ) / prep["initial_scaler"]["scale"][column]
    for column in prep["onehot"]["columns"]:
        categories = prep["onehot"]["categories"][column]
        emitted = categories[1:] if prep["onehot"]["drop_first"] else categories
        for category in emitted:
            frame[f"{column}_{category}"] = (frame[column] == category).astype(float)
        frame = frame.drop(columns=[column])
    frame = frame.reindex(columns=prep["encoded_feature_order"], fill_value=0.0)
    columns = prep["final_scaler"]["columns"]
    result = frame.reindex(columns=columns).copy()
    for column in columns:
        result[column] = (
            result[column] - prep["final_scaler"]["mean"][column]
        ) / prep["final_scaler"]["scale"][column]
    return result


def predict_partial_hazard(bundle: dict[str, Any], raw: pd.DataFrame) -> np.ndarray:
    """Reload a portable bundle and predict without a lifelines model object."""
    design = transform_features(bundle, raw)
    order = bundle["model"]["feature_order"]
    coefficients = np.asarray(
        [bundle["model"]["coefficients"][name] for name in order], dtype=float
    )
    center = np.asarray(
        [bundle["model"]["cox_center"][name] for name in order], dtype=float
    )
    return np.exp((design[order].to_numpy(dtype=float) - center) @ coefficients)


def write_bundle(bundle: dict[str, Any], output: Path) -> str:
    """Write canonical JSON plus a neighboring SHA-256 receipt."""
    payload = canonical_bytes(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    return digest


def assert_privacy_safe(bundle: dict[str, Any]) -> None:
    """Reject row-level fields and TCGA-like patient barcodes."""
    rendered = canonical_bytes(bundle).decode("utf-8")
    forbidden = ('"case_id"', '"patient_id"', '"predictions"')
    matches = [token for token in forbidden if token.lower() in rendered.lower()]
    if re.search(r"\\bTCGA-[A-Z0-9]{2}-[A-Z0-9]{4}\\b", rendered, flags=re.IGNORECASE):
        matches.append("TCGA patient barcode")
    if matches:
        raise ValueError(f"Bundle contains forbidden row-level tokens: {matches}")
