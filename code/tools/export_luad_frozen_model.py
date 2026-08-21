#!/usr/bin/env python3
"""Export the full-development TCGA-LUAD stage Cox model as portable JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from components.adapters.ingestion.tabular.utils.extractor import TCGAExtractor
from core.frozen_cox_bundle import (
    assert_privacy_safe,
    fit_frozen_stage_model,
    sha256_file,
    write_bundle,
)
from core.reproducibility import resolve_runtime_paths


DEFAULT_CONFIG = REPO_ROOT / "code/experiments/experiment_config_nigma_luad_baseline_os_v2.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "publication/luad_frozen_stage_model_v1/model.json"
BASELINE_TAG = "tcga-luad-baseline-os-v2"
BASELINE_MERGE = "f9719d5c7db12d7fc9de45178091b7ed137a5582"


def export(config_path: Path, output: Path) -> dict:
    manifest_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = resolve_runtime_paths(manifest_config, config_path.resolve())
    feature_config = Path(config["data"]["feature_config"])
    inventory = REPO_ROOT / "publication/luad_baseline_os_v2/source_inventory.json"
    baseline_manifest = REPO_ROOT / "publication/luad_baseline_os_v2/manifest.json"
    extractor = TCGAExtractor(str(feature_config))
    features, targets = extractor.extract_cohort(config["data"]["xml_dir"])
    provenance = {
        "development_baseline_tag": BASELINE_TAG,
        "development_baseline_merge_commit": BASELINE_MERGE,
        "experiment_config_sha256": sha256_file(config_path),
        "feature_config_sha256": sha256_file(feature_config),
        "source_inventory_sha256": sha256_file(inventory),
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
    }
    bundle, _ = fit_frozen_stage_model(features, targets, provenance=provenance)
    assert_privacy_safe(bundle)
    digest = write_bundle(bundle, output)
    return {"output": str(output), "sha256": digest, **bundle["training_summary"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export(args.config.resolve(), args.output.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
