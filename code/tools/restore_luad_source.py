"""Restore the frozen open-access TCGA-LUAD XML source from GDC file UUIDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = REPO_ROOT / "publication/luad_baseline_os_v2/source_inventory.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/raw/TCGA-LUAD/clinical_supplement"
DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "clinical-core.gdc-source-inventory/v1":
        raise ValueError("unsupported source inventory schema")
    files = payload.get("files")
    if not isinstance(files, list) or payload.get("file_count") != len(files):
        raise ValueError("source inventory count mismatch")
    seen = set()
    for item in files:
        file_id = item.get("file_id")
        if not isinstance(file_id, str) or not UUID_PATTERN.fullmatch(file_id):
            raise ValueError("invalid GDC file UUID")
        if file_id in seen:
            raise ValueError(f"duplicate GDC file UUID: {file_id}")
        seen.add(file_id)
        if not MD5_PATTERN.fullmatch(item.get("md5sum", "")):
            raise ValueError(f"invalid MD5 for {file_id}")
        if not isinstance(item.get("file_size"), int) or item["file_size"] <= 0:
            raise ValueError(f"invalid size for {file_id}")
    return payload


def _download_one(item: dict, output_dir: Path, timeout: int) -> dict:
    file_id = item["file_id"]
    destination = output_dir / f"{file_id}.xml"
    if (
        destination.is_file()
        and destination.stat().st_size == item["file_size"]
        and md5(destination) == item["md5sum"]
    ):
        return {"file_id": file_id, "status": "cached"}

    temporary = output_dir / f".{file_id}.part"
    last_error = None
    for attempt in range(1, 6):
        try:
            with requests.get(
                f"{DATA_ENDPOINT}/{file_id}",
                stream=True,
                timeout=timeout,
                headers={"User-Agent": "clinical-core-luad-reproduction/1"},
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if temporary.stat().st_size != item["file_size"]:
                raise ValueError(f"size mismatch for {file_id}")
            if md5(temporary) != item["md5sum"]:
                raise ValueError(f"MD5 mismatch for {file_id}")
            temporary.replace(destination)
            return {"file_id": file_id, "status": "downloaded"}
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 5:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"failed to restore {file_id}: {last_error}")


def restore(
    inventory_path: Path,
    output_dir: Path,
    *,
    limit: int = 0,
    workers: int = 6,
    timeout: int = 600,
) -> dict:
    inventory = load_inventory(inventory_path)
    selected = inventory["files"][:limit] if limit > 0 else inventory["files"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_download_one, item, output_dir, timeout): item
            for item in selected
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{index}/{len(selected)}] {result['status']}", flush=True)
    receipt = {
        "schema": "clinical-core.gdc-restore-receipt/v1",
        "inventory_sha256": sha256(inventory_path),
        "project_id": inventory["project_id"],
        "requested_files": len(selected),
        "status_counts": {
            status: sum(item["status"] == status for item in results)
            for status in sorted({item["status"] for item in results})
        },
        "complete": len(results) == len(selected),
    }
    (output_dir.parent / "RESTORE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    result = restore(
        args.inventory,
        args.output_dir,
        limit=args.limit,
        workers=args.workers,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
