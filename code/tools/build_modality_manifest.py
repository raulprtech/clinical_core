"""
Build a GDC modality availability manifest for the TCGA-KIRC cohort.

Verifies (does NOT download) the presence of four modalities per case:
  - WSI (Diagnostic Slide only — FFPE permanent sections)
  - mRNA-Seq (Gene Expression Quantification from RNA-Seq)
  - miRNA-Seq (miRNA Expression Quantification)
  - Methylation 450K (Methylation Beta Value on Illumina Human Methylation 450)

Outputs:
  - <output_dir>/gdc_modality_manifest_<project>_<YYYYMMDD>.csv
      Columns: case_id, has_wsi, has_mrna_seq, has_mirna_seq, has_methylation_450k,
               all_four_modalities
  - <output_dir>/gdc_modality_manifest_<project>_<YYYYMMDD>.json (sidecar)
      Documents: query date, project, exact filters used, per-modality counts,
      intersection counts.

Caching:
  If a manifest with today's date already exists, it is reused unless --force.

Usage:
  python3 tools/build_modality_manifest.py \\
      --case-ids-from results/<some_run>/raw_targets.csv \\
      --output-dir data/manifests/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

# Ensure repo root is on sys.path so `core.gdc_downloader` imports cleanly
# whether invoked from /code or elsewhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.gdc_downloader import GDCDataFetcher  # noqa: E402

logger = logging.getLogger("build_modality_manifest")

# Modality definitions (strict MMEM-style)
MODALITY_QUERIES: Dict[str, Dict] = {
    'wsi': {
        'data_types': ['Slide Image'],
        'experimental_strategy': ['Diagnostic Slide'],
        'platform': None,
        'description': 'WSI Diagnostic Slide (FFPE permanent section)',
    },
    'mrna_seq': {
        'data_types': ['Gene Expression Quantification'],
        'experimental_strategy': ['RNA-Seq'],
        'platform': None,
        'description': 'mRNA-Seq gene expression quantification (RNA-Seq)',
    },
    'mirna_seq': {
        'data_types': ['miRNA Expression Quantification'],
        'experimental_strategy': None,
        'platform': None,
        'description': 'miRNA expression quantification',
    },
    'methylation_450k': {
        'data_types': ['Methylation Beta Value'],
        'experimental_strategy': None,
        'platform': ['Illumina Human Methylation 450'],
        'description': 'DNA methylation beta values on Illumina Human Methylation 450K',
    },
}


def load_case_ids(case_ids_csv: Path) -> List[str]:
    """Load submitter_ids (TCGA-XX-XXXX) from a raw_targets.csv file."""
    df = pd.read_csv(case_ids_csv)
    if 'case_id' in df.columns:
        ids = df['case_id'].dropna().astype(str).str.strip().unique().tolist()
    elif df.index.name == 'case_id':
        ids = df.index.dropna().astype(str).str.strip().unique().tolist()
    else:
        # Heuristic: any column that looks like TCGA-XX-XXXX
        candidates = [c for c in df.columns if df[c].astype(str).str.match(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}').any()]
        if not candidates:
            raise ValueError(f"Could not find a case_id column in {case_ids_csv}")
        ids = df[candidates[0]].dropna().astype(str).str.strip().unique().tolist()
    # Filter to TCGA-XX-XXXX shape
    ids = [i for i in ids if len(i) == 12 and i.startswith('TCGA-')]
    return sorted(set(ids))


def query_modality(
    fetcher: GDCDataFetcher,
    submitter_ids: List[str],
    query_spec: Dict,
) -> Set[str]:
    """
    Query GDC for one modality across the given submitter_ids. Returns the set of
    submitter_ids that have at least one matching file.
    """
    hits = fetcher.search_files(
        data_types=query_spec['data_types'],
        experimental_strategy=query_spec.get('experimental_strategy'),
        platform=query_spec.get('platform'),
        target_submitter_ids=submitter_ids,
        match_field='submitter_id',
        limit=20000,
    )
    present = set()
    for h in hits:
        for c in h.get('cases', []):
            sid = c.get('submitter_id')
            if sid:
                present.add(sid)
    return present


def build_manifest(
    submitter_ids: List[str],
    project_id: str = 'TCGA-KIRC',
    batch_size: int = 200,
) -> Dict:
    """
    Build the manifest by querying GDC for each modality in batches.

    Returns a dict with 'rows' (list of per-case dicts) and 'counts' (per-modality
    coverage counts).
    """
    fetcher = GDCDataFetcher(project_id=project_id)

    presence: Dict[str, Set[str]] = {m: set() for m in MODALITY_QUERIES}

    for mod_name, spec in MODALITY_QUERIES.items():
        logger.info(f"Querying GDC for modality '{mod_name}' "
                    f"({spec['description']}) on {len(submitter_ids)} cases...")
        for i in range(0, len(submitter_ids), batch_size):
            batch = submitter_ids[i:i + batch_size]
            try:
                hits_set = query_modality(fetcher, batch, spec)
                presence[mod_name].update(hits_set)
                logger.info(f"  batch {i}-{i + len(batch)}: {len(hits_set)} cases with {mod_name}")
            except Exception as e:
                logger.error(f"  batch {i}-{i + len(batch)} for {mod_name} FAILED: {e}")
                raise

    rows = []
    for sid in submitter_ids:
        row = {'case_id': sid}
        flags = {}
        for mod_name in MODALITY_QUERIES:
            flag = sid in presence[mod_name]
            row[f'has_{mod_name}'] = bool(flag)
            flags[mod_name] = flag
        row['all_four_modalities'] = bool(all(flags.values()))
        rows.append(row)

    counts = {
        mod_name: int(len(presence[mod_name])) for mod_name in MODALITY_QUERIES
    }
    counts['all_four_modalities'] = int(sum(1 for r in rows if r['all_four_modalities']))
    counts['queried_total'] = int(len(submitter_ids))

    return {'rows': rows, 'counts': counts}


def write_outputs(result: Dict, output_path: Path, project_id: str,
                  case_ids_source: Path) -> None:
    """Persist the manifest CSV and a sidecar JSON describing the query."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(result['rows'])
    df.to_csv(output_path, index=False)
    logger.info(f"Manifest written: {output_path}  ({len(df)} cases)")

    sidecar_path = output_path.with_suffix('.json')
    payload = {
        'project_id': project_id,
        'query_date_utc': datetime.utcnow().isoformat() + 'Z',
        'gdc_api_endpoint': GDCDataFetcher.API_FILES_ENDPOINT,
        'case_ids_source': str(case_ids_source),
        'modality_query_spec': MODALITY_QUERIES,
        'counts': result['counts'],
        'manifest_csv': str(output_path.name),
    }
    with open(sidecar_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Sidecar written: {sidecar_path}")

    # Pretty summary
    print()
    print("=" * 60)
    print(f"MODALITY MANIFEST — {project_id}")
    print("=" * 60)
    print(f"Queried cases: {result['counts']['queried_total']}")
    for mod_name in MODALITY_QUERIES:
        cnt = result['counts'][mod_name]
        pct = cnt / max(1, result['counts']['queried_total']) * 100
        print(f"  has_{mod_name:<25} {cnt:>5d}  ({pct:5.1f}%)")
    cnt_all = result['counts']['all_four_modalities']
    pct_all = cnt_all / max(1, result['counts']['queried_total']) * 100
    print(f"  {'all four modalities':<29} {cnt_all:>5d}  ({pct_all:5.1f}%)")
    print(f"\nManifest: {output_path}")
    print(f"Sidecar:  {sidecar_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case-ids-from', required=True, type=Path,
                    help='Path to a CSV (e.g. raw_targets.csv) containing a case_id column.')
    ap.add_argument('--output-dir', default=REPO_ROOT.parent / 'data' / 'manifests',
                    type=Path,
                    help='Directory to write manifest CSV + sidecar JSON.')
    ap.add_argument('--project-id', default='TCGA-KIRC',
                    help='GDC project_id (default: TCGA-KIRC).')
    ap.add_argument('--batch-size', type=int, default=200,
                    help='Submitter IDs per GDC API call (default 200).')
    ap.add_argument('--force', action='store_true',
                    help='Re-query GDC even if today\'s manifest already exists.')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    today = datetime.utcnow().strftime('%Y%m%d')
    output_path = Path(args.output_dir) / f'gdc_modality_manifest_{args.project_id}_{today}.csv'

    if output_path.exists() and not args.force:
        logger.info(f"Manifest already exists for today: {output_path}")
        logger.info("Use --force to re-query GDC.")
        df = pd.read_csv(output_path)
        print(f"\nExisting manifest: {output_path}  ({len(df)} cases)")
        print(f"  cases with all 4 modalities: {df.get('all_four_modalities', pd.Series(dtype=bool)).sum()}")
        return 0

    submitter_ids = load_case_ids(args.case_ids_from)
    logger.info(f"Loaded {len(submitter_ids)} case_ids from {args.case_ids_from}")

    result = build_manifest(submitter_ids, project_id=args.project_id, batch_size=args.batch_size)
    write_outputs(result, output_path, args.project_id, args.case_ids_from)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
