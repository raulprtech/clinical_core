"""
TABULAR-CONN v2: XML Extraction Pipeline
Extracts all clinically relevant variables from TCGA-KIRC BCR XML files.
Separates FEATURES (input) from TARGETS (survival).

Usage:
    extractor = TCGAExtractor(config_path="config_v2.yaml")
    df_features, df_targets = extractor.extract_cohort(xml_directory)
"""

import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import yaml
import re
import warnings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))


class TCGAExtractor:
    """Extracts clinical variables from TCGA BCR XML files."""
    
    def __init__(self, config_path: str = "components/adapters/ingestion/tabular/configs/tabular_mapping.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Build reverse lookup: xml_tag -> (variable_name, section)
        self.tag_lookup = {}
        for var_name, var_config in self.config['features'].items():
            for source in var_config['sources']:
                self.tag_lookup[source.lower()] = (var_name, 'feature', var_config)
        
        # FIX (Apr 2026, BUG 2): for survival_days, store EACH source separately
        # under a synthetic key so _resolve_survival can apply clinical priority.
        for var_name, var_config in self.config['targets'].items():
            for source in var_config['sources']:
                if var_name == 'survival_days':
                    # Store under synthetic key 'target__source__<source_name>'
                    self.tag_lookup[source.lower()] = ('source__' + source.lower(), 'target', var_config)
                else:
                    self.tag_lookup[source.lower()] = (var_name, 'target', var_config)
    
    def parse_single_xml(self, filepath: Path) -> Dict:
        """Parse one TCGA BCR XML file into a flat dictionary of extracted values."""
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Extract case ID
        case_id = None
        raw_values = {}
        
        for elem in root.iter():
            # Strip namespace
            tag = elem.tag.split('}')[-1].lower() if '}' in elem.tag else elem.tag.lower()
            
            # Also check preferred_name attribute (TCGA uses this)
            preferred_name = elem.attrib.get('preferred_name', '').lower()
            
            # Try to find case barcode
            if tag == 'bcr_patient_barcode' and elem.text:
                case_id = elem.text.strip()
            
            # Extract value if tag or preferred_name matches our lookup
            text = elem.text.strip() if elem.text and elem.text.strip() else None
            
            # Skip explicitly nil or empty values
            if text is None or text == '':
                # Check if xsi:nil="true"
                nil = elem.attrib.get('{http://www.w3.org/2001/XMLSchema-instance}nil', 'false')
                if nil == 'true':
                    continue
                # Check procurement_status
                procurement = elem.attrib.get('procurement_status', '')
                if procurement in ['Not Available', 'Not Applicable']:
                    continue
                continue
            
            # Match by tag name
            if tag in self.tag_lookup:
                var_name, section, var_config = self.tag_lookup[tag]
                # Don't overwrite if already found (first match wins)
                key = f"{section}__{var_name}"
                if key not in raw_values:
                    raw_values[key] = text
            
            # Match by preferred_name
            if preferred_name and preferred_name in self.tag_lookup:
                var_name, section, var_config = self.tag_lookup[preferred_name]
                key = f"{section}__{var_name}"
                if key not in raw_values:
                    raw_values[key] = text
        
        if case_id is None:
            # Try filename
            match = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', filepath.name)
            case_id = match.group() if match else filepath.stem
        
        raw_values['case_id'] = case_id
        return raw_values
    
    def _apply_mapping(self, raw_value: str, var_config: dict) -> Optional[float]:
        """Convert raw string to numeric using config mapping."""
        if raw_value is None:
            return np.nan
        
        var_type = var_config.get('type', 'categorical')
        
        # Continuous variables: try direct float conversion
        if var_type == 'continuous':
            try:
                return float(raw_value)
            except (ValueError, TypeError):
                return np.nan
        
        # Categorical with mapping
        mapping = var_config.get('mapping', {})
        if mapping:
            # Try exact match first
            if raw_value in mapping:
                val = mapping[raw_value]
                return np.nan if val == -1 else float(val)
            
            # Try case-insensitive
            for key, val in mapping.items():
                if raw_value.lower().strip() == key.lower().strip():
                    return np.nan if val == -1 else float(val)
            
            # For stage/grade: try regex extraction
            if 'Stage' in str(list(mapping.keys())):
                for key, val in mapping.items():
                    if key.lower() in raw_value.lower():
                        return np.nan if val == -1 else float(val)
        
        # Default mapping for categorical_lab type
        # FIX (Apr 2026, BUG 3): use exact match first, then substring as fallback,
        # to avoid mismatches like "Low Normal" -> "Low" or "Below Normal" -> "Normal".
        if var_type == 'categorical_lab':
            lab_map = var_config.get('mapping', {})
            raw_clean = raw_value.lower().strip()
            # Pass 1: exact match (case- and whitespace-insensitive)
            for key, val in lab_map.items():
                if key.lower().strip() == raw_clean:
                    return float(val)
            # Pass 2: substring match as fallback, longest key first to avoid
            # "Normal" matching before "Low Normal" or "Below Normal"
            for key, val in sorted(lab_map.items(), key=lambda kv: -len(kv[0])):
                if key.lower().strip() in raw_clean:
                    return float(val)
        
        # Fallback: try float
        try:
            return float(raw_value)
        except (ValueError, TypeError):
            return np.nan
    
    def _resolve_survival(self, raw_values: dict) -> Tuple[float, int]:
        """
        Resolve survival time and censoring status.
        FIX (Apr 2026, BUG 2): the original implementation took whichever
        days_to_* tag appeared first in XML iter order, which is NOT necessarily
        the clinically correct one. This version applies the clinical rule:
            - vital_status == Dead  -> use days_to_death
            - vital_status == Alive -> use days_to_last_followup
        Falls back gracefully if the preferred source is missing.
        """
        vital_raw = raw_values.get('target__vital_status', None)
        days_to_death = raw_values.get('target__source__days_to_death', None)
        days_to_followup = raw_values.get('target__source__days_to_last_followup', None)

        # Determine event indicator
        if vital_raw is not None:
            vital_lower = str(vital_raw).lower().strip()
            if vital_lower == 'dead':
                event = 1
            elif vital_lower == 'alive':
                event = 0
            else:
                event = 0  # Default to censored if ambiguous
        else:
            event = 0

        # Apply clinical priority based on event
        survival_days = np.nan
        if event == 1:
            # Dead: prefer days_to_death; fall back to days_to_last_followup if missing
            preferred = days_to_death if days_to_death is not None else days_to_followup
        else:
            # Alive (or unknown): prefer days_to_last_followup
            preferred = days_to_followup if days_to_followup is not None else days_to_death

        if preferred is not None:
            try:
                survival_days = float(preferred)
            except (ValueError, TypeError):
                pass

        # FIX (Apr 2026, BUG 2-bis): drop clinically invalid times (negative or zero
        # survival_days indicate XML date errors and should be treated as missing).
        if survival_days is not None and not np.isnan(survival_days) and survival_days <= 0:
            survival_days = np.nan

        return survival_days, event
    
    def extract_cohort(self, xml_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract all XMLs in directory. Returns (features_df, targets_df).
        Both indexed by case_id.
        """
        xml_dir = Path(xml_dir)
        xml_files = list(xml_dir.glob("*.xml"))
        
        if not xml_files:
            # Try recursive
            xml_files = list(xml_dir.rglob("*.xml"))
        
        print(f"Found {len(xml_files)} XML files")
        
        all_records = []
        extraction_errors = []
        
        for xml_path in sorted(xml_files):
            try:
                raw = self.parse_single_xml(xml_path)
                all_records.append(raw)
            except Exception as e:
                extraction_errors.append((xml_path.name, str(e)))
        
        if extraction_errors:
            print(f"  Extraction errors: {len(extraction_errors)}")
            for name, err in extraction_errors[:5]:
                print(f"    {name}: {err}")
        
        print(f"  Successfully parsed: {len(all_records)} cases")
        
        # Build features dataframe
        feature_rows = []
        target_rows = []
        
        for raw in all_records:
            case_id = raw['case_id']
            
            # Extract features
            feat_row = {'case_id': case_id}
            for var_name, var_config in self.config['features'].items():
                key = f"feature__{var_name}"
                raw_val = raw.get(key, None)
                feat_row[var_name] = self._apply_mapping(raw_val, var_config)
            feature_rows.append(feat_row)
            
            # Extract targets
            survival_days, event = self._resolve_survival(raw)
            target_rows.append({
                'case_id': case_id,
                'survival_days': survival_days,
                'event': event
            })
        
        if not feature_rows:
            raise ValueError(f"No clinical XML records were successfully extracted from {xml_dir}. Make sure the directory has files and they are valid.")

        df_features = pd.DataFrame(feature_rows).set_index('case_id')
        df_targets = pd.DataFrame(target_rows).set_index('case_id')
        
        # Deduplicate cases based on index (case_id). Some TCGA cases might have multiple XMLs.
        df_features = df_features[~df_features.index.duplicated(keep='last')]
        df_targets = df_targets[~df_targets.index.duplicated(keep='last')]
        
        # Handle days_to_birth → age conversion if age is missing
        # TCGA stores age as days_to_birth (negative number)
        # Already handled via age_at_initial_pathologic_diagnosis
        
        # Quality report
        self._print_quality_report(df_features, df_targets)
        
        return df_features, df_targets
    
    def _print_quality_report(self, df_features: pd.DataFrame, df_targets: pd.DataFrame):
        """Print extraction quality summary."""
        n = len(df_features)
        print(f"\n{'='*60}")
        print(f"EXTRACTION QUALITY REPORT — {n} cases")
        print(f"{'='*60}")
        
        # Features completeness
        print(f"\n{'Variable':<35} {'Present':>8} {'Missing':>8} {'%Complete':>10}")
        print("-" * 63)
        for col in df_features.columns:
            present = df_features[col].notna().sum()
            missing = df_features[col].isna().sum()
            pct = present / n * 100
            print(f"{col:<35} {present:>8} {missing:>8} {pct:>9.1f}%")
        
        # Targets completeness
        print(f"\n--- TARGETS ---")
        for col in df_targets.columns:
            present = df_targets[col].notna().sum()
            pct = present / n * 100
            print(f"{col:<35} {present:>8} {n-present:>8} {pct:>9.1f}%")
        
        # Survival summary
        valid_surv = df_targets['survival_days'].dropna()
        events = df_targets['event'].sum()
        print(f"\nSurvival summary:")
        print(f"  Events (deaths): {int(events)} ({events/n*100:.1f}%)")
        print(f"  Censored (alive): {int(n - events)} ({(n-events)/n*100:.1f}%)")
        if len(valid_surv) > 0:
            print(f"  Median follow-up: {valid_surv.median():.0f} days ({valid_surv.median()/365.25:.1f} years)")
            print(f"  Range: {valid_surv.min():.0f} — {valid_surv.max():.0f} days")


# ============================================================
# DRUG DATA PARSER (supplementary — not used in embedding)
# ============================================================

def parse_drug_file(txt_path: str) -> pd.DataFrame:
    """
    Parse TCGA drug treatment TXT file.
    Handles the 3 header rows (names, aliases, CDE_IDs).
    Returns one row per treatment event.
    """
    df = pd.read_csv(txt_path, sep='\t', skiprows=[1, 2], na_values=['[Not Available]', '[Not Applicable]', '[Discrepancy]'])
    
    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    # Extract relevant columns
    keep_cols = [
        'bcr_patient_barcode', 'pharmaceutical_therapy_drug_name',
        'pharmaceutical_therapy_type', 'pharmaceutical_tx_started_days_to',
        'pharmaceutical_tx_ended_days_to', 'treatment_best_response',
        'pharmaceutical_tx_ongoing_indicator'
    ]
    # Use available columns (names may vary)
    available = [c for c in keep_cols if c in df.columns]
    if not available:
        # Try alternate names
        df_out = df[['bcr_patient_barcode']].copy() if 'bcr_patient_barcode' in df.columns else df
    else:
        df_out = df[available].copy()
    
    return df_out


if __name__ == "__main__":
    import sys
    xml_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    extractor = TCGAExtractor("components/adapters/ingestion/tabular/configs/tabular_mapping.yaml")
    df_feat, df_targ = extractor.extract_cohort(xml_dir)
    
    # Save
    df_feat.to_csv("tcga_kirc_features.csv")
    df_targ.to_csv("tcga_kirc_targets.csv")
    print(f"\nSaved features ({df_feat.shape}) and targets ({df_targ.shape})")