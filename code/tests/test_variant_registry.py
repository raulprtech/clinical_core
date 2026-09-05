import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "variants" / "registry-v1.json"


def _canonical_digest(payload: dict) -> str:
    content = {key: value for key, value in payload.items() if key != "catalog_digest"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_variant_registry_is_self_consistent() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert payload["schema"] == "clinical-core.variant-catalog/v1"
    assert payload["catalog_digest"] == _canonical_digest(payload)

    identities = set()
    for variant in payload["variants"]:
        identity = (variant["id"], variant["version"])
        assert identity not in identities
        identities.add(identity)
        assert variant["status"] in {
            "prepared", "frozen", "historical", "configuration_ready"
        }
        assert variant["leakage_control"] in {"safe", "locked"}
        assert not variant["entrypoint"]["value"].startswith(("/", "~"))

        for reference in variant["artifact_refs"]:
            relative = Path(reference["path"])
            assert not relative.is_absolute()
            assert ".." not in relative.parts
            artifact = ROOT / relative
            assert artifact.is_file(), reference["path"]
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            assert digest == reference["sha256"], reference["path"]
        for evidence_ref in variant["evidence_refs"]:
            relative = Path(evidence_ref)
            assert not relative.is_absolute()
            assert ".." not in relative.parts
            assert (ROOT / relative).is_file(), evidence_ref

        configuration_ref = variant.get("configuration_ref")
        if configuration_ref is not None:
            assert configuration_ref in variant["artifact_refs"]


def test_registry_contains_no_patient_payloads() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    forbidden = {"patient_id", "subject_id", "raw_values", "records"}
    def inspect(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(str(key).casefold() for key in value)
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(payload)
