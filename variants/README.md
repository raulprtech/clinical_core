# Prepared variant registry

`registry-v1.json` is the passive, versioned catalog consumed by Clinical-Nigma.
Clinical-Core does not select a variant, approve a run, or execute a request.

Every entry names an exact entrypoint and immutable artifact hashes. A catalog
entry describes its actual readiness and evidence; unsupported scientific
claims remain explicit. configuration_ready means that the portable contract
and fail-closed inputs are verified, not that a clean scientific reproduction
has already run.

All seven KIRC combinations of tabular, text and vision are described in
docs/modality_variant_configurations.md. Adding a variant requires updating its
evidence, hashes, catalog digest, and the registry tests.
