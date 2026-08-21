# Critical methodology and code review

Review date: 2026-08-20

## What the current evidence supports

The active line demonstrates a deterministic, leakage-conscious internal
evaluation of three clinicopathologic Cox specifications in TCGA-LUAD. It
supports reporting internal holdout discrimination, repeated cross-validation
stability and an earlier-to-later internal transport stress test.

## Methodological limitations

1. The random holdout and repeated cross-validation reuse one retrospective
   cohort. They estimate internal performance and stability, not transport.
2. The temporal split is highly asymmetric: 147 earlier cases with 92 events
   train the model and 350 later cases with 30 events validate it. Calendar-time
   changes, follow-up maturity and censoring can therefore drive differences.
3. The temporal cutoff and model specifications have been inspected on the same
   project during development. The temporal result is not an untouched external
   confirmation.
4. The three protocols are related, multiple comparisons are not adjusted and
   no protocol is declared clinically superior from the observed differences.
5. Missingness is handled inside training folds, but missingness mechanisms and
   complete-case sensitivity are not yet reported.
6. Stage and TNM encode overlapping clinical information. Rank checks reduce
   numerical duplication, but do not make Stage+TNM an independent biological
   signal.
7. Calibration is assessed at a selected two-year horizon. Additional horizons,
   calibration-in-the-large and slope sensitivity remain publication tasks.
8. The endpoint is retrospective overall survival from initial diagnosis. The
   data do not support a postoperative landmark or prospective decision claim.
9. No current-line external cohort, decision-curve analysis, clinical-utility
   study or prospective validation exists.

## Code and reproducibility findings

### Corrected in this package

- The active YAML previously contained machine-specific absolute paths.
- The config digest previously changed after runtime path resolution.
- Python emitted non-standard NaN tokens in summary JSON.
- The exact source was available only in an ignored local SOURCE.json.
- There was no executable comparison of a new run against frozen tolerances.
- The LUAD dependencies were captured in an explicit, independently installable
  lock file.
- Cohort rows are now sorted by clinical case identifier, so seeded splits do
  not change when GDC restores use UUID filenames instead of legacy filenames.
- Fresh restorations are verified through their checksummed RESTORE.json receipt.

### Remaining non-blocking debt

- Historical KIRC and deprecated LUAD configurations still contain absolute
  paths; they are outside the active publication route.
- The repository has no hosted CI workflow for the data-free test suite.
- The lock describes the verified direct environment but is not a
  hash-validated wheel lock across platforms.
- The verified Linux environment occupies about 5.1 GB because the default
  PyTorch wheel installs CUDA runtime packages. A CPU-only, platform-specific
  lock or removal of the unused Torch import is needed to reduce this.
- Runtime seconds and run paths intentionally differ between reproductions;
  scientific comparison uses declared aggregate fields instead of byte identity.
- GDC availability is an external dependency. Frozen UUID and MD5 failure must
  stop reproduction rather than trigger cohort substitution.

## Publication gates

The engineering/reproducibility gate passes only when a clean clone can:

1. install exclusively from requirements-luad.lock;
2. restore and checksum the frozen source inventory;
3. run the active YAML with no errors;
4. pass compare_luad_reproduction.py;
5. pass the data-free unit and contract suites.

The scientific publication gate remains open until at least one external cohort
is selected and frozen before model evaluation, mapped at the same prediction
time, evaluated without tuning, and reported with discrimination, censoring-aware
calibration, missingness and population-shift limitations.
