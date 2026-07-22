# Clinical moments and modality-specific cohorts

The survival system has two distinct evaluation moments:

| Moment | Pathology-report text | Intended interpretation |
|---|---:|---|
| `pre_surgery` | Forbidden | Prognosis before surgical pathology exists |
| `post_surgery` | Allowed | Prognosis after the report is clinically available |

`clinical_context.moment` is enforced by the experiment runner. The default
pathology-derived modality is `text`; it can be extended with
`clinical_context.pathology_modalities`.

## Cohort policy

Use `cohort_filter.modality_policy: per_modality` for multimodal work. Under
this policy, availability never shrinks the global cohort. Each experiment
selects the largest cohort required by its own modality subset:

- tabular: valid survival + tabular;
- text: valid survival + real text report;
- tabular + text: valid survival + both modalities.

The legacy `intersection` policy remains available for explicitly matched
cohort analyses, but should not be used as the default training population.

## Text-only survival protocol

The canonical post-surgery text benchmark is configured in
`code/experiments/experiment_config_text_only_postop_nested_cv.yaml`.

For each seed it uses five outer folds. Each outer-train pool is split again
into inner train and validation sets. Early stopping uses only inner
validation; the outer fold is evaluated once and never participates in model
selection. Cases with zero/missing text embeddings are excluded.

The primary metric is the mean C-index across untouched outer folds. The
pooled OOF diagnostic is retained in the artifacts but is not a primary
metric because independently trained fold models need not produce risk scores
on a common scale.

Current TCGA-KIRC result (Bio_ClinicalBERT off-the-shelf):

- 435 patients with a real report and valid overall survival;
- 138 observed events;
- outer-fold C-index: **0.6837 ± 0.0215 across five seeds**.
