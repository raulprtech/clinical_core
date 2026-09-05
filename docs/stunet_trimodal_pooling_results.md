# STU-Net volumétrico dentro de la fusión trimodal

Fecha: 2026-08-27.

## Pregunta

¿La mejora del embedding `renal_moments_512` frente a `mean_768` se conserva al
combinar visión con variables tabulares y texto, sin cambiar pacientes, splits o
protocolo de selección?

## Protocolo

- Intersección exacta de 72 pacientes y 20 eventos.
- Tres de los 75 casos STU-Net completos (`TCGA-B0-5709`, `TCGA-B0-5711` y
  `TCGA-B0-5712`) no tenían embedding textual válido y se excluyeron por
  disponibilidad de modalidad.
- Cinco folds externos estratificados, repetidos cinco veces.
- Tres folds internos para seleccionar cada Cox penalizado.
- PCA de 4 u 8 componentes para texto y visión, ajustado sólo en train.
- Penalizadores 0.1, 1, 10 y 100.
- Riesgos outer-train cross-fitted y transformados a percentiles empíricos.
- Pesos convexos tabular/texto/visión seleccionados sólo dentro de cada
  outer-train, con paso 0.1.
- Una predicción OOF por paciente y repetición.
- 5,000 bootstraps agrupados por paciente; el mismo paciente se remuestrea en
  las cinco repeticiones.

## Resultado

| Modelo | C-index OOF medio | DE entre repeticiones | Media por fold |
|---|---:|---:|---:|
| Tabular | 0.7764 | 0.0108 | 0.7860 |
| Texto | 0.3703 | 0.0568 | 0.3478 |
| STU-Net `mean_768` | 0.5734 | 0.0561 | 0.5682 |
| STU-Net `renal_moments_512` | **0.7531** | 0.0077 | **0.7406** |
| Fusión con `mean_768` | 0.7567 | 0.0319 | 0.7492 |
| Fusión con `renal_moments_512` | **0.7825** | 0.0251 | **0.7694** |

| Comparación | Delta medio OOF | IC95% agrupado | p bootstrap |
|---|---:|---:|---:|
| Visión moments − visión mean | **+0.1797** | **[+0.0747,+0.2800]** | 0.0008 |
| Fusión moments − fusión mean | +0.0258 | [-0.0060,+0.0599] | 0.1160 |
| Fusión moments − tabular | +0.0061 | [-0.0372,+0.0540] | 0.7680 |
| Fusión moments − visión moments | +0.0294 | [-0.0489,+0.1032] | 0.4580 |

La mejora visual fue positiva en 21/25 folds y en las cinco repeticiones. La
mejora de fusión fue positiva en 12 folds, empató en 10 y fue negativa en 3;
cuatro repeticiones mejoraron y una empeoró.

## Pesos

| Pooling visual | Tabular | Texto | Visión | Visión en cero |
|---|---:|---:|---:|---:|
| `mean_768` | 0.752 ± 0.153 | 0.044 ± 0.082 | 0.204 ± 0.167 | 24% |
| `renal_moments_512` | 0.592 ± 0.229 | 0.052 ± 0.077 | 0.356 ± 0.247 | 16% |

El selector reconoce más señal visual al recibir momentos volumétricos: el peso
medio de visión sube 0.152. Texto continúa débil y recibe peso cero en 64% de
los folds con moments.

## Decisión

1. **Promover `renal_moments_512` como embedding STU-Net operativo.** La mejora
   visual es grande, estable y su intervalo agrupado no cruza cero.
2. **No afirmar que la fusión multimodal mejoró.** El delta +0.0258 es prometedor,
   pero su IC95% cruza cero y una de cinco repeticiones empeora.
3. **No afirmar superioridad sobre tabular.** Fusión moments y tabular son
   estadísticamente compatibles en esta cohorte pequeña.
4. Mantener la fusión convexa como referencia; no añadir atención o
   concatenación de alta dimensión.
5. La ablación bimodal tabular+moments fue predeclarada a partir de este
   resultado y de evidencia anterior de texto inestable; se reporta abajo.

## Ablación posterior: tabular + moments sin texto

Se repitieron los mismos 25 outer folds y toda la selección interna, sustituyendo
la rejilla simplex de tres modalidades por pesos tabular/visión en pasos de 0.1.
Las ramas tabular, texto, mean, moments y fusión trimodal reprodujeron exactamente
sus métricas anteriores.

| Modelo | C-index OOF medio | DE entre repeticiones |
|---|---:|---:|
| Fusión trimodal moments | 0.7825 | 0.0251 |
| **Fusión bimodal tabular+moments** | **0.7890** | **0.0149** |
| Tabular | 0.7764 | 0.0108 |

- Bimodal − trimodal: +0.0065, IC95% [-0.0075,+0.0209], p=0.3804.
- Bimodal − tabular: +0.0126, IC95% [-0.0238,+0.0543], p=0.5192.
- Frente a trimodal, la bimodal ganó 4 folds, empató 20 y perdió 1.
- Pesos bimodales medios: tabular 0.636 y visión 0.364; visión quedó en cero
  en 12% de folds.

**Decisión:** eliminar texto es una simplificación operativa defendible porque no
produce pérdida detectable y reduce una modalidad débil, pero no constituye una
mejora confirmada. `tabular + renal_moments_512` queda como candidato multimodal
parsimonioso; tabular solo permanece como baseline estadísticamente compatible.

Artefactos agregados: `results_vision/stunet_bimodal_moments_nested_72/`.

## Límites

- Cohorte previamente explorada y sólo 20 eventos.
- La intersección trimodal reduce de 75 a 72 pacientes.
- Validación interna; falta una cohorte externa con CT volumétrico comparable.
- El contraste de fusión tiene mucha menos potencia que el contraste visual.

## Artefactos

Agregados versionables en `results_vision/stunet_trimodal_pooling_nested_72/`:

- `per_fold_metrics.csv`;
- `per_repeat_metrics.csv`;
- `paired_cluster_bootstrap.csv`;
- `summary.json`;
- `provenance.json`.

`cohort_common.csv`, `splits.csv` y `heldout_predictions.csv` permanecen locales
porque contienen identificadores o filas por paciente.
