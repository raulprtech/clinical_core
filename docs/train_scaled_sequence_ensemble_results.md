# Ensamble secuencial con escala derivada de outer-train

Fecha de ejecución: 2026-08-21.

## Pregunta

¿La complementariedad Mamba-64 + attention-32 persiste cuando la armonización
de riesgos se estima exclusivamente con outer-train y se aplica sin cambios al
held-out?

## Protocolo

- 214 pacientes y 64 eventos; 190 CT (53 eventos) y 24 MR (11 eventos).
- 5 outer folds por 3 repeticiones y 3 inner folds para seleccionar época.
- Mamba usa 64 tokens; attention 32; ambas sin posición explícita.
- Cada modelo se reinicializa y reajusta con todo outer-train.
- La transformación principal es la ECDF de los riesgos del outer-train. Cada
  riesgo held-out se convierte en el percentil que ocupa respecto a esa ECDF.
- Sensibilidades predefinidas: z-score estimado con outer-train y riesgo crudo.
- Pesos fijos 0.5/0.5; no se optimizan con outcomes.
- 5,000 bootstraps pareados y agrupados por paciente.
- El held-out no participa en selección, reajuste ni estimación de escala.

Los riesgos Mamba y attention reprodujeron exactamente, bit por bit, las
predicciones de la ablación factorial. Cada paciente tuvo una predicción OOF
por repetición y todos los riesgos fueron finitos.

## Resultado

| Modelo | Global | DE | CT | MR |
|---|---:|---:|---:|---:|
| Mamba | 0.6868 | 0.0100 | 0.7076 | **0.5676** |
| Attention | 0.6751 | 0.0199 | 0.6908 | 0.5270 |
| Ensamble percentil-train 50/50 | **0.7069** | **0.0063** | **0.7234** | 0.5484 |
| Ensamble z-score-train 50/50 | 0.7042 | 0.0067 | 0.7194 | 0.5653 |
| Ensamble crudo 50/50 | 0.6903 | 0.0018 | 0.7083 | 0.5631 |

| Comparación | Delta | IC95% | p bootstrap |
|---|---:|---:|---:|
| Percentil-train - Mamba | +0.0201 | [-0.0053, +0.0457] | 0.1256 |
| Percentil-train - attention | +0.0318 | [+0.0108, +0.0518] | 0.0048 |
| Z-score-train - Mamba | +0.0174 | [-0.0076, +0.0425] | 0.1680 |
| Crudo - Mamba | +0.0035 | [-0.0102, +0.0170] | 0.5888 |

El percentil-train ganó 10 de 15 folds frente a Mamba y perdió cinco. Reduce la
DE entre repeticiones y mejora la media global y CT, pero conserva dos pérdidas
grandes. El intervalo frente a Mamba todavía cruza cero. Z-score-train produce
un resultado cercano, mientras el promedio crudo casi no mejora, lo que
confirma la importancia de una regla de escala explícita.

MR no mejora con el ensamble principal. El subgrupo tiene sólo 24 pacientes y
11 eventos y no permite elegir una estrategia específica por modalidad.

## Decisión

- Promover el ensamble percentil-train 50/50 a **mejor candidato visual
  interno** por rendimiento, menor dispersión y procedimiento desplegable.
- Mantener Mamba-64 sin posición como referencia operativa simple: la ventaja
  del ensamble frente a Mamba no está confirmada.
- Predeclarar ECDF outer-train y pesos 0.5/0.5 para cualquier validación nueva;
  no ajustar pesos ni escoger z-score retrospectivamente.
- No interpretar este seguimiento como confirmación independiente: la
  hipótesis surgió al observar la misma cohorte.
- No realizar más búsquedas arquitectónicas sobre estos datos. El siguiente
  paso útil es validación externa o una cohorte realmente nueva.

## Reproducción

~~~bash
.venv/bin/python -u code/tools/evaluate_train_scaled_sequence_ensemble.py \
  --sequence-dir data/embeddings/vision/resnet18_2p5d_sequences \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --modality-manifest data/manifests/tcia_kirc/series_selected.csv \
  --output-dir results_vision/train_scaled_sequence_ensemble \
  --outer-folds 5 --outer-repeats 3 --inner-folds 3 \
  --random-state 4049 --epochs 200 --patience 20 \
  --bootstrap-iterations 5000 --device cuda
~~~

## Artefactos

Se versionan los cinco archivos de métricas/resumen/procedencia y
`training_scaling_by_fold.csv`, que sólo contiene estadísticas agregadas.
`cohort_common.csv`, `splits.csv` y `heldout_predictions.csv` permanecen
locales e ignorados por Git.
