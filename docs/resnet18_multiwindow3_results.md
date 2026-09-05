# Resultado: CT multi-window con ResNet18 2.5D secuencial

## Conclusión

La fusión fija de tres ventanas CT produjo el mejor C-index interno observado
para el ensamble Mamba+attention (**0.7139**), pero no superó de forma confirmada
ni material al ensamble single-window (**0.7069**). El delta primario fue
**+0.0069**, IC95% **[-0.0037, +0.0182]**, p=0.2052. No alcanza el umbral práctico
predeclarado de +0.02 y no sustituye al candidato anterior.

Attention mostró una señal secundaria favorable de +0.0129, IC95%
[+0.0018, +0.0241], p=0.0228. Esta comparación no era la primaria, forma parte
de varias lecturas de modelo/subgrupo y su magnitud sigue por debajo de +0.02;
se registra como hipótesis, no como confirmación para seleccionar arquitectura.

## Intervención predeclarada

El extractor baseline permanece intacto. La variante aplica a cada token CT las
siguientes ventanas HU, elegidas antes de consultar outcomes:

1. abdominal actual: `[-150, 250]`;
2. renal: `[-73, 304]`;
3. amplia: `[-200, 500]`.

Cada ventana conserva los tres cortes vecinos `[-1, 0, +1]` como canales de la
entrada ResNet18. El backbone ImageNet permanece congelado. Las tres salidas
512D se normalizan, promedian con pesos iguales y vuelven a normalizarse. Así,
Mamba y attention conservan exactamente el mismo número de parámetros. MRI usa
una sola pasada con el preprocesamiento histórico; sus tokens son numéricamente
equivalentes al baseline dentro de precisión float16.

Las ventanas renal y amplia se apoyan en rangos publicados para procesamiento
de CT renal: [2.5D renal `[-73, 304]`](https://pmc.ncbi.nlm.nih.gov/articles/PMC8938741/)
y [segmentación renal `[-200, 500]`](https://pmc.ncbi.nlm.nih.gov/articles/PMC11489355/).

## Auditoría del caché

- 214/214 pacientes y 13,526 tokens, sin fallos.
- Mismos IDs, SeriesInstanceUID, cantidades de tokens y posiciones axiales.
- CT: coseno medio por paciente/token 0.9908; mínimo token 0.9556.
- MR: coseno medio 1.0000.
- Outcomes ausentes durante toda la extracción.

## Protocolo de evaluación

- 214 pacientes, 64 eventos; CT 190/53 y MR 24/11.
- 5 outer folds x 3 repeticiones; 3 inner folds.
- Misma semilla 4049, splits, arquitectura, optimizador y selección de época.
- Mamba-64 y attention-32 sin posición explícita.
- Ensamble 50/50 con ECDF ajustada exclusivamente en outer-train.
- Bootstrap agrupado por paciente, 5,000 iteraciones.

## Resultados agregados

| Modelo | Single | Multi-window | Delta pareado | IC95% | p |
|---|---:|---:|---:|---:|---:|
| Mamba-64 | 0.6868 | 0.6783 | -0.0085 | [-0.0421, +0.0259] | 0.6172 |
| Attention-32 | 0.6751 | 0.6880 | +0.0129 | [+0.0018, +0.0241] | 0.0228 |
| Ensamble percentil-train | 0.7069 | **0.7139** | +0.0069 | [-0.0037, +0.0182] | 0.2052 |
| Ensamble riesgo crudo | 0.6903 | 0.6869 | -0.0034 | [-0.0288, +0.0222] | 0.7924 |
| Ensamble z-train | 0.7042 | 0.7096 | +0.0055 | [-0.0066, +0.0175] | 0.3724 |

Para el ensamble primario, CT pasó de 0.7234 a 0.7307: +0.0073, IC95%
[-0.0043, +0.0194], p=0.2120. MR pasó de 0.5484 a 0.5721, pero sólo contiene
24 pacientes/11 eventos y su IC de delta fue [-0.0307, +0.0856].

## Decisión

- No promover multi-window a baseline ni seguir ajustando límites HU sobre esta
  cohorte; hacerlo convertiría una ablación predeclarada en búsqueda post hoc.
- Mantener el ensamble single-window 0.7069 como referencia operativa y Mamba-64
  como referencia simple.
- Conservar multi-window attention como hipótesis secundaria para una cohorte
  nueva, sin reclamar superioridad.
- Avanzar a la siguiente prueba 2D/2.5D de mayor prioridad: backbone con
  preentrenamiento médico, manteniendo caché, cohortes y evaluador congelados.

## Reproducción

```bash
python3 code/tools/build_resnet_multiwindow_sequence_embeddings.py \
  --series-manifest data/manifests/tcia_kirc/series_selected.csv \
  --dicom-dir data/raw/tcia_kirc_dicom \
  --device cuda

python3 code/tools/evaluate_train_scaled_sequence_ensemble.py \
  --sequence-dir data/embeddings/vision/resnet18_2p5d_sequences_multiwindow3 \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --modality-manifest data/manifests/tcia_kirc/series_selected.csv \
  --output-dir results_vision/resnet18_multiwindow3_train_scaled_ensemble \
  --device cuda

python3 code/tools/compare_resnet_window_variants.py \
  --baseline-dir results_vision/train_scaled_sequence_ensemble \
  --candidate-dir results_vision/resnet18_multiwindow3_train_scaled_ensemble \
  --output-dir results_vision/resnet18_multiwindow3_vs_single
```

Los artefactos a nivel paciente permanecen ignorados. Los resúmenes, métricas
por fold/repetición, bootstrap, procedencia y auditoría del caché se encuentran
en `results_vision/resnet18_multiwindow3_train_scaled_ensemble/` y
`results_vision/resnet18_multiwindow3_vs_single/`.
