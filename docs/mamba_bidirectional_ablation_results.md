# Ablación Mamba bidireccional con pesos compartidos

Fecha de ejecución: 2026-08-21.

## Pregunta

¿Eliminar la dirección axial arbitraria mejora Mamba-64 sin posición explícita
sin aumentar el número de parámetros?

## Variante

La variante bidireccional aplica los mismos bloques Mamba al orden axial directo
y al inverso, revierte la segunda salida a su orientación original y promedia
ambas representaciones antes de la normalización y el attention pooling. Todos
los pesos son compartidos:

- unidireccional: 319,041 parámetros;
- bidireccional: 319,041 parámetros.

La variante bidireccional no aumenta parámetros, pero ejecuta dos selective
scans. Ambas configuraciones usan 64 tokens uniformes y posición explícita
apagada.

## Protocolo

- 214 pacientes y 64 eventos; 190 CT (53 eventos) y 24 MR (11 eventos).
- 5 outer folds por 3 repeticiones, compartidos entre variantes.
- 3 inner folds para seleccionar época; reinicialización y reajuste con todo el
  outer-train.
- Semilla 4049, idéntica a la ablación factorial.
- 5,000 bootstraps pareados y agrupados por paciente.
- Una predicción OOF por paciente y repetición; riesgos finitos.
- El control unidireccional reprodujo exactamente, bit por bit, las predicciones
  Mamba-64 sin posición de la ablación factorial anterior.

## Resultado

| Variante | Global | DE | CT | MR |
|---|---:|---:|---:|---:|
| Mamba unidireccional | **0.6868** | 0.0100 | **0.7076** | **0.5676** |
| Mamba bidireccional | 0.6731 | 0.0142 | 0.6996 | 0.5383 |

El contraste bidireccional menos unidireccional fue:

- delta medio: **-0.0136**;
- IC95% agrupado: **[-0.0313, +0.0033]**;
- p bootstrap: **0.1080**.

Por fold, la bidireccional tuvo 8 victorias, 2 empates y 5 derrotas. Sus
ganancias fueron pequeñas, mientras una derrota alcanzó -0.1281. En ese fold,
la bidireccional seleccionó 9 épocas frente a 17 de la unidireccional y obtuvo
validación interna similar (0.7546 vs 0.7584). La caída no se explica por haber
seleccionado simplemente demasiadas épocas.

Las cifras CT/MR son diagnósticas. En particular, MR sólo tiene 24 pacientes y
11 eventos y no soporta conclusiones separadas.

## Decisión

- Rechazar la variante bidireccional como configuración operativa.
- Conservar Mamba-64 sin posición unidireccional.
- No afirmar que la bidireccional sea perjudicial en población: el IC95% todavía
  incluye cero. Sí afirmar que no mostró beneficio y añade un segundo scan.
- No invertir en más bloques o mayor dimensión Mamba con esta cohorte.
- Si se continúa localmente, priorizar estabilidad y regularización mediante un
  protocolo nuevo y predeclarado, no mayor capacidad.

## Reproducción

~~~bash
.venv/bin/python -u code/tools/evaluate_mamba_bidirectional_ablation.py \
  --sequence-dir data/embeddings/vision/resnet18_2p5d_sequences \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --modality-manifest data/manifests/tcia_kirc/series_selected.csv \
  --output-dir results_vision/mamba_bidirectional_ablation \
  --max-tokens 64 --outer-folds 5 --outer-repeats 3 --inner-folds 3 \
  --random-state 4049 --epochs 200 --patience 20 \
  --bootstrap-iterations 5000 --device cuda
~~~

## Artefactos

Se versionan `per_fold_metrics.csv`, `per_repeat_metrics.csv`,
`paired_cluster_bootstrap.csv`, `summary.json` y `provenance.json`.
Los archivos a nivel paciente permanecen locales e ignorados por Git.
