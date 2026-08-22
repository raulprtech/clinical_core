# Diagnóstico de ensamble fijo Mamba + attention

Fecha de ejecución: 2026-08-21.

## Pregunta

¿Los errores de Mamba-64 sin posición y attention-32 sin posición son
complementarios, de modo que un promedio fijo 50/50 mejore el riesgo visual?

## Problema de escala

El riesgo Cox no tiene escala absoluta. Entre folds, la desviación de los
riesgos varió aproximadamente entre 0.9 y 4.0, así que promediar valores crudos
no representa pesos efectivos iguales. Se predefinieron tres lecturas:

- principal: promedio 50/50 de rangos percentiles dentro de cada outer
  held-out fold;
- sensibilidad 1: promedio 50/50 de riesgos crudos;
- sensibilidad 2: promedio 50/50 después de z-score dentro del held-out fold.

Los rangos y z-scores usan la distribución de riesgos no etiquetada del
held-out. Por ello son diagnósticos transductivos post hoc, no un procedimiento
desplegable. Una versión formal tendría que estimar la transformación con
predicciones del outer-train y fijarla antes de evaluar el held-out.

## Protocolo

- Predicciones OOF existentes: 214 pacientes, 64 eventos, 5 folds por 3
  repeticiones.
- Mamba-64 sin posición y attention-32 sin posición.
- Pesos fijos 0.5/0.5; no se optimizó ningún peso con outcomes.
- 5,000 bootstraps pareados y agrupados por paciente.
- Una predicción por paciente y repetición; todos los riesgos fueron finitos.

## Resultado

| Modelo | Global | DE | CT | MR |
|---|---:|---:|---:|---:|
| Mamba | 0.6868 | 0.0100 | 0.7076 | **0.5676** |
| Attention | 0.6751 | 0.0199 | 0.6908 | 0.5270 |
| Ensamble por rangos 50/50 | **0.6997** | 0.0091 | **0.7169** | 0.5552 |
| Ensamble crudo 50/50 | 0.6903 | 0.0018 | 0.7083 | 0.5631 |
| Ensamble z-score 50/50 | 0.7040 | 0.0071 | 0.7211 | 0.5563 |

| Comparación | Delta | IC95% | p bootstrap |
|---|---:|---:|---:|
| Rangos - Mamba | +0.0129 | [-0.0116, +0.0364] | 0.2952 |
| Rangos - attention | +0.0246 | [+0.0035, +0.0457] | 0.0268 |
| Crudo - Mamba | +0.0035 | [-0.0102, +0.0174] | 0.5964 |
| Z-score - Mamba | +0.0172 | [-0.0055, +0.0399] | 0.1252 |

El ensamble por rangos ganó 6 folds, empató 1 y perdió 8 frente a Mamba. Su
mejora agregada depende principalmente de una repetición. Los dos métodos
armonizados mejoran la media, pero ninguno demuestra ventaja frente a Mamba.
El promedio crudo apenas cambia el resultado, confirmando sensibilidad a la
regla de escala. MR no mejora y sólo contiene 24 pacientes/11 eventos.

## Decisión

- No reemplazar Mamba-64 unidireccional como configuración operativa.
- Conservar el ensamble fijo como señal de complementariedad potencial.
- No elegir z-score retrospectivamente por tener la mayor media.
- Si se continúa, realizar una única evaluación nueva donde percentiles o
  z-score se estimen exclusivamente con riesgos del outer-train y se apliquen
  sin cambios al held-out.
- No optimizar pesos sobre estas mismas predicciones.

## Reproducción

~~~bash
.venv/bin/python -u code/tools/evaluate_fixed_sequence_ensemble.py \
  --predictions \
    results_vision/sequence_factorial_ablation/heldout_predictions.csv \
  --output-dir results_vision/fixed_sequence_ensemble \
  --bootstrap-iterations 5000 --random-state 7049
~~~

## Artefactos

Se versionan `per_fold_metrics.csv`, `per_repeat_metrics.csv`,
`paired_cluster_bootstrap.csv`, `summary.json` y `provenance.json`.
Las predicciones derivadas permanecen locales e ignoradas por Git.
