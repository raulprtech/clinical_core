# Resultado confirmatorio interno: ResNet18 secuencial

Fecha de ejecución: 2026-08-21.

## Pregunta

¿La señal observada en el Fast Proof persiste al eliminar la asimetría de
reajuste y evaluar cada paciente fuera de muestra mediante nested repeated
cross-validation?

## Protocolo

- Cohorte común: 214 pacientes y 64 eventos.
- Outer CV: 5 folds estratificados por evento, repetidos 3 veces.
- Inner CV: 3 folds dentro de cada outer-train.
- PCA+Cox: selección interna de dimensión y penalización, seguida de reajuste
  con todo el outer-train.
- Attention y Mamba: mejor época seleccionada en cada inner fold; se usa la
  mediana de las tres épocas, se reinicializa el modelo y se reajusta con todo
  el outer-train.
- Attention y Mamba usan exactamente los mismos tokens ResNet18 congelados.
- Cada paciente tiene una predicción OOF por repetición.
- Incertidumbre: 5,000 remuestreos de pacientes. El mismo paciente bootstrap se
  reutiliza en las tres repeticiones para respetar el agrupamiento.
- El held-out no participa en selección, early stopping ni reajuste.

## Resultado OOF

| Modelo | C-index OOF medio | DE entre repeticiones | Repetición 1 | Repetición 2 | Repetición 3 |
|---|---:|---:|---:|---:|---:|
| PCA+Cox oficial | 0.6432 | 0.0264 | 0.6574 | 0.6596 | 0.6128 |
| Attention | 0.6957 | 0.0242 | 0.6818 | 0.7237 | 0.6817 |
| Mamba | **0.7030** | **0.0194** | 0.7236 | 0.6850 | 0.7005 |

Mamba superó al baseline oficial en 12 de 15 outer folds, empató uno y perdió
dos. Attention lo superó en 13 de 15 folds.

| Comparación | Delta medio | IC 95% agrupado | p bootstrap |
|---|---:|---:|---:|
| Mamba - PCA+Cox oficial | **+0.0598** | **[+0.0156, +0.1050]** | **0.0100** |
| Mamba - Attention | +0.0073 | [-0.0136, +0.0284] | 0.5076 |

Las épocas seleccionadas confirman regularización temprana: mediana 7 para
Mamba (rango 3--15) y 9 para attention (rango 4--40).

## Interpretación permitida

La ventaja de la tubería secuencial sobre el baseline de tres vistas centrales
persiste bajo evaluación interna más estricta y reajuste simétrico. El
intervalo Mamba--PCA+Cox excluye cero en el bootstrap agrupado.

No hay evidencia de que Mamba sea superior a attention: su diferencia es
pequeña y el intervalo cruza cero. Además, PCA+Cox resume tres vistas centrales
mientras attention y Mamba recorren el stack axial. Por tanto, la evidencia
actual apoya el modelado secuencial de mayor cobertura, no atribuye todavía la
ganancia específicamente al selective SSM.

El C-index absoluto de PCA+Cox es menor que en los cinco hold-outs del Fast
Proof. No es una contradicción: cambian la topología de partición, la cobertura
OOF y el modo de agregación. Las comparaciones válidas son las pareadas dentro
de este protocolo.

## Límites

- Es confirmación por remuestreo interno de la misma cohorte, no validación
  externa.
- Sólo hay tres repeticiones outer; el bootstrap cuantifica incertidumbre por
  paciente, pero no sustituye otra cohorte.
- La variante Mamba usa un selective scan puro de PyTorch, no el kernel
  fusionado de mamba-ssm.
- La cobertura de entrada difiere entre el baseline oficial y los modelos
  secuenciales.
- Falta comprobar si el riesgo secuencial aporta señal incremental en la fusión
  multimodal convexa.

## Reproducción

~~~bash
.venv/bin/python -u code/tools/evaluate_resnet_sequence_nested_cv.py \
  --baseline-embeddings \
    data/embeddings/vision/vision_resnet18_2p5d_embeddings_768.csv \
  --sequence-dir data/embeddings/vision/resnet18_2p5d_sequences \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --output-dir results_vision/resnet18_sequence_nested_cv \
  --outer-folds 5 --outer-repeats 3 --inner-folds 3 \
  --random-state 2026 --epochs 200 --patience 20 \
  --bootstrap-iterations 5000 --device cuda
~~~

## Artefactos

Se versionan únicamente agregados no identificables:

- per_fold_metrics.csv
- per_repeat_metrics.csv
- paired_cluster_bootstrap.csv
- summary.json
- provenance.json

cohort_common.csv, splits.csv y heldout_predictions.csv se conservan
localmente y están excluidos de Git porque contienen información a nivel
paciente.
