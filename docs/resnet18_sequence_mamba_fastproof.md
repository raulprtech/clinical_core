# Fast Proof: ResNet18 2.5D + attention/Mamba

## Objetivo

Comparar, sobre exactamente los mismos pacientes y particiones de
supervivencia:

1. resnet18_2p5d_baseline: el embedding VISION-L0 histórico de tres vistas
   centrales y una cabeza Cox lineal.
2. resnet18_2p5d_attention: ventanas axiales 2.5D, ResNet18 congelada,
   attention pooling y cabeza Cox.
3. resnet18_2p5d_mamba: los mismos tokens y pooling que el punto anterior,
   con dos bloques selectivos Mamba antes del pooling.

STU-Net queda fuera de este Fast Proof.

## Diseño sin leakage

La extracción de secuencias no recibe desenlaces. Produce como máximo 64
tokens ordenados sobre el eje físico del stack, cada uno con 512 características
ResNet18 L2-normalizadas. Los tokens se muestrean uniformemente, conservando ambos
extremos del volumen.

Attention, Mamba y sus cabezas Cox se inicializan y entrenan dentro de cada
partición externa. La parada temprana usa únicamente el subgrupo de validación
del outer-train. El held-out se evalúa una sola vez.

Attention y Mamba comparten tokens, proyección, posición relativa y gated
attention pooling. Por eso Mamba - attention es el contraste limpio para
dependencias longitudinales. La comparación de cualquiera de ellos contra el
baseline también cambia la cobertura (volumen axial frente a tres vistas
centrales), por lo que no debe atribuirse sólo al agregador.

## 1. Materializar el caché secuencial

Los DICOM descargados se conservan bajo data/raw/tcia_kirc_dicom:

    python code/tools/build_resnet_sequence_embeddings.py \
      --series-manifest data/manifests/tcia_kirc/series_selected.csv \
      --dicom-dir data/raw/tcia_kirc_dicom \
      --output-dir data/embeddings/vision/resnet18_2p5d_sequences \
      --weights-dir data/models/torch \
      --device auto \
      --max-tokens 64

El proceso es reanudable por paciente. Guarda float16 por defecto para reducir
almacenamiento, vuelve a normalizar en la evaluación y registra fallos y
procedencia.

## 2. Ejecutar el benchmark pareado

    python code/tools/evaluate_resnet_sequence_models.py \
      --baseline-embeddings data/embeddings/vision/vision_resnet18_2p5d_embeddings_768.csv \
      --sequence-dir data/embeddings/vision/resnet18_2p5d_sequences \
      --targets results/20260715_174428_6da68b83/raw_targets.csv \
      --output-dir results_vision/resnet18_attention_mamba_fastproof \
      --device auto

Valores por defecto: semillas 42, 123, 456, 789 y 1024; hold-out 20%;
validación 20% del outer-train; 200 épocas; paciencia 20; model_dim=128; dos
bloques Mamba y estado de dimensión 16.

El baseline histórico PCA train-only + Cox ridge se reproduce sobre los mismos
outer splits con:

    python code/tools/evaluate_resnet_official_comparison.py \
      --baseline-embeddings data/embeddings/vision/vision_resnet18_2p5d_embeddings_768.csv \
      --targets results/20260715_174428_6da68b83/raw_targets.csv \
      --benchmark-dir results_vision/resnet18_attention_mamba_fastproof

Los resultados ejecutados se documentan en
docs/resnet18_sequence_mamba_preliminary_results.md.

## Artefactos

- cohort_common.csv: cohorte exacta evaluada.
- splits.csv: pertenencia train/validation/held-out por semilla.
- heldout_predictions.csv: riesgos pareados de los tres modelos.
- per_seed_metrics.csv: C-index, deltas, parámetros y parada temprana.
- summary.json: agregados y limitaciones.
- provenance.json: hashes de entradas y argumentos efectivos.

## Implementación Mamba

sequence_pooling.py contiene un selective state-space block de referencia en
PyTorch: B, C y delta dependen de cada token, mientras A es diagonal y estable.
Esto conserva la hipótesis arquitectónica y funciona sin compilar mamba-ssm.
No utiliza sus kernels CUDA fusionados; si el Fast Proof es positivo, reemplazar
el scan de referencia por esos kernels sería una optimización de velocidad que
debe validarse numéricamente por separado.
