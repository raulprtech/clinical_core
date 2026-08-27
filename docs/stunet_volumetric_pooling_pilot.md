# Piloto STU-Net-S: pooling volumétrico renal

## Estado

En ejecución local. La implementación, la prueba matemática y un smoke test real
están completos. La extracción de la cohorte estricta queda diseñada como un
proceso reanudable antes de evaluar supervivencia.

## Pregunta preespecificada

¿El STU-Net-S congelado mejora su representación pronóstica si se conserva una
medida de heterogeneidad volumétrica, en vez de reducir todo el volumen a medias?

La comparación primaria usa **la misma inferencia 3D, segmentación, ROI y casos**:

1. `mean_768` (control histórico): media de 256 canales en ROI renal con margen,
   riñón izquierdo y riñón derecho; concatenación y normalización L2.
2. `renal_moments_512` (candidato): media de 256 canales y desviación estándar de
   256 canales en la ROI renal con margen; concatenación y normalización L2.

El candidato no usa desenlaces, etiquetas tumorales, proyecciones aprendidas ni
fine-tuning. La desviación estándar se obtiene acumulando primer y segundo momento
con exactamente las mismas máscaras y pesos gaussianos de los mosaicos 3D.

## Cohorte y costo

La lista fija es
`data/embeddings/vision/stunet_turboconv_preliminary_20x50/splits/eligible_qc_pass.csv`.
Contiene 76 CT que aprobaron el control estricto de geometría DICOM. Al cruzarla
con `results/20260715_174428_6da68b83/raw_targets.csv` hay 20 eventos.

El manifiesto predice 1,483 mosaicos (mediana 18 por caso, máximo 36). El smoke
test `TCGA-B0-4839` procesó 18 mosaicos en 647.0 s, con pico de 2.18 GiB de VRAM
y 5.23 GiB de RSS. La extrapolación conservadora es de unas 15 horas; el proceso
guarda un `complete.json` por paciente y reanuda sin repetir casos válidos.

Artefactos ignorados por Git:

- `data/embeddings/vision/stunet_volumetric_moments_pilot_76/`
- `stunet_s_fp32_embeddings_768.csv`
- `stunet_s_fp32_renal_moments_512.csv`
- máscaras, métricas y procedencia por caso

## Evaluación sin fuga

`code/tools/evaluate_stunet_volumetric_pooling.py` ejecuta una comparación pareada:

- 5 repeticiones de validación cruzada externa estratificada de 5 folds;
- PCA ajustado sólo en entrenamiento;
- selección interna (3 folds) de 4 u 8 componentes y penalización ridge Cox
  (`alpha` 100, 10 o 1);
- la misma familia de cabeza para ambas representaciones;
- C-index por fold externo y delta pareada en 25 folds;
- bootstrap por paciente de riesgos convertidos a rango dentro de cada fold para
  evitar mezclar escalas Cox entrenadas por separado.

Es un análisis exploratorio interno. Un intervalo que cruce cero o resultados
inestables implicarán no adoptar el nuevo pooling.

## Auditoría de supuesto fine-tuning previo

No se encontró evidencia de una corrida real de fine-tuning ligero de STU-Net-S.
Lo que sí está documentado y/o ejecutado es:

- STU-Net-S congelado con checkpoint TotalSegmentator (`docs/stunet_fp32_pilot.md`);
- evaluación preliminar de supervivencia congelada sobre 50 casos/13 eventos
  (`docs/stunet_vs_resnet18_2p5d_preliminary_survival.md`);
- fake quantization W4A8 y TurboConv, sin beneficio suficiente
  (`docs/stunet_turboconv_preliminary.md`);
- un **scaffold** no ejecutado para KiTS23 en
  `code/components/adapters/ingestion/vision/utils/finetune_vision_kits23.py`.

Ese scaffold requiere datos KiTS23 preprocesados, variables de nnUNetv2, el trainer
STU-Net parcheado y al menos 8 GiB de VRAM (12 GiB recomendados). La GPU local
tiene 4 GiB. Por tanto, presentar sus instrucciones como un experimento realizado
sería incorrecto. En esta PC tiene mejor relación evidencia/costo probar pooling
con pesos congelados; un fine-tuning real debe hacerse en Colab u otra GPU mayor.

## Reproducción

Extracción reanudable:

```bash
.venv/bin/python code/tools/build_stunet_embeddings.py \
  --case-id-file data/embeddings/vision/stunet_turboconv_preliminary_20x50/splits/eligible_qc_pass.csv \
  --output-root data/embeddings/vision/stunet_volumetric_moments_pilot_76 \
  --limit 0 --precision amp --step-size 1.0
```

Evaluación al completar 76 casos:

```bash
.venv/bin/python code/tools/evaluate_stunet_volumetric_pooling.py \
  --embedding-dir data/embeddings/vision/stunet_volumetric_moments_pilot_76 \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --cohort data/embeddings/vision/stunet_turboconv_preliminary_20x50/splits/eligible_qc_pass.csv \
  --output-dir results_vision/stunet_volumetric_pooling_nested_76
```

## Criterio de decisión

Se adopta `renal_moments_512` sólo si la mejora es consistente: delta media positiva,
mayoría clara de folds positivos y bootstrap por paciente compatible con beneficio.
Una mejora puntual pequeña, intervalo amplio o alta dependencia del fold se registra
como resultado negativo y mantiene el pooling histórico.
