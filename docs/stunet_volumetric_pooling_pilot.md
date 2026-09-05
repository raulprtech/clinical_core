# Piloto STU-Net-S: pooling volumétrico renal

## Estado

Completado para 75 de 76 CT elegibles. Un volumen de 820 cortes fue excluido por
un límite técnico preespecificable (OOM durante el resampling nnU-Net en 6.7 GiB
de RAM), antes de evaluar resultados. La comparación anidada sobre 75 casos y 20
eventos está completa.

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

El manifiesto predijo 1,483 mosaicos (mediana 18 por caso, máximo 36). Se
completaron 75 pacientes sin fallos de inferencia. `TCGA-B0-5115` no pudo superar
el resampling: el kernel registró OOM con ~5.97 GiB de RSS del preprocesador,
incluso sin cargar la red. La exclusión fue exclusivamente técnica y el caso era
censurado (`event=0`), comprobado sólo después de fijar la exclusión.

El proceso guardó un `complete.json` por paciente y reanudó sin repetir casos
válidos. El pico de GPU permaneció aproximadamente entre 2.12 y 2.18 GiB; el
cuello de botella del caso excluido fue RAM de sistema, no VRAM.

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

## Resultado final (75 casos, 20 eventos)

La evaluación produjo 25 folds externos, todos con 4 eventos en test:

| Representación | C-index medio por fold | Mediana | DE | C-index por rangos OOF promediados |
|---|---:|---:|---:|---:|
| `mean_768` histórico | 0.537 | 0.533 | 0.132 | 0.538 |
| `renal_moments_512` | **0.671** | **0.667** | 0.133 | **0.691** |

La diferencia candidata menos histórica fue:

- media por fold: **+0.134**;
- mediana por fold: **+0.083**;
- folds positivos: **21/25** (uno adicional empatado);
- promedio positivo en cada una de las 5 repeticiones;
- delta OOF por bootstrap de pacientes: **+0.153**;
- IC95% bootstrap del delta: **[+0.025, +0.292]**;
- proporción bootstrap con delta positivo: **0.9888**.

La mejora no dependió de una sola repetición: sus deltas medios fueron +0.084,
+0.189, +0.117, +0.135 y +0.147. La selección interna repartió configuraciones
entre 4/8 componentes y `alpha` 1/10/100; por tanto, no hay evidencia de que el
resultado dependa de un único hiperparámetro.

### Conclusión

**Sí vale la pena conservar esta modificación de STU-Net.** El problema del
pooling histórico era plausible: tres medias eliminaban la heterogeneidad del
volumen. Añadir la desviación estándar del bottleneck renal mejora de forma
consistente la discriminación interna sin fine-tuning, sin etiquetas de desenlace
en el extractor y reduciendo la dimensión de 768 a 512.

La conclusión sigue siendo exploratoria: son 75 pacientes, 20 eventos y no hay
validación externa. No debe compararse directamente el C-index 0.671 con el 0.632
del informe antiguo, porque aquel usó 50 casos, holdouts 80/20 y una cabeza distinta.
El contraste defendible aquí es sólo el pareado `renal_moments_512` vs `mean_768`
bajo exactamente el mismo protocolo.

Artefactos finales:

- `results_vision/stunet_volumetric_pooling_nested_75/summary.json`;
- `fold_metrics.csv`, `outer_predictions.csv` y `cohort_complete.csv` en el mismo directorio.

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

Evaluación final sobre los casos completos:

```bash
.venv/bin/python code/tools/evaluate_stunet_volumetric_pooling.py \
  --embedding-dir data/embeddings/vision/stunet_volumetric_moments_pilot_76 \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --cohort data/embeddings/vision/stunet_turboconv_preliminary_20x50/splits/eligible_qc_pass.csv \
  --output-dir results_vision/stunet_volumetric_pooling_nested_75
```

## Criterio de decisión

El criterio se cumplió: delta media positiva, 21/25 folds positivos, las cinco
repeticiones con promedio positivo y bootstrap por paciente cuyo IC95% del delta
no cruza cero. Se adopta `renal_moments_512` como representación STU-Net preferida
para los experimentos posteriores, manteniendo `mean_768` como control histórico.
