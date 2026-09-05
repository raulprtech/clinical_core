# Resultado: ResNet50 RadImageNet 2.5D secuencial

## Conclusión

ResNet50 RadImageNet **no mejora el candidato visual actual**. El ensamble
primario obtuvo C-index **0.6903**, frente a **0.6719** del control ResNet50
ImageNet y **0.7069** de ResNet18 ImageNet. El efecto específico del contrato de
preentrenamiento médico fue +0.0183, IC95% [-0.0405,+0.0746], p=0.5372: no
alcanza el umbral práctico predeclarado de +0.02 ni demuestra una diferencia.

Frente a ResNet18, RadImageNet50 perdió -0.0167, IC95% [-0.0666,+0.0316],
p=0.4972, y ganó sólo 3 de 15 folds en el ensamble primario. Por tanto, no se
promueve ninguno de los dos ResNet50 y se mantiene ResNet18 single-window como
baseline 2D/2.5D.

## Intervención y control arquitectónico

Se compararon tres extractores congelados:

1. ResNet18 ImageNet1K V1, baseline histórico;
2. ResNet50 ImageNet1K V2, control del aumento de capacidad;
3. ResNet50 RadImageNet, intervención de preentrenamiento médico.

Los dos ResNet50 comparten exactamente las series, ventana CT `[-150,250]`,
normalización MR por percentiles `[1,99]`, vecinos axiales `[-1,0,+1]`, muestreo
uniforme de hasta 64 tokens y una proyección gaussiana fija 2048D -> 512D con
semilla 2026. Esto conserva en los poolers los mismos **84,033 parámetros de
attention** y **319,041 de Mamba** que usa ResNet18.

Cada modelo usa su normalización prescrita. ImageNet usa mean/std de
torchvision; RadImageNet escala `[0,1]` a `[-1,1]`, como en el
[ejemplo PyTorch oficial](https://github.com/BMEII-AI/RadImageNet/blob/main/pytorch_example.ipynb).
En consecuencia, RadImageNet frente a ImageNet estima el efecto del contrato
desplegable `pesos + normalización`, no de los pesos aislados.

El checkpoint oficial ResNet50 se obtuvo del
[repositorio RadImageNet](https://github.com/BMEII-AI/RadImageNet), tiene SHA256
`08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734` y se
cargó con coincidencia estricta de todas las capas. Pesos, DICOM y cachés
permanecen locales y excluidos de Git.

## Auditoría del caché

- 214/214 pacientes y 13,526 tokens por encoder; cero fallos.
- Mismos case IDs, SeriesInstanceUID, longitudes y posiciones axiales.
- Todos los tokens finitos y L2-normalizados después de la proyección fija.
- Coseno ImageNet50/RadImageNet50: 0.1927 medio en CT y 0.1823 en MR.
- Outcomes ausentes durante toda la extracción.

## Protocolo de evaluación

- 214 pacientes y 64 eventos; CT 190/53, MR 24/11.
- 5 outer folds x 3 repeticiones y 3 inner folds.
- Semilla 4049, mismos splits, optimizador, selección de época y dimensiones.
- Mamba-64 y attention-32 sin posición explícita.
- Ensamble fijo 50/50 con ECDF estimada sólo en outer-train.
- Bootstrap agrupado por paciente, 5,000 iteraciones por comparación/subgrupo.

## Resultados agregados

| Encoder | Mamba | Attention | Ensamble primario | CT | MR |
|---|---:|---:|---:|---:|---:|
| ResNet18 ImageNet | **0.6868** | **0.6751** | **0.7069** | **0.7234** | 0.5484 |
| ResNet50 ImageNet | 0.6377 | 0.6499 | 0.6719 | 0.6765 | **0.5822** |
| ResNet50 RadImageNet | 0.6494 | 0.6406 | 0.6903 | 0.7193 | 0.4707 |

Los subgrupos CT/MR de la tabla corresponden al ensamble primario. MR contiene
sólo 24 pacientes y 11 eventos, por lo que sus estimaciones son inestables.

## Comparaciones pareadas del ensamble primario

| Contraste | Todos: delta [IC95%], p | CT: delta [IC95%], p | MR: delta [IC95%], p |
|---|---|---|---|
| RadImageNet50 - ImageNet50 | +0.0183 [-0.0405,+0.0746], 0.5372 | +0.0428 [-0.0144,+0.0991], 0.1460 | -0.1115 [-0.3513,+0.1395], 0.3704 |
| RadImageNet50 - ResNet18 | -0.0167 [-0.0666,+0.0316], 0.4972 | -0.0041 [-0.0556,+0.0456], 0.8716 | -0.0777 [-0.2773,+0.1118], 0.4288 |
| ImageNet50 - ResNet18 | -0.0350 [-0.0719,+0.0021], 0.0660 | -0.0469 [-0.0855,-0.0101], 0.0116 | +0.0338 [-0.0952,+0.1458], 0.5664 |

RadImageNet recupera en CT la degradación que introduce ResNet50 ImageNet, pero
no supera a ResNet18: 0.7193 frente a 0.7234. En MR, el contrato RadImageNet
produce resultados por debajo del azar en las tres repeticiones del ensamble
(0.4054, 0.5676, 0.4392). No hay fundamento para crear un selector CT/MR post
hoc: el baseline ResNet18 ya iguala o mejora el resultado CT y es más estable.

## Decisión

- No promover ResNet50 ImageNet ni RadImageNet.
- Mantener ResNet18 single-window + ensamble percentil-train como referencia
  visual interna (0.7069).
- No ajustar normalización, proyección o capas RadImageNet sobre esta cohorte;
  hacerlo después de observar CT/MR constituiría búsqueda post hoc.
- Cerrar la línea de encoders 2D congelados grandes: ni la capacidad adicional
  ni el preentrenamiento médico general aportaron mejora agregada fiable.
- El siguiente cambio arquitectónico justificable es contexto espacial 3D,
  preservando esta misma evaluación y una comparación contra ResNet18.

## Reproducción

```bash
python3 code/tools/build_resnet50_sequence_embeddings.py \
  --series-manifest data/manifests/tcia_kirc/series_selected.csv \
  --dicom-dir data/raw/tcia_kirc_dicom \
  --output-dir data/embeddings/vision/resnet50_imagenet_2p5d_sequences \
  --pretraining imagenet --device cuda

python3 code/tools/build_resnet50_sequence_embeddings.py \
  --series-manifest data/manifests/tcia_kirc/series_selected.csv \
  --dicom-dir data/raw/tcia_kirc_dicom \
  --output-dir data/embeddings/vision/resnet50_radimagenet_2p5d_sequences \
  --pretraining radimagenet --device cuda

python3 code/tools/evaluate_train_scaled_sequence_ensemble.py \
  --sequence-dir data/embeddings/vision/resnet50_radimagenet_2p5d_sequences \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --modality-manifest data/manifests/tcia_kirc/series_selected.csv \
  --output-dir results_vision/resnet50_radimagenet_train_scaled_ensemble \
  --device cuda

python3 code/tools/compare_sequence_encoder_results.py \
  --baseline-dir results_vision/resnet50_imagenet_train_scaled_ensemble \
  --candidate-dir results_vision/resnet50_radimagenet_train_scaled_ensemble \
  --baseline-label imagenet50 --candidate-label radimagenet50 \
  --output-dir results_vision/resnet50_radimagenet_vs_imagenet
```

Los agregados y la procedencia se conservan bajo
`results_vision/resnet50_*`; las predicciones por paciente siguen ignoradas.
