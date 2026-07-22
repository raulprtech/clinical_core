# Integración de los baselines VISION-L0

Los tres notebooks de Colab están disponibles como adaptadores `VISION-IN`
intercambiables en Phase 5:

| Notebook | `vision_conn` | Representación antes de 768D |
|---|---|---|
| ResNet18 2D (V4.1) | `vision_resnet18_2d` | `mean3`, 512D, padding a 768D |
| ResNet50 2D (V5) | `vision_resnet50_2d` | `mean3`, 2048D, proyección fija a 768D |
| ResNet18 2.5D (V6) | `vision_resnet18_2p5d` | `mean3`, 512D, padding a 768D |

Todos congelan el backbone ImageNet, extraen las vistas axial, coronal y
sagital centrales y normalizan L2 el resultado. La variante 2.5D usa los
cortes vecinos `[-1, 0, 1]` como canales RGB. Ninguno ajusta PCA, scaler o
proyección usando outcomes; el pronóstico se entrena después dentro de cada
split de Phase 5.

## Opción A: reutilizar los embeddings de Colab

Es la vía más rápida y reproduce exactamente los artefactos ya calculados.
Configura el CSV `vision_embedding_patient_level_all_usable.csv` exportado por
el notebook:

```yaml
phase_5_multimodal:
  enabled: true
  modalities: [tabular, text, vision]
  modality_dim: 768
  text_conn: text_baseline_docling_clinicalbert
  vision_conn: vision_resnet18_2p5d
  vision_embeddings_csv: /ruta/al/vision_embedding_patient_level_all_usable.csv
  text_dir: /ruta/a/informes
  vision_dir: null
  fusion_proc: fusion_baseline_concat
  prognosis_proc: prognosis_baseline_linear_cox
```

Los IDs se normalizan a mayúsculas. El lector exige `case_id`, exactamente 768
columnas `z000...z767` y acepta opcionalmente `vision_confidence`. Los vectores
se vuelven a normalizar antes de entrar en la caché multimodal.

## Opción B: inferencia desde imágenes

`vision_dir` puede contener volúmenes `.nii`, `.nii.gz` o un árbol DICOM como
`case_id/series_uid/*.dcm`. También se admiten los ficheros DICOM sin extensión
que entrega TCIA. Para cambiar el experimento sólo hay que sustituir
`vision_conn`:

```yaml
phase_5_multimodal:
  enabled: true
  modalities: [tabular, text, vision]
  modality_dim: 768
  vision_conn: vision_resnet18_2p5d
  vision_embeddings_csv: null
  vision_dir: /ruta/a/tcga_kirc_tcia_dicom
  vision_params:
    use_imagenet_weights: true
    image_size: 224
    aggregation: mean3
    window_low: -150
    window_high: 250
    min_slices: 16
    slice_offsets: [-1, 0, 1]
    projection_seed: 2026
    device: auto
    weights_dir: /home/raulprtech/clinical_core/data/models/torch
```

La primera ejecución con pesos ImageNet puede requerir descargarlos. Para un
smoke test sin descarga se puede usar `use_imagenet_weights: false`, aunque no
es comparable con los resultados oficiales de los notebooks.

## Comparación controlada

Para comparar arquitectura y contexto, ejecuta tres corridas con el mismo
archivo de configuración, semillas, folds y cohortes; cambia únicamente
`vision_conn` (o el `vision_embeddings_csv` correspondiente). Phase 5 produce
`phase5_multimodal_ablation.csv`, incluida la fila
`tabular+text+vision`, y conserva el manifest de disponibilidad por paciente.
