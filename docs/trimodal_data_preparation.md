# Preparación de datos trimodales TCGA-KIRC

## Cohorte

La extracción clínica contiene 537 IDs TCGA-KIRC. El protocolo oficial de
supervivencia conserva 444 casos con `survival_days > 0`, incluidos 139
eventos. Los adaptadores aplican disponibilidad por modalidad, por lo que la
ausencia de imagen o informe no reduce los experimentos unimodales.

## Fuentes y artefactos

| Modalidad | Fuente | Artefacto local |
|---|---|---|
| Tabular | GDC, Clinical Supplement BCR XML | `data/raw/clinicalsupplement/` |
| Texto | TCGA-Reports (Kefeli y Tatonetti), CC BY 4.0 | `data/embeddings/text_embeddings_TCGA-KIRC_20260528.npz` |
| Visión | TCIA TCGA-KIRC v3, CC BY 3.0 | `data/raw/tcia_kirc_dicom/` |

La colección radiológica debe citarse mediante el DOI
[`10.7937/K9/TCIA.2016.V6PBVTDR`](https://doi.org/10.7937/K9/TCIA.2016.V6PBVTDR).
El catálogo y la selección exacta quedan en `data/manifests/tcia_kirc/`.

## Descarga reanudable de TCIA

El descargador filtra automáticamente el CSV de targets a OS válido, descarta
localizers, exige al menos 16 cortes y selecciona una serie por paciente con el
mismo ranking de los notebooks: CT antes que MR, indicio renal/abdominal y
mayor número de cortes.

```bash
.venv/bin/python code/tools/download_tcia_kirc.py \
  --case-ids-from results/20260715_174428_6da68b83/raw_targets.csv \
  --workers 4
```

Cada serie terminada contiene `.complete.json`. Repetir el comando sólo procesa
las series incompletas. En caso de interrupción no se debe lanzar una segunda
instancia simultánea; el siguiente intento puede reanudar cuando la anterior
haya terminado.

## Embeddings visuales

Los pesos ImageNet se guardan en `data/models/torch/`. Una vez terminada la
descarga DICOM se generan los tres caches con:

```bash
.venv/bin/python code/tools/build_vision_embeddings.py \
  --series-manifest data/manifests/tcia_kirc/series_selected.csv \
  --dicom-dir data/raw/tcia_kirc_dicom \
  --output-dir data/embeddings/vision \
  --weights-dir data/models/torch
```

El proceso guarda checkpoints CSV cada diez pacientes y puede reanudarse. La
configuración trimodal usa por defecto el cache ResNet18-2.5D, mientras que los
otros dos permiten repetir la comparación cambiando únicamente
`vision_embeddings_csv` y `vision_conn`.

## Corrida

```bash
.venv/bin/python code/main.py \
  --config code/experiments/experiment_config_trimodal_vision_2p5d.yaml
```

La configuración evalúa tabular, texto, visión y la combinación completa con
las mismas cinco semillas y cinco folds.
