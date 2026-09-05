# Auditoría de MMIST/CPTAC-CCRCC para validación externa

## Conclusión ejecutiva

MMIST completo no es una cohorte externa independiente para un modelo entrenado
con TCGA-KIRC: 429 de sus 618 pacientes proceden de TCGA-KIRC. La porción
CPTAC sí es independiente por fuente y sus 189 identificadores enlazan con GDC,
pero la cohorte con imagen tiene muy pocos eventos para una validación visual
confirmatoria.

La consulta pública realizada el 2026-08-21 encontró:

| Cohorte CPTAC | Mapeados | OS válido | Eventos OS | Mediana días |
|---|---:|---:|---:|---:|
| Clínica completa | 189 | 167 | 31 | 1,292.0 |
| CT MMIST | 51 | 50 | 4 | 1,475.0 |
| MRI MMIST | 5 | 5 | 0 | 1,174.0 |
| CT o MRI | 55 | 54 | 4 | 1,468.0 |
| CT y MRI | 1 | 1 | 0 | 1,472.0 |

Esto corrige una impresión inicial: los dos fallecimientos del endpoint binario
a 12 meses no son todos los eventos disponibles. Al usar `days_to_death` y el
seguimiento completo de GDC aparecen cuatro eventos en CT, tres de ellos después
de 12 meses. Aun así, cuatro eventos no permiten estimar con precisión C-index,
calibración o beneficio sobre un comparador.

## Método de enlace y endpoint

- La cohorte se restringe a identificadores MMIST `C3L-*` y `C3N-*`; se excluye
  por completo `TCGA-*`.
- Los 189 identificadores CPTAC tuvieron correspondencia exacta en el proyecto
  GDC `CPTAC-3`.
- Evento: `vital_status=Dead` y `days_to_death` no negativo.
- Censura: `vital_status=Alive` en el máximo valor no negativo entre
  `diagnoses.days_to_last_follow_up`,
  `diagnoses.days_to_last_known_disease_status` y
  `follow_ups.days_to_follow_up`.
- Los campos temporales de GDC usan como origen el diagnóstico inicial. Antes
  de comparar cifras con TCGA-KIRC debe comprobarse que el target interno usa
  el mismo origen temporal.

En 151 casos fue posible reconstruir de GDC un estado inequívoco a 12 meses.
Hubo 150 concordancias y una discrepancia con `vital_status_12` de MMIST. Esto
refuerza la decisión de conservar el endpoint derivado y registrar versiones y
hashes de las fuentes, en lugar de combinar silenciosamente ambos targets.

## Decisión experimental

1. **No descargar todavía todo CPTAC-CCRCC.** Primero debe cerrarse el diseño
   de validación y comprobar la disponibilidad de series utilizables.
2. **No usar MMIST completo como test externo.** El solapamiento TCGA produciría
   contaminación directa.
3. **No presentar los 50 CT/4 eventos como validación confirmatoria.** Puede
   ejecutarse como prueba de transporte y cambio de dominio, con estimaciones e
   intervalos explícitamente exploratorios.
4. **Congelar antes de evaluar:** selección de series, extractor ResNet18 2.5D,
   Mamba-64, attention-32, ensamble percentil-train 50/50 y cualquier regla de
   normalización. No se ajustarán hiperparámetros con CPTAC.
5. **Inventario TCIA comprobado:** amplía CT sin añadir eventos; no cambia la
   decisión de potencia y no justifica descargar los DICOM.

## Comprobación del inventario TCIA actual

La API pública de TCIA devuelve 71 identificadores con alguna serie en la
colección: 64 con CT, 14 con MRI y 60 con RTSTRUCT. De ellos, 57 pertenecen a
la cohorte clínica CPTAC definida por MMIST; los otros 14 se mantienen fuera
porque incluyen diagnósticos no-ccRCC o no clasificados y no deben incorporarse
sin una regla histológica predeclarada.

TCIA aporta dos pacientes CT de la cohorte clínica que no estaban en ningún
mapeo radiológico MMIST y reclasifica como CT un caso que MMIST sólo mapeaba a
MRI. La cohorte CT conocida crece de 51 a 54 casos, con 53 endpoints OS válidos,
pero conserva exactamente cuatro eventos. La actualización de inventario no
mejora la potencia estadística y confirma que no conviene descargar todavía
los DICOM.

## Reproducción

La auditoría descarga únicamente metadatos públicos y deja los artefactos a
nivel paciente bajo `data/`, excluido de Git:

```bash
python3 code/tools/audit_mmist_cptac_survival.py \
  --work-dir data/external/mmist_cptac_audit
```

El primer programa genera `patient_level_audit.csv` y `summary.json`, además
de hashes SHA-256 de cada entrada. Después se audita el inventario TCIA:

```bash
python3 code/tools/audit_tcia_cptac_inventory.py \
  --work-dir data/external/mmist_cptac_audit
```

Los agregados están versionados en
`results_external/mmist_cptac_survival_audit/`.

## Fuentes

- [MMIST-ccRCC: composición, modalidades y descargas](https://multi-modal-ist.github.io/datasets/ccRCC/)
- [Artículo MMIST: cohorte y endpoint a 12 meses](https://arxiv.org/abs/2405.01658)
- [TCIA: colección CPTAC-CCRCC e inventario radiológico](https://www.cancerimagingarchive.net/collection/cptac-ccrcc/)
- [GDC: campos requeridos para análisis de supervivencia](https://docs.gdc.cancer.gov/Data_Portal/Users_Guide/clinical_data_analysis/)
- [GDC: proyecto CPTAC y acceso](https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac)
