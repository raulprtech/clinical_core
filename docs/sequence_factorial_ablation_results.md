# Ablación factorial de la secuencia ResNet18

Fecha de ejecución: 2026-08-21.

## Pregunta

¿Qué parte del diseño secuencial explica el rendimiento visual: la arquitectura
attention/Mamba, conservar 32 o 64 cortes, o inyectar explícitamente la posición
axial?

## Protocolo

- Cohorte: 214 pacientes, 64 eventos; 190 CT (53 eventos) y 24 MR
  (11 eventos).
- Ocho configuraciones pareadas: attention/Mamba, 32/64 tokens y posición
  apagada/encendida.
- Las secuencias mayores al límite se muestrean uniformemente, conservando los
  extremos. Todas usan el mismo cache ResNet18 congelado.
- Outer CV: 5 folds por 3 repeticiones, estratificada por evento y compartida
  entre configuraciones; semilla 4049.
- Inner CV: 3 folds para seleccionar época; reinicialización y reajuste sobre
  todo el outer-train antes de predecir el held-out.
- Incertidumbre: 5,000 bootstraps pareados y agrupados por paciente.
- Cada paciente tiene una predicción OOF por repetición. Todos los riesgos
  fueron finitos y los conteos CT/MR coincidieron con el manifiesto.

Apagar posición elimina por completo la proyección de coordenadas relativas.
Mamba conserva el orden mediante su selective scan. Attention sin posición es
invariante a permutaciones.

## Resultado global

| Configuración | C-index OOF medio | DE entre repeticiones |
|---|---:|---:|
| Mamba, 64, sin posición | **0.6868** | 0.0100 |
| Attention, 32, sin posición | 0.6751 | 0.0199 |
| Attention, 64, sin posición | 0.6738 | 0.0185 |
| Mamba, 32, sin posición | 0.6722 | 0.0156 |
| Attention, 32, con posición | 0.6718 | 0.0145 |
| Mamba, 64, con posición | 0.6699 | 0.0029 |
| Attention, 64, con posición | 0.6632 | 0.0038 |
| Mamba, 32, con posición | 0.6561 | 0.0185 |

| Contraste principal | Delta | IC 95% agrupado | p bootstrap |
|---|---:|---:|---:|
| Mamba - attention, 64 sin posición | +0.0130 | [-0.0161, +0.0405] | 0.3532 |
| 64 - 32, Mamba sin posición | +0.0146 | [-0.0076, +0.0365] | 0.1744 |
| Posición on - off, Mamba 64 | -0.0169 | [-0.0388, +0.0047] | 0.1196 |
| Posición on - off, attention 64 | -0.0106 | [-0.0263, +0.0050] | 0.1904 |

Ninguno de los 12 contrastes excluyó cero. La clasificación de medias favorece
Mamba con 64 tokens y sin posición, pero no demuestra una ventaja aislable de
arquitectura, longitud o coordenadas explícitas.

## CT y MR

| Configuración | CT | MR |
|---|---:|---:|
| Mamba, 64, sin posición | **0.7076** | **0.5676** |
| Mamba, 32, sin posición | 0.6831 | 0.5653 |
| Attention, 32, sin posición | 0.6908 | 0.5270 |
| Attention, 64, sin posición | 0.6871 | 0.5338 |

CT reproduce la preferencia global. MR es sólo diagnóstico: contiene 24
pacientes y 11 eventos, sus resultados por repetición son variables y no se
calcularon contrastes confirmatorios separados por modalidad.

## Decisión

- Usar **Mamba, 64 tokens y sin posición explícita** como configuración
  operativa para el próximo experimento local.
- Mantener attention-32 sin posición como control compacto; queda a 0.0117 de
  la mejor media global.
- No afirmar superioridad de Mamba, beneficio de 64 tokens ni daño de la
  posición: todos los intervalos pareados incluyen cero.
- No reemplazar post hoc el resultado confirmatorio anterior de Mamba 0.7030.
  Esta corrida cambia semilla y espacio de configuraciones y es una ablación.
- Predeclarar Mamba-64 sin posición para una validación nueva o externa. No
  seguir ampliando capacidad con la cohorte actual.

## Reproducción

~~~bash
.venv/bin/python -u code/tools/evaluate_sequence_factorial_ablation.py \
  --sequence-dir data/embeddings/vision/resnet18_2p5d_sequences \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --modality-manifest data/manifests/tcia_kirc/series_selected.csv \
  --output-dir results_vision/sequence_factorial_ablation \
  --outer-folds 5 --outer-repeats 3 --inner-folds 3 \
  --random-state 4049 --epochs 200 --patience 20 \
  --bootstrap-iterations 5000 --device cuda
~~~

## Artefactos

Se versionan `per_fold_metrics.csv`, `per_repeat_metrics.csv`,
`paired_cluster_bootstrap.csv`, `summary.json` y `provenance.json`.
Los tres CSV a nivel paciente permanecen locales y excluidos de Git.
