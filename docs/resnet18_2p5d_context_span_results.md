# Resultado: separación del contexto ResNet18 2.5D

Fecha de ejecución: 2026-08-27.

## Conclusión

Aumentar la separación axial entre los tres canales no mejoró Mamba. La
referencia adyacente `[-1, 0, +1]` conservó el mejor C-index OOF: **0.6868**,
frente a 0.6536 para `[-2, 0, +2]` y 0.6466 para `[-4, 0, +4]`.

Ninguna variante cumple la regla predeclarada para reemplazar la referencia.
Los intervalos de los deltas rozan cero, por lo que no se afirma daño
confirmatorio; la evidencia sí descarta promover una separación mayor a partir
de esta cohorte.

## Protocolo

El diseño se selló antes de extraer las variantes nuevas y antes de evaluar sus
desenlaces en `docs/resnet18_2p5d_context_span_protocol.md` (commit `beeb929`).

- Cohorte común: 214 pacientes y 64 eventos; 190 CT/53 eventos y 24 MR/11
  eventos.
- Tres cachés outcome-independent con las mismas series, 214/214 casos y cero
  fallos.
- ResNet18 ImageNet1K V1 congelada, ventana renal `[-150, 250]`, 64 tokens.
- Mamba-64 unidireccional, dos bloques y posición explícita apagada.
- Outer CV 5 folds x 3 repeticiones, inner CV de 3 folds para época y reajuste
  completo del outer-train.
- 5,000 bootstraps pareados agrupados por paciente.
- Mismos splits, semillas e inicializaciones entre separaciones.

Como control de reproducción, las 642 predicciones OOF de `span1` fueron
idénticas bit por bit a `risk_mamba_t64_posoff` de la ablación factorial previa
(máxima diferencia absoluta 0.0).

## Resultado global

| Contexto por token | C-index OOF medio | DE entre repeticiones | Repeticiones |
|---|---:|---:|---|
| `[-1,0,+1]` | **0.6868** | 0.0100 | 0.6954, 0.6758, 0.6892 |
| `[-2,0,+2]` | 0.6536 | 0.0190 | 0.6752, 0.6462, 0.6393 |
| `[-4,0,+4]` | 0.6466 | 0.0448 | 0.5951, 0.6682, 0.6765 |

| Contraste | Delta | IC95% agrupado | p bootstrap |
|---|---:|---:|---:|
| `span2 - span1` | -0.0332 | [-0.0675, +0.0006] | 0.0548 |
| `span4 - span1` | -0.0402 | [-0.0840, +0.0020] | 0.0632 |
| `span4 - span2` | -0.0070 | [-0.0389, +0.0263] | 0.7080 |

Por folds, `span2` ganó 5 y perdió 10 frente a `span1`; `span4` ganó 6 y perdió
9. La mayor separación también aumentó la variabilidad entre repeticiones.

## Subgrupos diagnósticos

| Contexto | CT | MR |
|---|---:|---:|
| `[-1,0,+1]` | **0.7076** | **0.5676** |
| `[-2,0,+2]` | 0.6817 | 0.4595 |
| `[-4,0,+4]` | 0.6766 | 0.4527 |

CT sigue la conclusión global. MR no permite inferencia separada por sus 24
casos, 11 eventos y alta variabilidad.

## Interpretación y decisión

- Conservar `[-1,0,+1]` como representación 2.5D secuencial operativa.
- No ensayar separaciones adicionales en índices de corte sobre esta cohorte.
- No elegir `span2` o `span4` por algún fold o subgrupo favorable.
- La ablación usa índices, no milímetros; una futura cohorte nueva podría
  evaluar contexto físico normalizado, pero no se justifica abrir esa búsqueda
  post hoc aquí.
- El avance principal continúa siendo validación volumétrica externa o
  fine-tuning 3D con recursos mayores, no más búsqueda 2.5D local.

## Reproducción

~~~bash
.venv/bin/python -u code/tools/evaluate_2p5d_context_span_ablation.py \
  --span1-dir data/embeddings/vision/resnet18_2p5d_sequences \
  --span2-dir data/embeddings/vision/resnet18_2p5d_sequences_span2 \
  --span4-dir data/embeddings/vision/resnet18_2p5d_sequences_span4 \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --modality-manifest data/manifests/tcia_kirc/series_selected.csv \
  --output-dir results_vision/resnet18_2p5d_context_span_nested \
  --outer-folds 5 --outer-repeats 3 --inner-folds 3 \
  --random-state 4049 --epochs 200 --patience 20 \
  --bootstrap-iterations 5000 --device cuda
~~~

Se versionan los agregados `summary.json`, `per_repeat_metrics.csv`,
`per_fold_metrics.csv`, `paired_cluster_bootstrap.csv` y `provenance.json`. Los
IDs, splits y riesgos por paciente permanecen locales y excluidos de Git.
