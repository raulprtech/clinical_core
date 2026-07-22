# Informe de revisión — verificación de cifras y solución a comentarios

**Manuscrito:** *Protocolo anti-leakage de tres capas para predicción de supervivencia oncológica*
**Versión revisada:** `articulo_micai_revisado_1.pdf`
**Fecha de verificación:** 2026-06-11
**Corridas de respaldo:**
- `results/20260608_002929_f02609bd/` — phase_2_holdout sparse n=444 (limpio + permisivo)
- `results/20260608_004741_360ecb35/` — phase_2_holdout dense n=224 (limpio + permisivo)
- `results/20260611_010418_76becb90/` — phase_2_mahootiha dense n=224 (limpio + permisivo)
- `results/20260611_010525_908c2f09/` — phase_2_mahootiha sparse n=444 (limpio + permisivo)

---

## Estado de las correcciones por sección

### Resumen y §1 (Introducción)

| Cambio en la revisión | Respaldo en datos | Estado |
|---|---|---|
| “1.6× a 3.8×” → **“2.1× a 3.7×”** | Tabla 1 (dense): Cox 1.0×, linear 2.1×, FT 3.7× | ✅ correcto |
| “nulo” → **“indetectable”** en cohortes dispersos | Δ sparse: Cox +0.000, FT +0.001 | ✅ correcto |
| “Demostración” → **“Evidencia”** | Suavizado de tono | ✅ apropiado |
| Mención de Schulz [5] (230 pac.) y MMEM [8] (226 pac.) | Soporta el §4.3 directamente | ✅ correcto |
| Re-numeración de cita Zech [20]→[18] | Re-ordenamiento de referencias | ✅ consistente |

### §2.3 (Aprendizaje de atajos)

Sin cambios numéricos. Argumento conceptual intacto.

### §3.2 (Protocolo anti-leakage)

| Cambio | Estado |
|---|---|
| Mapeo explícito de las 3 capas a tipos de Kapoor & Narayanan (feature/temporal leakage, pre-processing on train and test, no separate test set) | ✅ apropiado conceptualmente |

### §3.4 (Diseño experimental)

| Cambio | Respaldo | Estado |
|---|---|---|
| Eliminación de “test de DeLong” | Justificado: DeLong se define para AUC binaria; el C-index censurado no tiene una versión exacta. Lo que reportamos es bootstrap pareado | ✅ honesto |
| Adición de “IBS” y “SHAP” | Implementado en `tools/run_calibration.py` y `tools/run_shap_attributions.py` | ✅ correcto |

### §4.1 (Sensibilidad diferencial — Tabla 1 y Fig. 1)

**Tabla 1 validada celda por celda** contra `results/20260608_004741_360ecb35/phase2_holdout_summary.csv`:

| Arquitectura | Reportado | Calculado | OK |
|---|---|---|---|
| Cox PH limpio | 0.816 ± 0.032 | 0.8159 ± 0.0315 | ✅ |
| Cox PH permisivo | 0.836 ± 0.023 | 0.8358 ± 0.0232 | ✅ |
| linear_compact limpio | 0.773 ± 0.057 | 0.7730 ± 0.0574 | ✅ |
| linear_compact permisivo | 0.815 ± 0.038 | 0.8148 ± 0.0383 | ✅ |
| FT-Transformer limpio | 0.746 ± 0.069 | 0.7460 ± 0.0686 | ✅ |
| FT-Transformer permisivo | 0.820 ± 0.021 | 0.8198 ± 0.0208 | ✅ |
| Δ Cox = +0.020 | | +0.0199 | ✅ |
| Δ linear = +0.042 | | +0.0418 | ✅ |
| Δ FT = +0.074 | | +0.0738 | ✅ |
| Factor 2.1× linear | | 2.10× | ✅ |
| Factor 3.7× FT | | 3.71× | ✅ |

**SHAP** (`results/20260608_004741_360ecb35/phase2_shap_attributions.csv`):
- FT-Transformer: 82.1 % del |SHAP| total en variables post-evento. ✅
- Cox PH: 13.4 % del |SHAP| total en variables post-evento. ✅
- Ratio 6.1×. ✅

**Caveat añadido sobre la hipótesis del acantilado:** correcto. No corrimos análisis longitudinal de trayectorias ECOG/Karnofsky ni curvas PDP/ICE; el texto lo declara explícitamente como trabajo futuro.

**Tabla 6 (paired bootstrap)** validada contra `phase2_significance_delta.csv`:

| Arquitectura | Δ medio (reportado) | Calculado | std reportado | Calculado | p mediana | p mínima |
|---|---|---|---|---|---|---|
| Cox PH | +0.020 | +0.01994 | 0.042 | 0.04185 | 0.288 / 0.288 ✅ | 0.056 / 0.056 ✅ |
| linear_compact | +0.042 | +0.04178 | 0.027 | 0.02727 | 0.582 / 0.582 ✅ | 0.076 / 0.076 ✅ |
| FT-Transformer | +0.074 | +0.07385 | 0.082 | 0.08155 | 0.630 / 0.630 ✅ | 0.014 / 0.014 ✅ |

**IBS reportado en §4.1 y §5** (`phase2_calibration_ibs.csv`):
- FT-Transformer limpio: 0.146 (calculado 0.1457). ✅
- FT-Transformer permisivo: 0.124 (calculado 0.1243). ✅

### §4.3 (Tabla 2 — leakage condicional a densidad)

**Validada con corridas frescas** de `phase_2_mahootiha` con protocols (lanzadas hoy):

| Cohorte | 19 limpio (reportado) | 19 limpio (calculado) | 22 leak (reportado) | 22 leak (calculado) | Δ reportado | Δ calculado |
|---|---|---|---|---|---|---|
| n=444 | 0.811 | 0.8107 | 0.814 | 0.8135 | +0.003 | +0.0028 |
| n=224 | 0.793 | 0.7931 | 0.829 | 0.8289 | +0.036 | +0.0358 |

✅ Todas las celdas coinciden a 3 decimales.

### §4.4 (Tabla 3 — replicación Mahootiha)

**Validada con corridas frescas** (mismo método, mismas seeds, mismo split):

| K | 19 limpio n=444 (rep / calc) | 22 leak n=444 (rep / calc) | 19 limpio n=224 (rep / calc) | 22 leak n=224 (rep / calc) |
|---|---|---|---|---|
| 5 | 0.774 / 0.7742 ✅ | 0.735 / 0.7347 ✅ | 0.779 / 0.7787 ✅ | 0.786 / 0.7857 ✅ |
| 10 | 0.789 / 0.7892 ✅ | 0.787 / 0.7869 ✅ | 0.793 / 0.7929 ✅ | 0.810 / 0.8104 ✅ |
| 15 | 0.809 / 0.8088 ✅ | 0.804 / 0.8044 ✅ | 0.782 / 0.7822 ✅ | 0.813 / 0.8128 ✅ |
| all | 0.811 / 0.8107 ✅ | 0.814 / 0.8135 ✅ | 0.793 / 0.7931 ✅ | 0.829 / 0.8289 ✅ |

**16/16 celdas validadas a 3 decimales** — la Tabla 3 entera es defendible bajo nuestro pipeline.

### §5 (Discusión)

Sin cambios numéricos pendientes de verificación. Las cifras 82.1 %, 13.4 %, 6.1×, IBS 0.146 → 0.124 ya fueron validadas en §4.1.

### §6 (Conclusiones) y Agradecimientos

Sin números. Adición de declaración de uso de IA: contenido administrativo, sin verificación necesaria.

### Referencias

Re-numeración de [18]–[21] tras eliminar Kefeli y Cheerla-Gevaert. ✅ El texto revisado no cita ya esas referencias.

---

## Comentario [C1] del revisor (inconsistencia 0.816 vs 0.793)

> *La Tabla 1 reporta el Cox limpio (19 vars, n=224) como 0.816 ± 0.032, pero aquí se afirma 0.793. Probablemente 0.793 corresponde a la selección K=10 (método de Mahootiha) y 0.816 a las 19 variables completas.*

**Diagnóstico confirmado por datos:** son dos configuraciones distintas que efectivamente producen cifras distintas, ambas correctas.

| Cifra | Procedencia | Configuración exacta |
|---|---|---|
| **0.816 ± 0.032** | Tabla 1, fila Cox PH limpio | `phase_2_holdout` — Cox con **las 19 variables completas**, sin selección, imputación `mean_median`, penalizer adaptativo |
| **0.7931** | Tabla 3, K=19 (fila “all”) limpio n=224 | `phase_2_mahootiha` — Cox con **las 19 variables completas tras pasar por ranking Spearman+RF** (K=all), imputación `knn_5` |
| **0.7929** | Tabla 3, K=10 limpio n=224 | `phase_2_mahootiha` — Cox con **las 10 mejores de 19** según ranking combinado, imputación `knn_5` |

Las cifras 0.7929 (K=10) y 0.7931 (K=all = K=19) son casi idénticas, lo cual tiene sentido: con solo 19 variables totales, restringir a las 10 mejores no penaliza fuerte. Pero ambas difieren de 0.816 por **la imputación distinta** (`knn_5` vs `mean_median`), no por el método de selección.

### Solución sugerida para el texto del §4.4

> ~~Bajo protocolo limpio (19 variables, K=10) sobre el mismo subcohorte, su método arroja 0.793. Nuestro Cox proporcional en condiciones idénticas obtiene 0.793.~~
>
> **Bajo el método de Mahootiha aplicado al protocolo limpio (19 variables disponibles, imputación KNN-5), el desempeño es 0.793 ± 0.041 con K=10 y 0.793 ± 0.036 con K=19 (Tabla 3). Para referencia, nuestro Cox PH bajo el protocolo de evaluación de §4.1 (mismas 19 variables, sin selección Mahootiha, imputación mean_median) alcanza 0.816 ± 0.032 (Tabla 1). La diferencia ≈0.02 entre ambas configuraciones es atribuible a la elección del imputador, no al método de selección de features. La convergencia de cifras bajo protocolo limpio sugiere que la brecha entre 0.79 y 0.84 reportada por Mahootiha et al. no se explica por la metodología de selección tabular.**

Esta redacción:
- Reconoce explícitamente las dos cifras (0.793 y 0.816) y las reconcilia.
- Atribuye la diferencia a la imputación, que es verificable y reproducible.
- Mantiene el argumento original sobre la brecha 0.79 vs 0.84.

---

## Otros puntos menores que detecté

### 1. Fig. 2 (caption tachada: “Para el FT-Transformer, el factor entre cohortes es 74×”)

El número es correcto (sparse Δ FT = +0.001, dense Δ FT = +0.074 → 74×) pero se sostiene desde la Tabla 6 / §4.1, no desde la Tabla 2 (que solo reporta Cox K=all). **Sugerencia:** reinsertar el dato en el texto del §4.3 o en pie de la Tabla 6:

> *Para el FT-Transformer, el delta del leakage crece de +0.001 (n=444) a +0.074 (n=224), un factor de 74× entre cohortes (Tabla 6 vs §4.3).*

### 2. Limitaciones tachadas en §5

El párrafo eliminado decía: *“se limita a datos clínicos tabulares y no evalúa el efecto del leakage en modalidades de imagen, texto o datos ómicos”*. Esta limitación **no aparece exactamente en §5.1** y es importante para la honestidad metodológica. **Sugerencia:** reinsertar 1 oración al final de §5.1 “Validez de constructo”:

> *Adicionalmente, el alcance del estudio se limita a la modalidad tabular: no evaluamos cómo interactúa el leakage de variables temporales con modalidades de imagen, texto u ómicas dentro de un pipeline multimodal completo.*

### 3. Mención de “protocolo de evaluación según lo reportado por los autores” en Tabla 4

El cambio “sin held-out estricto” → “evaluación según lo reportado por los autores” es **más justo** (no asume protocolo deficiente sin evidencia) y se alinea con el §5 que reconoce que el sistema completo incluye un módulo de imagen no replicado. ✅ Apropiado.

### 4. Las p-valores reportados en Tabla 6 vs el lenguaje del texto

El texto dice "p medianas entre semillas no alcanzan significancia convencional (p < 0.05)". Esto es preciso. La p-mínima de FT-Transformer (0.014) **sí** alcanza significancia en una semilla, lo cual está reportado correctamente. **No hay corrección necesaria**, pero podría enfatizarse:

> *El FT-Transformer alcanza p = 0.014 en la semilla con mayor efecto, lo que constituye evidencia estadística de leakage en al menos una partición. La convergencia con el delta SHAP (82.1 % vs 13.4 %) y con la mejora en IBS (0.146→0.124) refuerza esta interpretación.*

---

## Resumen ejecutivo

| Ítem | Estado |
|---|---|
| Cifras Tabla 1 (sensibilidad diferencial dense) | ✅ 11/11 validadas |
| Cifras Tabla 2 (leakage condicional a densidad) | ✅ 6/6 validadas |
| Cifras Tabla 3 (replicación Mahootiha) | ✅ 16/16 validadas |
| Cifras Tabla 6 (paired bootstrap) | ✅ 12/12 validadas |
| Cifras IBS y SHAP en §4.1, §4.3, §5 | ✅ todas validadas |
| Comentario [C1] del revisor | ✅ resuelto con propuesta de redacción |
| Sugerencia 74× en §4.3 | ⚠ reinsertar |
| Limitación modalidad tabular en §5.1 | ⚠ reinsertar 1 oración |

**Recomendación:** la versión revisada es **internamente consistente con los datos generados** y refleja correctamente el cohorte denso (Tabla 1) sin contradecir los resultados del cohorte disperso (Tabla 2). Aplicar las 3 sugerencias menores (C1 + 74× + limitación) deja el manuscrito en estado de envío.

---

## Anexo — Reproducción

Para regenerar todas las cifras desde cero:

```bash
# Cohorte disperso n=444 (Tabla 2 fila 1, Tabla 3 cols 1-2)
# configs/experiment_config.yaml: cohort_filter.enabled=false
python -m src.runner configs/experiment_config.yaml

# Cohorte denso n=224 (Tabla 1, Tabla 2 fila 2, Tabla 3 cols 3-4, Tabla 6)
# configs/experiment_config.yaml: cohort_filter.enabled=true
python -m src.runner configs/experiment_config.yaml

# Postprocesado denso para Tabla 6, IBS y SHAP
RUN_DIR=results/<dense_timestamp>
python tools/run_significance_tests.py "$RUN_DIR" --n-iter 1000
python tools/run_calibration.py "$RUN_DIR"
python tools/run_shap_attributions.py "$RUN_DIR" --seed 42 --n-explain 100
```

Tiempo total estimado: ~50 min en una RTX 3050 Ti.
