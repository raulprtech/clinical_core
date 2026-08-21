# Resultado preliminar: ResNet18 2.5D + attention/Mamba

Fecha de ejecución: 2026-08-21.

## Resultado

Se descargaron y procesaron las 214 series de imagen disponibles para la
cohorte TCGA-KIRC usada por VISION-L0.6. La cohorte común contiene 214
pacientes y 64 eventos. Cada hold-out pareado contiene 43 pacientes y 13
eventos.

| Modelo | C-index medio | DE entre semillas |
|---|---:|---:|
| ResNet18 2.5D + PCA train-only + Cox ridge oficial | 0.7170 | 0.0987 |
| ResNet18 2.5D axial + attention pooling | 0.7252 | 0.0539 |
| ResNet18 2.5D axial + Mamba | 0.7563 | 0.0127 |

El baseline oficial reproduce el resultado histórico de aproximadamente
0.714. Mamba mejora el promedio en 0.0392, gana cuatro de cinco semillas y
muestra menor dispersión.

| Semilla | PCA+Cox oficial | Attention | Mamba | Mamba - oficial |
|---:|---:|---:|---:|---:|
| 42 | 0.8717 | 0.7664 | 0.7467 | -0.1250 |
| 123 | 0.7538 | 0.7538 | 0.7748 | +0.0210 |
| 456 | 0.6221 | 0.7391 | 0.7458 | +0.1237 |
| 789 | 0.6637 | 0.6313 | 0.7640 | +0.1003 |
| 1024 | 0.6739 | 0.7355 | 0.7500 | +0.0761 |

## Incertidumbre

Los intervalos bootstrap pareados de Mamba menos PCA+Cox oficial fueron:

| Semilla | Delta | IC 95% | p bootstrap |
|---:|---:|---:|---:|
| 42 | -0.1250 | [-0.2823, 0.0293] | 0.1152 |
| 123 | +0.0210 | [-0.1125, 0.2000] | 0.7740 |
| 456 | +0.1237 | [-0.1071, 0.3473] | 0.2952 |
| 789 | +0.1003 | [-0.0458, 0.2397] | 0.1920 |
| 1024 | +0.0761 | [-0.0569, 0.2188] | 0.2532 |

Ningún intervalo individual excluye cero. Los cinco hold-outs se solapan y no
constituyen cinco réplicas independientes. Por eso el promedio superior y la
menor dispersión son una señal para continuar, no una demostración de
superioridad.

## Auditoría técnica

- 214 de 214 descargas completas: 190 CT y 24 MR.
- 36,282 archivos de imagen, aproximadamente 17 GB.
- 214 cachés secuenciales, cero fallos.
- 13,526 tokens ResNet18; entre 32 y 64 por paciente.
- Norma de token tras serialización float16: 0.99990 a 1.00008.
- Attention y Mamba reciben exactamente los mismos tokens, posiciones y
  máscaras.
- No hay IDs duplicados, riesgos no finitos ni uso del held-out para selección.
- ResNet18 permanece congelada y la extracción no recibe outcomes.

## Asimetrías del protocolo

El baseline oficial selecciona PCA y penalización mediante tres folds internos
y después reajusta PCA+Cox con todo el outer-train. Attention y Mamba reservan
20% del outer-train para early stopping y no se reajustan después con esa
validación. Por tanto, la comparación conserva el held-out y evita leakage,
pero no iguala por completo el uso de muestras de entrenamiento.

También cambia la cobertura de entrada: el baseline resume tres vistas
centrales, mientras los modelos secuenciales recorren el stack axial. El
contraste Mamba frente a attention aísla mejor el valor del modelado
longitudinal que el contraste de cualquiera de ellos frente al baseline.

## Conclusión permitida

La formulación ResNet18 2.5D + Mamba superó el umbral histórico en este Fast
Proof y fue notablemente estable entre las cinco particiones. El resultado
justifica una evaluación confirmatoria con nested repeated cross-validation,
reajuste outer-train simétrico y, preferentemente, una cohorte externa.

No permite todavía afirmar superioridad estadística o generalización externa.

## Contexto histórico del módulo de visión

Este Fast Proof se interpreta dentro de una secuencia de experimentos que no
son todos directamente comparables:

| Experimento | Cohorte/protocolo | Resultado principal | Lectura vigente |
|---|---|---|---|
| ResNet18 2D V4.1 | tres vistas centrales, backbone congelado | baseline de integración | referencia arquitectónica, no secuencial |
| ResNet50 2D V5 | tres vistas centrales y proyección fija a 768D | baseline de mayor capacidad | no aisló una ventaja suficiente para desplazar V6 |
| ResNet18 2.5D V6 | 214 pacientes; PCA+Cox train-only | histórico ~0.714; reproducción actual 0.7170 | comparador oficial del Fast Proof |
| STU-Net FP32 frozen | piloto técnico de 10 casos | 10/10 completos; coseno de repetición 0.99999975 | factible, pero requiere intersección QC por geometría |
| STU-Net vs ResNet18 2.5D | 50 pacientes, 13 eventos | 0.6318 vs 0.4796 | señal exploratoria; no comparable con los 214 del baseline oficial |
| STU-Net TurboConv W4A8 | 20 calibración + 50 evaluación | mejor que PTQ, pero no pasó las compuertas de retención | W4A8 rechazado; W6A8 quedó como siguiente candidato |
| ResNet18 axial + Mamba | 214 pacientes, cinco hold-outs | 0.7563 ± 0.0127 | señal más estable; requiere confirmación simétrica |

El resultado STU-Net de 50 pacientes no contradice el baseline ResNet18 de
214: cambian cohorte, número de eventos y protocolo. De igual forma, el piloto
TurboConv evaluó fidelidad de cuantización, no beneficio real de tiempo o
memoria, porque utilizó kernels de fake quantization en punto flotante.

## Relación con la historia de fusión multimodal

Mamba modifica la representación y el pronóstico de `VISION-IN`; no cambia por
sí mismo `FUSION-PROC`. La evolución documentada de la fusión fue:

1. La concatenación tardía sustituyó diagnósticamente al VAE Stage A para
   comprobar si el cuello de botella diluía la señal de las modalidades.
2. La concatenación cruda de 2,307 variables sobre aproximadamente 168
   pacientes de entrenamiento por fold mostró un régimen extremo `p >> n` y
   además forzaba modalidades débiles en todos los splits.
3. La reparación pareada y libre de leakage pasó a riesgos cross-fitted y
   fusión convexa train-only. Con ResNet18 obtuvo 0.8111, frente a 0.7516 de la
   concatenación proyectada; asignó peso cero a texto y 0.1--0.3 a visión.

Por tanto, la concatenación fue una intervención diagnóstica importante, no la
recomendación final. La siguiente integración multimodal de Mamba debe aportar
riesgos o representaciones generados dentro de cada split al mismo evaluador
pareado de fusión convexa; una concatenación global volvería a introducir el
problema de dimensionalidad y podría contaminar la evaluación.

## Artefactos

Los artefactos están en:

    results_vision/resnet18_attention_mamba_fastproof/

Los CSV con IDs, outcomes, riesgos o particiones por paciente se conservan
sólo en el entorno local y están excluidos de Git. El repositorio publica
únicamente métricas agregadas y procedencia no identificable.

Archivos clave:

- official_comparison_summary.json
- official_pca_cox_metrics.csv
- official_pca_cox_predictions.csv
- mamba_vs_official_paired_bootstrap.csv
- per_seed_metrics.csv
- heldout_predictions.csv
- splits.csv
- provenance.json

El historial consolidado de decisiones, comparabilidad y próximos pasos se
mantiene en `docs/research_decision_log.md`.
