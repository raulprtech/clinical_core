# Resultado diagnóstico: Mamba en fusión trimodal convexa

Fecha de ejecución: 2026-08-21.

## Pregunta

¿La mejora de la representación visual secuencial aporta señal incremental al
fusionador trimodal, manteniendo los mismos 210 pacientes y los cinco outer
hold-outs del diagnóstico ResNet18?

## Control de leakage

Para cada seed:

- tabular, texto y ResNet reproducen el evaluador histórico train-only;
- el riesgo Mamba del outer-train se genera mediante tres folds cross-fitted;
- cada modelo cross-fit selecciona su época dentro de su propio train, sin usar
  el fold que predice;
- el riesgo Mamba held-out proviene de otro modelo cuya época se selecciona
  dentro del outer-train y que después se reajusta con todo ese outer-train;
- los riesgos se convierten a percentiles usando sólo la distribución de train;
- los pesos convexos se seleccionan exclusivamente con riesgos cross-fitted del
  outer-train.

El baseline se reprodujo exactamente: tabular 0.7990, visión ResNet 0.6397 y
fusión ResNet 0.8111.

## Resultado

| Modelo | C-index medio | DE entre seeds |
|---|---:|---:|
| Tabular | 0.7990 | 0.0989 |
| Visión ResNet18 2.5D | 0.6397 | 0.0447 |
| Visión Mamba secuencial | **0.7265** | 0.0658 |
| Fusión convexa con ResNet | 0.8111 | 0.0820 |
| Fusión convexa con Mamba | **0.8180** | 0.0913 |

Mamba visual superó a ResNet visual en las cinco seeds, con una diferencia
media de +0.0868. La fusión Mamba mejoró a la fusión ResNet en dos seeds,
empató una y perdió dos: delta medio +0.0069 ± 0.0204. Frente a tabular, la
fusión Mamba obtuvo +0.0190, con cuatro victorias y un empate.

| Seed | Visión ResNet | Visión Mamba | Fusión ResNet | Fusión Mamba | Delta de fusión |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.6811 | 0.7591 | 0.8571 | 0.8571 | 0.0000 |
| 123 | 0.6677 | 0.7710 | 0.8144 | 0.8308 | +0.0165 |
| 456 | 0.6284 | 0.7123 | 0.8767 | 0.8699 | -0.0068 |
| 789 | 0.6536 | 0.7727 | 0.8370 | 0.8746 | +0.0376 |
| 1024 | 0.5678 | 0.6172 | 0.6703 | 0.6575 | -0.0128 |

## Pesos seleccionados con Mamba

| Seed | Tabular | Texto | Visión |
|---:|---:|---:|---:|
| 42 | 0.7 | 0.0 | 0.3 |
| 123 | 0.8 | 0.0 | 0.2 |
| 456 | 0.7 | 0.1 | 0.2 |
| 789 | 0.8 | 0.0 | 0.2 |
| 1024 | 0.8 | 0.0 | 0.2 |

La señal visual más fuerte no desplaza al tabular: Mamba recibe 0.2--0.3. Texto
sigue en cero salvo en la seed 456, donde su peso 0.1 coincide con una pequeña
caída de la fusión frente a ResNet.

## Incertidumbre

Se ejecutaron 5,000 bootstraps pareados por seed. Sólo uno de cinco intervalos
individuales excluyó cero para Mamba visual frente a ResNet visual; ninguno lo
hizo para fusión Mamba frente a fusión ResNet; uno lo hizo para fusión Mamba
frente a tabular.

Los hold-outs se solapan y no son réplicas independientes. Por ello no se
interpreta el promedio de las cinco seeds como una prueba confirmatoria global.

## Decisión

- Conservar Mamba como la representación visual local preferida: mejora la
  modalidad visual en 5/5 seeds y ya tiene soporte confirmatorio unimodal.
- Mantener la fusión convexa ResNet de 0.8111 como referencia formal por ahora.
- Registrar la fusión Mamba de 0.8180 como candidata diagnóstica, no como nuevo
  baseline: su ventaja sobre la fusión ResNet es pequeña e inestable.
- Mantener texto bajo revisión; aumentar complejidad del fusionador no corrige
  una modalidad sin aporte incremental estable.
- La siguiente prueba sin Colab es evaluar estabilidad de pesos con outer
  repeated CV alineada. STU-Net y validación externa quedan pendientes.

## Artefactos

Se versionan sólo artefactos agregados:

- per_seed_metrics.csv
- paired_bootstrap.csv
- summary.json
- provenance.json

Los archivos con IDs, outcomes, riesgos o splits por paciente permanecen
locales y excluidos de Git.
