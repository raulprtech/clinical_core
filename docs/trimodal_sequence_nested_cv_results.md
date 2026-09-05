# Confirmación interna de fusión: outer repeated CV

Fecha de ejecución: 2026-08-21.

## Pregunta

¿La ventaja diagnóstica de la fusión Mamba (0.8180 frente a 0.8111) persiste
cuando cada paciente recibe predicciones OOF en outer CV repetida y los pesos
se vuelven a seleccionar dentro de cada outer-train?

## Protocolo

- 210 pacientes y 64 eventos.
- Cinco outer folds estratificados, repetidos tres veces.
- Tres folds internos para tabular, texto y ResNet.
- Riesgos Mamba del outer-train generados mediante cross-fitting anidado.
- Riesgo Mamba held-out generado por un modelo separado, seleccionado dentro
  del outer-train y reajustado con todo ese train.
- Pesos convexos seleccionados sólo con riesgos cross-fitted.
- Una predicción OOF por paciente y repetición.
- 5,000 bootstraps agrupados: el mismo remuestreo de pacientes se reutiliza en
  las tres repeticiones.

La auditoría confirmó 210 predicciones únicas y 64 eventos por repetición,
riesgos finitos y pesos que suman uno en los 15 folds.

## Resultado OOF

| Modelo | C-index medio | DE entre repeticiones | Rep. 1 | Rep. 2 | Rep. 3 |
|---|---:|---:|---:|---:|---:|
| Tabular | **0.7892** | 0.0097 | 0.7910 | 0.7788 | 0.7978 |
| Visión ResNet18 | 0.6327 | 0.0208 | 0.6200 | 0.6214 | 0.6566 |
| Visión Mamba | 0.6774 | 0.0213 | 0.6891 | 0.6903 | 0.6529 |
| Fusión ResNet | 0.7841 | 0.0150 | 0.7996 | 0.7697 | 0.7829 |
| Fusión Mamba | 0.7866 | 0.0107 | 0.7960 | 0.7750 | 0.7888 |

| Comparación | Delta medio OOF | IC 95% agrupado | p bootstrap |
|---|---:|---:|---:|
| Visión Mamba - visión ResNet | +0.0447 | [-0.0076, +0.0995] | 0.0988 |
| Fusión Mamba - fusión ResNet | +0.0025 | [-0.0117, +0.0169] | 0.7328 |
| Fusión Mamba - tabular | -0.0026 | [-0.0175, +0.0110] | 0.7372 |

Por outer fold, Mamba visual ganó 11 y perdió 4. La fusión Mamba ganó 9 y
perdió 6 frente a la fusión ResNet; frente a tabular ganó 5 y perdió 10.

## Estabilidad de pesos

| Fusionador | Peso tabular | Peso texto | Peso visión | Visión en cero |
|---|---:|---:|---:|---:|
| ResNet | 0.700 ± 0.120 | 0.067 ± 0.062 | 0.233 ± 0.098 | 0/15 |
| Mamba | 0.747 ± 0.092 | 0.080 ± 0.041 | 0.173 ± 0.088 | 1/15 |

El fusionador Mamba asigna menos peso promedio a visión que el ResNet, aunque
su C-index visual es mayor. Esto indica que la señal secuencial no es
necesariamente complementaria a tabular y que la selección por C-index interno
es variable. Texto recibió peso 0.1 en la mayoría de folds pese a no mostrar
aporte estable en diagnósticos anteriores.

## Interpretación

La mejora visual Mamba es prometedora, pero su intervalo agrupado todavía cruza
cero debido a variabilidad entre folds y una tercera repetición sin ventaja.
No existe evidencia de mejora de la fusión Mamba sobre ResNet ni sobre tabular.

El resultado diagnóstico 0.8180 no se reproduce como ganancia confirmada. Bajo
OOF completo, tabular, fusión ResNet y fusión Mamba son estadísticamente
compatibles. Por tanto:

- Mamba continúa como candidato visual, no como baseline confirmado.
- La fusión convexa no supera de manera estable al tabular.
- No se justifica añadir más complejidad al fusionador en esta cohorte.
- La siguiente mejora local debe enfocarse en robustez de la representación y
  calibración/selección de pesos, no en atención fusionadora adicional.

## Diagnóstico del fold extremo

El fold con C-index Mamba 0.3611 fue investigado sin excluirlo. Su composición
CT/MR, longitud de tokens y geometría no explica por sí sola la caída. Diez
inicializaciones a 19 épocas permanecieron bajas (0.4170 ± 0.0367), mientras
3--5 épocas con la seed original mejoraron a 0.5451--0.5313.

Un cap post hoc de 5 épocas mejoró el promedio por fold en +0.0173, pero perdió
en 6 de 15 folds. No se adopta por haber sido elegido después de ver los
resultados. El detalle está en docs/mamba_epoch_stability_diagnostic.md.

## Límites

- Es validación interna repetida sobre una cohorte ya explorada.
- La tercera repetición contiene un fold con degradación Mamba marcada, que
  amplía la incertidumbre pero no debe eliminarse post hoc.
- La rejilla de pesos avanza en pasos de 0.1.
- Falta validación externa y la comparación STU-Net sobre la misma cohorte QC.

## Artefactos

Se versionan únicamente:

- per_fold_metrics.csv
- per_repeat_metrics.csv
- paired_cluster_bootstrap.csv
- summary.json
- provenance.json

Predicciones, splits y cohortes a nivel paciente permanecen locales.
