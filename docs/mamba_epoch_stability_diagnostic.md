# Diagnóstico de estabilidad de épocas Mamba

Fecha: 2026-08-21.

## Motivación

En la repeated CV trimodal, el fold repetición 2/fold 4 produjo C-index visual
Mamba 0.3611 frente a 0.5885 de ResNet. El fold se conservó en el resultado
principal. Este análisis busca explicar la degradación sin modificar post hoc
la estimación confirmatoria.

## Composición del fold

El held-out contiene 42 pacientes y 12 eventos:

- 39 CT y 3 MR;
- 63.6 tokens en promedio, rango 48--64;
- 25 series con geometría marcada como fail y 14 como pass;
- mediana de slice-gap ratio aproximadamente 1.000002.

La composición no es extrema frente a los otros 14 folds. La longitud de
secuencia está en el percentil 80 y la proporción MR es baja, pero otros folds
con composición similar no fallaron. No se identifica un covariate shift
simple por modalidad, tokens o geometría.

## Sensibilidad a inicialización

Manteniendo 19 épocas y el mismo outer-train, diez inicializaciones obtuvieron:

- media 0.4170;
- DE 0.0367;
- mínimo 0.3611;
- máximo 0.4792.

La inicialización original fue la peor, pero ninguna seed recuperó un C-index
competitivo. Por tanto, el fallo no se explica principalmente por una única
inicialización desafortunada.

## Sensibilidad a épocas en el outlier

Con la misma inicialización:

| Épocas | C-index |
|---:|---:|
| 3 | 0.5451 |
| 5 | 0.5313 |
| 10 | 0.4688 |
| 19 seleccionadas | 0.3611 |
| 30 | 0.4063 |
| 50 | 0.4028 |

El fold muestra sobreentrenamiento: la época seleccionada por inner CV no
generaliza al outer held-out. Sin embargo, las épocas seleccionadas no tienen
una asociación global significativa con C-index en sólo 15 folds.

## Caps conservadores en los 15 folds

Se repitió el refit visual con la misma inicialización y
min(épocas_seleccionadas, cap):

| Cap | C-index medio por fold | Delta vs selección original | G/E/P | Folds modificados |
|---:|---:|---:|---:|---:|
| 5 | 0.6966 | +0.0173 | 7/2/6 | 11 |
| 10 | 0.6843 | +0.0050 | 5/2/8 | 8 |
| 15 | 0.6748 | -0.0045 | 2/3/10 | 3 |
| Selección original | 0.6793 | — | — | — |

Cap 5 mejora el promedio y reduce la dispersión, pero pierde en seis folds.
Como los caps se propusieron después de observar el outlier y se evaluaron en
la misma cohorte, no pueden reemplazar el protocolo confirmado.

## Decisión

- No eliminar ni corregir el fold outlier en el informe principal.
- No cambiar automáticamente la selección mediana por cap 5.
- Registrar cap 5 como hiperparámetro predeclarado para una validación futura o
  cohorte externa.
- Priorizar regularización/estabilidad de entrenamiento antes de aumentar
  capacidad.
- Mantener la conclusión de fusión: Mamba no demostró ventaja sobre ResNet o
  tabular.

## Artefactos

- results_fusion/trimodal_sequence_outlier_diagnostic/
- results_fusion/mamba_epoch_cap_sensitivity/

Todos los artefactos publicados son agregados y no contienen IDs de pacientes.
