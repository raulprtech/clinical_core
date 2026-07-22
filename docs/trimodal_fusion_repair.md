# Reparación de la fusión trimodal

## Diagnóstico

La caída del C-index de la primera Phase 5 no demuestra que la información
multimodal sea perjudicial. Había tres problemas en la evaluación:

1. Las ablaciones usaban cohortes distintas: tabular se evaluó sobre 444
   pacientes y la combinación trimodal sobre 210.
2. La concatenación directa producía 2,307 entradas para aproximadamente 168
   pacientes de entrenamiento por fold. La cabeza lineal quedaba en un régimen
   extremo de más variables que observaciones.
3. La imputación y el escalado tabular se ajustaban sobre la cohorte completa
   antes de crear los folds, en lugar de ajustarse exclusivamente con train.

La concatenación también obligaba a que texto y visión entrasen en la cabeza,
aunque una de esas modalidades no añadiera señal en un split concreto.

## Estrategia reparada

`code/tools/evaluate_trimodal_fusion.py` implementa una comparación pareada:

- intersección exacta de pacientes con las tres modalidades;
- cinco hold-outs 80/20 estratificados por evento;
- imputación, escalado, PCA y Cox ajustados solo con el 80 % de entrenamiento;
- selección interna de dimensión PCA y penalización Cox mediante tres folds;
- riesgos cross-fitted de tabular, texto y visión;
- conversión de cada riesgo a percentil empírico usando únicamente la
  distribución de riesgo de entrenamiento;
- fusión tardía convexa con pesos no negativos que suman uno;
- selección de pesos usando solo predicciones cross-fitted del outer-train;
- evaluación única y ciega sobre el 20 % held-out.

La rejilla de pesos incluye `(1,0,0)`, `(0,1,0)` y `(0,0,1)`. Por tanto, el
modelo puede descartar una modalidad que no aporte señal durante el ajuste, en
vez de forzarla dentro de una concatenación de alta dimensión.

## Resultado diagnóstico con ResNet18 2.5D local

Este resultado valida el método de fusión, pero aún no usa el nuevo export de
160 pacientes de STU-Net obtenido en Colab.

| Modelo | C-index medio | SD entre semillas |
|---|---:|---:|
| Tabular | 0.7990 | 0.0989 |
| Texto | 0.6358 | 0.0587 |
| ResNet18 2.5D | 0.6397 | 0.0447 |
| Fusión tardía igualitaria | 0.7758 | 0.0593 |
| **Fusión tardía convexa train-only** | **0.8111** | **0.0820** |

La media igualitaria perdió 0.0232 frente al tabular. La fusión convexa ganó
0.0121, con tres victorias, un empate y una derrota en cinco semillas.

Los pesos seleccionados fueron:

| Semilla | Tabular | Texto | Visión |
|---:|---:|---:|---:|
| 42 | 0.9 | 0.0 | 0.1 |
| 123 | 0.7 | 0.0 | 0.3 |
| 456 | 0.7 | 0.0 | 0.3 |
| 789 | 0.7 | 0.0 | 0.3 |
| 1024 | 0.7 | 0.0 | 0.3 |

La selección sistemática de peso cero para texto indica que el embedding de
texto actual no añade señal incremental una vez incluida la información
tabular. Esto debe investigarse por separado; aumentar la complejidad del
fusionador no solucionará un embedding textual redundante o débil.

## Siguiente corrida con STU-Net

El evaluador acepta un CSV por paciente con `case_id` y `z000...z767`. También
acepta el formato crudo del notebook con columnas numéricas `f*`. Cuando esté
disponible el export de los 160 pacientes comunes:

```bash
.venv/bin/python -u code/tools/evaluate_trimodal_fusion.py \
  --features results_trimodal/20260721_012302_d9a7599d/raw_features.csv \
  --targets results_trimodal/20260721_012302_d9a7599d/raw_targets.csv \
  --text-embeddings data/embeddings/text_embeddings_TCGA-KIRC_20260528.npz \
  --vision-embeddings /ruta/stunet_160_embeddings.csv \
  --vision-label stunet_s_fp32_frozen \
  --output-dir results_fusion/trimodal_risk_fusion_stunet_160
```

Los artefactos del diagnóstico actual están en
`results_fusion/trimodal_risk_fusion_resnet_diagnostic_210/`.

## Tercera referencia: atención cruzada

Se añadió una comparación controlada entre tres estrategias. La concatenación
y la atención cruzada comparten proyectores de modalidad de 32 dimensiones y
la misma cabeza Cox lineal. La atención es la única diferencia arquitectónica:
opera sobre tres tokens —tabular, texto y visión— mediante cuatro cabezas, una
capa feed-forward residual y pooling aprendido por paciente.

Para contener el sobreajuste, texto y visión usan PCA fija de 16 dimensiones
ajustada solo con train. El número de épocas se selecciona en una validación
interna y después el modelo se reinicializa y ajusta sobre el 80 % completo
durante ese número de épocas. El 20 % held-out permanece ciego.

Resultado diagnóstico sobre los mismos 210 pacientes y splits:

| Fusión | C-index medio | SD entre semillas | Parámetros de fusión aprox. |
|---|---:|---:|---:|
| Concatenación proyectada | 0.7516 | 0.0786 | 2.4 k |
| Atención cruzada | 0.7915 | 0.0868 | 10.9 k |
| **Convexa train-only** | **0.8111** | **0.0820** | 3 pesos globales |

La atención cruzada superó a la convexa en dos semillas y perdió en tres; su
diferencia media fue −0.0196. Sí aportó una mejora clara de +0.0399 frente a la
concatenación. Por ahora, la convexa permanece como referencia principal y la
atención cruzada como comparador no lineal.

El pooling de atención asignó en promedio aproximadamente 37–45 % a tabular,
25–39 % a texto y 23–34 % a visión según la semilla. Estos valores describen
la asignación interna del modelo, pero no deben interpretarse causalmente. El
hecho de que mantenga peso apreciable en texto, mientras la convexa lo descarta,
es una posible explicación de la diferencia y deberá reevaluarse con STU-Net.

Artefactos:

- `code/components/processors/fusion/models/cross_attention.py`
- `code/tools/evaluate_cross_attention_fusion.py`
- `results_fusion/three_fusion_resnet_diagnostic_210/`
