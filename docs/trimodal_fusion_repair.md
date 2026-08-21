# Reparación de la fusión trimodal

## Aclaración histórica: VAE, concatenación y fusión convexa

El cambio fechado el 2026-05-28 desde VAE Stage A a concatenación tardía fue
una prueba diagnóstica: buscaba separar una posible dilución de señal en el VAE
de la hipótesis de que texto o visión no aportaban información incremental.
Los YAML `experiment_config_late_fusion_text_n444.yaml` y
`experiment_config_late_fusion_turbolatent_n444.yaml` conservan esa decisión.

La concatenación no quedó como solución final. Los experimentos pareados que
siguen en este documento mostraron su desventaja en el régimen `p >> n` y
motivaron la fusión de riesgos convexa, que puede descartar una modalidad. La
secuencia ResNet18 + Mamba de 2026-08-21 es un nuevo candidato de visión, no un
cambio automático del fusionador. Debe evaluarse dentro de cada split con este
mismo protocolo para evitar leakage.

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

## Cuarta referencia: alineación ortogonal y fusión jerárquica

Se evaluaron cuatro variantes adicionales sobre los mismos 210 pacientes,
hold-outs y cinco semillas del diagnóstico ResNet18 2.5D:

- concatenación después de una rotación Procrustes de visión;
- fusión jerárquica con tabular como ancla y residuos gated de texto y visión;
- fusión jerárquica híbrida que concatena tabular-texto y añade visión mediante gate;
- la combinación de rotación Procrustes y fusión jerárquica.

La rotación se ajustó solo con el outer-train. Su objetivo fue alinear visión
con un contexto construido mediante PCA tabular y texto, también ajustados
solo con train. La selección de época se realizó en una partición interna y
el modelo se reinicializó y reajustó con todo el outer-train.

| Fusión | C-index medio | SD entre semillas | Diferencia vs convexa | Victorias/empates/derrotas |
|---|---:|---:|---:|---:|
| Convexa train-only | **0.8111** | 0.0820 | — | — |
| Jerárquica residual | 0.8066 | **0.0714** | −0.0045 | 3/0/2 |
| Jerárquica concat + gate visual | 0.8005 | 0.0910 | −0.0107 | 1/0/4 |
| Ortogonal + jerárquica | 0.7972 | 0.0726 | −0.0139 | 1/2/2 |
| Atención cruzada | 0.7915 | 0.0868 | −0.0196 | 2/0/3 |
| Concatenación proyectada | 0.7516 | 0.0786 | −0.0595 | 0/0/5 |
| Ortogonal + concatenación | 0.7483 | 0.0772 | −0.0628 | 1/0/4 |

La transformación sí consiguió alineación geométrica: el coseno medio entre
visión y contexto pasó de −0.035 a 0.114 en held-out y el error de
ortogonalidad fue aproximadamente 5.8e−15. Sin embargo, esa alineación no se
tradujo en mejor discriminación de supervivencia. Esto indica que aproximar
los espacios no añade señal pronóstica y puede eliminar complementariedad.

La jerárquica residual es la única variante nueva competitiva. Superó a la
convexa en tres de cinco semillas y redujo la dispersión, pero su media fue
0.0045 menor. Por tanto, no reemplaza todavía a la convexa; queda como
comparador secundario que merece repetirse con STU-Net.

La híbrida concat + gate obtuvo 0.8005 ± 0.0910. Mejoró con claridad en una
semilla, pero perdió contra la convexa en las otras cuatro y fue menos estable
que la jerárquica residual. La concatenación del contexto añade unos 4.6 k
parámetros y no justificó escalar a una doble atención jerárquica con esta
cohorte.

Artefactos:

- `code/tools/evaluate_hierarchical_orthogonal_fusion.py`
- `results_fusion/hierarchical_orthogonal_resnet_diagnostic_210/`

Las decisiones históricas y los enlaces cruzados con los experimentos de
visión se registran también en `docs/research_decision_log.md`.

## Quinta referencia: riesgo Mamba secuencial

Se integró Mamba mediante riesgos cross-fitted alineados con los mismos 210
pacientes y cinco outer splits. La visión Mamba mejoró a ResNet18 en las cinco
seeds (0.7265 frente a 0.6397), pero la fusión convexa sólo subió de 0.8111 a
0.8180: dos victorias, un empate y dos derrotas.

Ningún intervalo bootstrap individual de fusión Mamba menos fusión ResNet
excluyó cero. Por tanto, Mamba queda como representación visual preferida y
candidata de fusión, mientras 0.8111 permanece como referencia formal. El
detalle está en docs/trimodal_mamba_fusion_results.md.

## Sexta referencia: outer repeated CV de fusión

La ventaja diagnóstica Mamba no quedó confirmada al pasar a 5 outer folds por
3 repeticiones. Los C-index OOF fueron 0.7892 tabular, 0.7841 fusión ResNet y
0.7866 fusión Mamba. Mamba menos ResNet fue +0.0025 con IC95%
[-0.0117, +0.0169] y p=0.7328.

La visión Mamba mantuvo una señal media mayor que ResNet (+0.0447), pero su
intervalo también cruzó cero. La decisión vigente es no reemplazar el baseline
de fusión ni aumentar su complejidad. Véase
docs/trimodal_sequence_nested_cv_results.md.
