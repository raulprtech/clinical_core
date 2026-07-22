# Comparación preliminar de supervivencia: STU-Net-S frente a ResNet18 2.5D

## Resultado

Sobre la intersección exacta de 50 pacientes con 13 eventos, STU-Net-S FP32
congelada obtuvo un C-index medio de **0.6318** y ResNet18 2.5D ImageNet
congelada obtuvo **0.4796**. La diferencia media pareada fue **+0.1521** a
favor de STU-Net.

La dispersión entre los cinco hold-outs fue grande: 0.2030 para STU-Net y
0.1779 para ResNet. Este resultado es exploratorio y no demuestra superioridad
estadística, porque cada conjunto held-out contiene solo 10 pacientes y 3
eventos. Los intervalos bootstrap pareados por semilla incluyen cero.

| Semilla | STU-Net | ResNet18 2.5D | Diferencia STU-Net − ResNet |
|---:|---:|---:|---:|
| 42 | 0.7273 | 0.7273 | 0.0000 |
| 123 | 0.4444 | 0.5556 | −0.1111 |
| 456 | 0.3846 | 0.3077 | +0.0769 |
| 789 | 0.7692 | 0.3077 | +0.4615 |
| 1024 | 0.8333 | 0.5000 | +0.3333 |
| **Media** | **0.6318** | **0.4796** | **+0.1521** |

STU-Net ganó tres semillas, empató una y perdió una. Su C-index mediano fue
0.7273 frente a 0.5000 para ResNet.

## Protocolo pareado

- Mismos 50 identificadores y mismos desenlaces de supervivencia global.
- Cinco semillas: 42, 123, 456, 789 y 1024.
- Hold-out 80/20 estratificado por evento para cada semilla.
- Dentro del 80 % de entrenamiento: 32 pacientes para optimización y 8 para
  selección por parada temprana; el held-out no participa en el ajuste.
- Misma cabeza `PrognosisProc_LinearCox`, arquitectura 768→1, tasa de aprendizaje
  1e-3, weight decay 1e-3, máximo 200 épocas y paciencia 20.
- Mismos pesos iniciales de la cabeza para ambos encoders dentro de cada semilla.
- Embeddings L2 normalizados, sin ajuste basado en outcomes.
- Predicciones pareadas sobre exactamente los mismos pacientes held-out.
- 5,000 remuestreos bootstrap estratificados por semilla para intervalos
  exploratorios del C-index y su diferencia.

## Qué representa esta comparación

Los dos encoders están congelados. STU-Net usa el checkpoint STU-Net-S FP32 y
pooling reproducible del bottleneck dentro de la ROI renal. El ResNet18 2.5D
integrado localmente usa un backbone ImageNet congelado, tres vistas centrales
y contexto de cortes vecinos.

Por tanto, esta comparación controla la cohorte, los splits y la cabeza, y sirve
para comparar las representaciones congeladas disponibles. **No equivale** a la
comparación final contra un ResNet18 2.5D entrenado dentro de cada split según el
protocolo original del notebook. Si se desea comparar contra ese modelo
entrenado, será necesario materializar embeddings held-out producidos por un
checkpoint específico de cada semilla, sin exposición del held-out.

## Limitaciones

1. Solo hay 13 eventos y tres eventos por hold-out; pequeños cambios en el
   orden de riesgo producen saltos grandes del C-index.
2. Los cinco hold-outs se solapan. La desviación entre semillas mide
   sensibilidad al split, no constituye un intervalo de confianza con cinco
   muestras independientes.
3. La cabeza tiene 768 entradas y pocos pacientes. Se mantuvo así para respetar
   el protocolo común, pero la comparación final necesita una cohorte mayor.
4. No debe compararse el 0.4796 de este subconjunto con el 0.6319 previo de
   ResNet calculado sobre 214 pacientes mediante validación cruzada; cambian la
   cohorte y el esquema de evaluación.

## Artefactos reproducibles

- `results_vision/stunet_vs_resnet18_2p5d_preliminary_50/summary.json`
- `results_vision/stunet_vs_resnet18_2p5d_preliminary_50/per_seed_metrics.csv`
- `results_vision/stunet_vs_resnet18_2p5d_preliminary_50/heldout_predictions.csv`
- `results_vision/stunet_vs_resnet18_2p5d_preliminary_50/splits.csv`
- `results_vision/stunet_vs_resnet18_2p5d_preliminary_50/cohort_common.csv`
- `results_vision/stunet_vs_resnet18_2p5d_preliminary_50/provenance.json`

La corrida se reproduce con `code/tools/evaluate_stunet_vs_resnet2p5d.py`.
