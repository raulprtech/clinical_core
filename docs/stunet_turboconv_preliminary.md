# Evaluación preliminar STU-Net-S: PTQ frente a TurboConv

## Conclusión ejecutiva

La prueba W4A8 se completó con 20 casos de calibración y 50 casos de evaluación
independientes, sin solapamiento. TurboConv preservó mejor la salida FP32 que PTQ
convencional, pero no alcanzó todos los umbrales de retención predeclarados. Por
ello, **W4A8 debe rechazarse para ambas variantes en su estado actual**.

TurboConv sí merece una siguiente prueba en W6A8: mejoró PTQ en logits en 50/50
casos, en embeddings en 47/50 y en Dice renal en 48/50. No se ha demostrado una
ventaja de velocidad o memoria porque ambas variantes usan fake-quant en punto
flotante y la transformación Hadamard no está fusionada en un kernel entero.

## Diseño

- Encoder: STU-Net-S congelada, checkpoint SHA-256
  `f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6`.
- Referencia: AMP/FP32 funcional.
- Cuantización: pesos de 4 bits y activaciones de 8 bits (W4A8).
- Calibración: 20 pacientes.
- Evaluación ciega: 50 pacientes diferentes.
- Solapamiento calibración–evaluación: 0.
- Semilla de rotación TurboConv: 2026.
- Salidas comparadas por paciente: embedding de 768 dimensiones, boceto de
  logits y máscara STU-Net (riñón izquierdo/derecho, etiquetas 38/39).
- Se excluyeron 6 casos QC-pass únicamente por el guard de cómputo basado en
  tamaño. Las tres variantes usaron exactamente los mismos 50 identificadores.
- TCGA-BP-4970 usó interpolación de máscara de orden 0 en las tres variantes
  debido a OOM reproducible con orden 1. Esto no modifica logits ni embeddings.

## Resultados de fidelidad respecto a FP32

| Métrica | PTQ W4A8 | TurboConv W4A8 | Umbral |
|---|---:|---:|---:|
| Coseno embedding, mediana | 0.989577 | **0.994955** | ≥ 0.99 |
| Coseno embedding, p05 | 0.705139 | **0.746960** | — |
| Coseno logits, mediana | 0.957216 | **0.976522** | ≥ 0.99 |
| Coseno logits, p05 | 0.944162 | **0.967467** | — |
| L2 relativo logits, mediana | 0.290375 | **0.216558** | — |
| Dice renal, mediana | 0.932771 | **0.963929** | ≥ 0.95 |
| Dice renal, p05 | 0.861003 | **0.893795** | ≥ 0.90 |
| Acuerdo global de máscara, mediana | 0.960964 | **0.970108** | — |
| Cambio de volumen renal, mediana | +8.36% | **−3.43%** | — |

PTQ falló los cuatro gates. TurboConv pasó los gates de embedding mediano y Dice
mediano, pero falló coseno mediano de logits y Dice p05. El p05 de Dice quedó
0.0062 por debajo del umbral; el coseno de logits quedó 0.0135 por debajo.

## Comparación pareada TurboConv − PTQ

Los intervalos son bootstrap exploratorios de la mediana pareada (20,000
remuestreos, semilla 2026); no sustituyen los gates predeclarados.

| Métrica | Diferencia mediana | IC 95% | Casos ganados por TurboConv |
|---|---:|---:|---:|
| Coseno embedding | +0.005175 | [0.003405, 0.007881] | 47/50 |
| Coseno logits | +0.017684 | [0.014807, 0.020182] | 50/50 |
| L2 relativo logits | −0.068057 | [−0.074325, −0.060272] | 50/50 |
| Dice renal | +0.028691 | [0.021257, 0.034620] | 48/50 |
| Acuerdo global de máscara | +0.008754 | [0.007534, 0.011544] | 48/50 |

TurboConv redujo el error L2 relativo mediano de logits aproximadamente 23.4%
respecto al valor PTQ mediano. La cuantización de pesos también tuvo menor error
relativo L2: 0.178196 frente a 0.259751 (31.4% menos).

## Eficiencia

| Medida | FP32 | PTQ W4A8 | TurboConv W4A8 |
|---|---:|---:|---:|
| Tiempo mediano por caso | 177.80 s | 178.22 s | 185.97 s |
| Pico mediano de memoria CUDA | 2.181 GiB | 2.244 GiB | 2.183 GiB |
| Reducción teórica de almacenamiento de pesos | 1× | 8× | 8× |

La diferencia pareada mediana de tiempo TurboConv−PTQ fue +0.59 s, con IC 95%
[−4.56, 14.37]; no hay evidencia de aceleración o penalización concluyente. El
pico de memoria TurboConv fue esencialmente igual a FP32. Estos números son los
esperados para fake-quant: los pesos se simulan a 4 bits, pero se ejecutan con
kernels de punto flotante. Solo un backend entero/fusionado permitiría comprobar
la reducción real de almacenamiento, VRAM, energía y latencia.

## Casos de cola y cautelas

- El peor Dice renal TurboConv fue 0.7680 (TCGA-B0-5099), con cambio de volumen
  de −34.9%.
- Los peores cosenos de embedding TurboConv fueron 0.5593 (TCGA-B8-4620),
  0.5817 (TCGA-CW-5589) y 0.6548 (TCGA-CW-6097).
- TCGA-CW-6087, TCGA-BP-4797 y TCGA-CJ-4887 presentaron advertencias de
  muestreo DICOM no uniforme, pero pasaron el QC predefinido y se procesaron de
  forma idéntica en las tres variantes.
- La prueba mide deriva frente al encoder FP32, no accuracy contra segmentaciones
  manuales ni C-index de una cabeza de supervivencia. Antes de uso clínico se
  requiere validar el impacto downstream con la misma cohorte y splits.

## Recomendación

1. No usar PTQ W4A8 ni TurboConv W4A8 para la comparación final de supervivencia.
2. Conservar TurboConv como candidato: es inequívocamente mejor que PTQ al mismo
   bit-width y estuvo cerca de los gates de máscara.
3. Ejecutar un kill test W6A8 TurboConv frente a PTQ sobre los mismos 20+50 casos.
   El sanity check de pesos favorece esta opción (L2 relativo 0.040277 frente a
   0.060520 en PTQ) y ofrece una reducción teórica de pesos de 5.33× frente a
   FP32.
4. Si W6A8 supera todos los gates, materializar un kernel entero/fusionado y medir
   latencia, VRAM, energía y tamaño reales. Solo entonces avanzar a la cabeza de
   supervivencia y cinco semillas.

## Artefactos autoritativos

- `data/embeddings/vision/stunet_turboconv_preliminary_20x50/paired_drift.csv`
- `data/embeddings/vision/stunet_turboconv_preliminary_20x50/summary.csv`
- `data/embeddings/vision/stunet_turboconv_preliminary_20x50/verdict.json`
- `data/embeddings/vision/stunet_turboconv_preliminary_20x50/provenance.json`
- `data/embeddings/vision/stunet_turboconv_preliminary_20x50/calibration/weight_sanity.csv`

