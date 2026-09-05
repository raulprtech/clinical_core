# Resultados del programa renal 2.5D

Estado: en ejecución. Protocolo: `renal_2p5d_research_program.md`.

## Cobertura y verificación

75 CT con máscaras STU-Net existentes; 20 eventos. No se agregaron pacientes ni
se generaron máscaras sintéticas. Correspondencia de geometría, hashes de entrada
y centros axiales del control/recorte verificados en los 75 casos. Siete casos
presentan un único riñón en la máscara; no equivale a confirmar nefrectomía.
La caja con margen10mm ocupa mediana 15.59% del área del corte (2.75%-28.47%).
El montaje de cuatro ejemplos permanece local. Revisión visual pendiente de
autorización; no confundir controles geométricos con validación clínica del ROI.

Las máscaras son renales, no tumorales, y dependen de inferencia 3D previa.

## E1 — Cox sobre media de tokens y radiomics explícita

75 pacientes/20 eventos, 5x3 outer folds, 3 inner folds. Media de C-index
calculados dentro de cada outer fold (no OOF pooled histórico).

| Representación | Media C-index dentro de fold |
|---|---:|
| Campo completo, media de tokens | 0.7829 |
| Selección axial renal, media de tokens | 0.7567 |
| Recorte renal, media de tokens | 0.7343 |
| Radiomics renal 2D, 27 medidas | 0.6655 |

Recorte-campo completo: -0.0486, IC95% [-0.1836,+0.0781].
Selección renal-campo completo: -0.0262, IC95% [-0.1194,+0.0654].
Radiomics-recorte: -0.0688, IC95% [-0.2005,+0.1028]. Todos los p ajustados Holm
dentro de corrida son 1.0. Ninguna diferencia demuestra superioridad.

Artefactos: `results_vision/renal_2p5d_program_cox_v1/`.

## E2 — Mamba y recorte renal

En ejecución: mismos tres brazos, cohorte y particiones; Mamba-64 sin posición.
No se escogerá la representación por folds individuales.

## E3 — Adaptación ligera

Piloto superado: 60 pacientes de entrenamiento, una época en 3.54 s, memoria
GPU asignada máxima 182.77 MiB; cambio máximo observado de peso layer4 1e-5,
riesgos held-out finitos. No se evaluó mejora de C-index en este piloto.
La comparación nested congelado/adaptado está en ejecución.

Artefacto técnico: `results_vision/renal_2p5d_adaptation_pilot/technical_pilot.json`.

## Oportunidades derivadas de E1: predeclaración de seguimientos

F1. El campo completo conserva más señal media y el pooling medio puede perder
heterogeneidad. Comparar media+desviación estándar de tokens con su propia media,
tanto en campo completo como en recorte. Mismos 75 pacientes y splits, misma
familia PCA4/8 + Cox alpha100/10/1. Dos contrastes pareados, bootstrap5000 y Holm.
Motivación adicional: antecedente de mejora con momentos de STU-Net; no asumir
que se transfiere a ResNet. No mezclar cifras de ambas cohortes/evaluadores.

F2. Una representación débil por sí sola puede ser complementaria. Evaluar
tabular+campo completo y tabular+radiomics frente al mismo tabular. Usar utilidad
de fusión existente: PCA4/8 para imagen/radiomics, penalizadores0.1/1/10/100,
riesgos cross-fitted de tres inner folds y pesos convexos en pasos0.1 elegidos
solo en outer-train; percentiles de riesgo referidos a train. Misma 5x3 CV con
semilla4049. Evaluar ambos unimodales y ambas fusiones en la intersección común.
Dos contrastes frente a tabular con bootstrap5000 y Holm. La búsqueda interna
de hiperparámetros y pesos comparte validación dentro de outer-train; la
evaluación externa del fold permanece independiente de esas selecciones.

Estos seguimientos se registran después de E1 y antes de sus propias corridas;
son exploratorios y no equivalen a una validación independiente.

F1 completado: campo completo media+std 0.7993 vs media0.7829, delta+0.0164,
IC95% [-0.0483,+0.0923]; recorte media+std0.7196 vs media0.7343,
delta-0.0146, IC95% [-0.0837,+0.0687]. Controles reproducidos exactamente
en sus 450 predicciones. Ningún contraste significativo (Holm1.0).

F2 completado: tabular0.7319, campo completo0.7818, radiomics0.6651,
tabular+campo completo0.8047, tabular+radiomics0.7372.
Delta fusión visual-tabular+0.0728, IC95% [-0.0214,+0.1653], Holm0.2536;
delta fusión radiomics-tabular+0.0053, IC95% [-0.0599,+0.1066], Holm0.7224.
Los modelos unimodales de F2 se reajustan con el comparador de fusión existente
(lifelines y percentiles train); no son los mismos fits scikit-survival de E1.

F3 predeclarado tras F1/F2: combinar momentos de campo completo con tabular.
Comparar contra tabular+media y contra tabular dentro del mismo protocolo F2.
No seleccionar pesos con held-out. Reproducir los controles F2 como verificación.

F4 predeclarado tras F1/F2: extender solo media vs media+std de campo completo a
los 214 casos/64 eventos del caché DICOM histórico, sin necesidad de máscaras.
Los dos brazos usarán ese mismo caché, PCA4/8 y Cox alpha100/10/1, outer5x3
semilla4049, inner3 y bootstrap5000. No comparar numéricamente medias entre las
cohortes75/214 como un efecto arquitectónico. No es validación externa: incluye
los casos anteriores y datos usados en experimentos históricos.

## Reproducción inicial

```bash
.venv/bin/python code/tools/build_renal_2p5d_program_cache.py \
  --source data/embeddings/vision/stunet_volumetric_moments_pilot_76 \
  --output data/embeddings/vision/renal_2p5d_program_v1 --device cuda
.venv/bin/python code/tools/evaluate_renal_2p5d_program.py \
  --cache data/embeddings/vision/renal_2p5d_program_v1 \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --output results_vision/renal_2p5d_program_cox_v1 --kind cox --device cpu
# Para Mamba: cambiar --kind a mamba, --device a cuda y usar otra salida.
.venv/bin/python code/tools/evaluate_renal_resnet_adaptation.py \
  --cache data/embeddings/vision/renal_2p5d_program_v1 \
  --prefix-cache data/embeddings/vision/renal_2p5d_prefix_v1 \
  --targets results/20260715_174428_6da68b83/raw_targets.csv \
  --output results_vision/renal_2p5d_adaptation_v1 --device cuda
```

Pendientes para cierre: terminar E2/E3, ejecutar F1/F2, verificar resultados,
registrar decisiones y evaluar si justifican otro seguimiento concreto.
