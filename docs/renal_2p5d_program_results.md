# Resultados del programa renal 2.5D

Estado: en ejecución. Protocolo: `renal_2p5d_research_program.md`.

## Cobertura y verificación

75 CT con máscaras STU-Net existentes; 20 eventos. No se agregaron pacientes ni
se generaron máscaras sintéticas. Correspondencia de geometría, hashes de entrada
y centros axiales del control/recorte verificados en los 75 casos. Siete casos
presentan un único riñón en la máscara; no equivale a confirmar nefrectomía.
La caja con margen10mm ocupa mediana 15.59% del área del corte (2.75%-28.47%).
El montaje de cuatro ejemplos permanece fuera de GitHub. El usuario autorizó
su inspección en esta tarea el 2026-09-05. Se revisaron los cuatro cortes:
sin desplazamiento global evidente entre imagen, contornos y caja; el tercer
contorno es más irregular y fragmentado. No se estableció cobertura del tumor
ni exactitud clínica de las máscaras. No confundir esta inspección limitada
con validación radiológica del ROI ni con revisar todos los cortes de75 CT.

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

Completado: mismos tres brazos, cohorte y particiones; Mamba-64 sin posición.
Campo completo 0.8187, selección axial renal sin recorte 0.8242, recorte 0.6990.
Recorte-campo completo: -0.1197, IC95% [-0.2286,-0.0128], Holm 0.0688.
Selección renal-campo completo: +0.0055, IC95% [-0.0639,+0.0639], Holm 0.8504.
Recorte-selección renal: -0.1252, IC95% [-0.2208,-0.0281], Holm 0.0372.
El contraste más limpio de recorte, con idénticos centros axiales, evidencia
degradación dentro de esta corrida. Conservar contexto de campo completo;
no promover el recorte ni atribuir ventaja confirmada a selección axial renal.
Esto no demuestra que el tumor esté mal segmentado ni identifica el mecanismo.

## E3 — Adaptación ligera

Piloto superado: 60 pacientes de entrenamiento, una época en 3.54 s, memoria
GPU asignada máxima 182.77 MiB; cambio máximo observado de peso layer4 1e-5,
riesgos held-out finitos. No se evaluó mejora de C-index en este piloto.
La comparación nested congelado/adaptado terminó: congelado 0.6799,
adaptado 0.7161, delta +0.0362, IC95% [-0.0207,+0.0958], p=0.1792.
El congelado eligió 5 épocas en 15/15 folds y el adaptado en 13/15 (los otros
dos eligieron 3). La selección toca el límite superior; no asumir convergencia.

F5 predeclarado después de E3: repetir ambos brazos con candidatos1/3/5/10/20
épocas. Mantener datos, arquitectura, inicializaciones, learning rates,
BatchNorm, particiones y algoritmo de selección. El contraste principal sigue
siendo adaptado-congelado dentro de esta nueva corrida. Registrar selección de
épocas y cualquier persistencia del límite, sin promover el mejor valor post hoc.

Artefacto técnico: `results_vision/renal_2p5d_adaptation_pilot/technical_pilot.json`.

F5 completado: congelado 0.7474, adaptado 0.7545; delta +0.0071,
IC95% [-0.0460,+0.0778], p=0.7368. La diferencia inicial de E3 se reduce al
permitir más entrenamiento al control. No demuestra una ventaja incremental
de adaptar layer4. Los 270 valores de validación interna compartidos de épocas
1/3/5 reproducen exactamente E3. Esto verifica continuidad del protocolo, no
convierte la comparación entre corridas en validación independiente.

| Brazo F5 | Selección 5 épocas | Selección 10 | Selección 20 |
|---|---:|---:|---:|
| Congelado | 2/15 | 5/15 | 8/15 |
| Adaptado | 2/15 | 2/15 | 11/15 |

El límite 20 continúa activo; no afirmar convergencia ni descartar todo posible
fine-tuning. Las medias de validación interna entre 10 y 20 suben de 0.7256 a 0.7328
(congelado) y de 0.7393 a 0.7495 (adaptado). Son diagnósticos descriptivos de folds
solapados, no nuevas observaciones independientes. No justificar otra ampliación
automática de épocas con el resultado held-out. F6 prueba la oportunidad de
preservar contexto anatómico; una búsqueda de convergencia más extensa queda
como limitación, no como experimento ya realizado.

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

F1 completado: campo completo media+std 0.7993 vs media 0.7829, delta +0.0164,
IC95% [-0.0483,+0.0923]; recorte media+std 0.7196 vs media 0.7343,
delta -0.0146, IC95% [-0.0837,+0.0687]. Controles reproducidos exactamente
en sus 450 predicciones. Ningún contraste significativo (Holm 1.0).

F2 completado: tabular 0.7319, campo completo 0.7818, radiomics 0.6651,
tabular+campo completo 0.8047, tabular+radiomics 0.7372.
Delta fusión visual-tabular +0.0728, IC95% [-0.0214,+0.1653], Holm 0.2536;
delta fusión radiomics-tabular +0.0053, IC95% [-0.0599,+0.1066], Holm 0.7224.
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

F3 completado: tabular+momentos 0.8004 frente a tabular+media 0.8047,
delta -0.0043, IC95% [-0.0530,+0.0512], Holm 0.8604. Frente a tabular,
delta +0.0685, IC95% [-0.0407,+0.1768], Holm 0.4344. No se promueve momentos
como reemplazo de media en fusión.

F4 completado (214 casos/64 eventos): media 0.6771, momentos 0.6515,
delta -0.0256, IC95% [-0.0586,+0.0057], p=0.108. La señal positiva de la
subcohorte 75 no se sostiene en esta extensión interna. No seguir ampliando
estadísticas de pooling ResNet a partir de la clasificación de medias.

## Reproducción inicial

F6 predeclarado el 2026-09-05 mientras F5 aún está en ejecución: la degradación
del recorte en E2 motiva trasladar la adaptación ligera a campo completo.
Reconstruir 16 imágenes de los 64 centros del brazo full ya existente, mediante
la misma selección uniforme de posiciones usada por E3/F5. Mismos 75 pacientes,
CT NIfTI, ventana HU, vecinos, resolución, inicializaciones, optimizer y folds;
sin nuevos pacientes ni elección por desenlace. Usar candidatos1/3/5/10/20 como
F5 y comparar adaptado-congelado dentro de F6 (único contraste principal,
bootstrap5000). Las comparaciones entre F5 y F6 son descriptivas, no una nueva
búsqueda del mejor brazo. No atribuir a campo completo un efecto independiente
del cambio de centros axiales. Esta prueba no establece convergencia si vuelve
a elegirse el máximo de épocas. No ampliar automáticamente la búsqueda de
épocas o arquitecturas por el mayor C-index observado.

Salida: `results_vision/fullfield_2p5d_adaptation_v1/`; extractor específico
`code/tools/build_fullfield_adaptation_cache.py` y evaluador de adaptación F5
sin cambios. Entradas nuevas separadas de los cachés históricos.

F7 predeclarado tras la inspección visual autorizada, antes de extracción o
entrenamiento: el código de recorte redimensiona rectángulos a224x224 sin
conservar proporciones. Esta deformación confunde la interpretación de E2.
Contrastar Mamba con recorte renal más padding simétrico hasta cuadrado frente
al recorte estirado de E2. Padding al valor de fondo de la ventana (-150 HU),
sin añadir anatomía ni cambiar caja, centros, vecinos, intensidad, encoder,
semillas, folds o hiperparámetros. Comparación secundaria con campo completo.
Ambos contrastes bootstrap5000 y Holm dentro de corrida; sigue exploratorio.
Los controles de E2 se reutilizan literalmente, comprobando cohortes, splits,
fuentes y hashes; no se presentarán como modelos recién entrenados. Solo se
entrena el nuevo brazo. El padding conserva proporción de píxeles, no corrige
anisotropía física ni garantiza cobertura tumoral. No ajustar margen/padding
según resultados held-out. Salida: `renal_2p5d_aspect_mamba_v1`.

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

## Registro de corridas y reproducción

| ID | Salida bajo results_vision/ | Revisión de implementación |
|---|---|---|
| E1 | renal_2p5d_program_cox_v1 | 66d3cf8 |
| E2 | renal_2p5d_program_mamba_v1 | 66d3cf8 |
| E3 | renal_2p5d_adaptation_v1 | 25a28fa |
| F1 | renal_2p5d_followup_moments_v1 | e8794f5 |
| F2 | renal_2p5d_followup_fusion_v1 | e8794f5 |
| F3 | renal_2p5d_followup_fusion_moments_v1 | 3cd1224 |
| F4 | fullfield_moments_214_v1 | 3cd1224 |
| F5 | renal_2p5d_adaptation_extended_v1 | 59d4e07 |
| F6 | fullfield_2p5d_adaptation_v1 | 486c92f (extractor); evaluador59d4e07 |

Cada salida completa incluye summary, métricas por fold/repetición, bootstrap,
selección de hiperparámetros y procedencia. Cohortes, splits y predicciones,
incluidos checkpoints reanudables folds/, quedan excluidos de Git.
Las corridas rechazan reanudar con un contrato distinto. Para reproducir,
usar la revisión correspondiente y las entradas locales, o una salida nueva;
no sobrescribir una corrida histórica con código cambiado.

F1/F2/F3: `code/tools/evaluate_renal_2p5d_followups.py` con el cache y targets
anteriores, `--kind moments`, `--kind fusion` o `--kind fusion_moments` y salida
propia. Fusión requiere además
`--features results/20260715_174428_6da68b83/raw_features.csv`.

F4: `code/tools/evaluate_fullfield_moments_214.py` con
`--sequence-dir data/embeddings/vision/resnet18_2p5d_sequences`, los mismos
targets y salida propia. F5: el comando E3 con salida propia y
`--epoch-grid 1 3 5 10 20`. Usar `--pilot` para repetir únicamente el piloto
técnico de adaptación y `--prepare-only` para construir los prefijos congelados.

F6: construir imágenes con
`code/tools/build_fullfield_adaptation_cache.py --parent-cache data/embeddings/vision/renal_2p5d_program_v1 --output data/embeddings/vision/fullfield_2p5d_adaptation_v1`.
Ejecutar el evaluador F5 con ese `--cache`,
`--prefix-cache data/embeddings/vision/fullfield_2p5d_prefix_v1`,
`--output results_vision/fullfield_2p5d_adaptation_v1` y la misma cuadrícula de
épocas1/3/5/10/20. Los75 CT y centros se comprobaron mediante hashes;
las imágenes son finitas y están en[0,1]. No se realizó revisión visual.

Verificación agregada: `.venv/bin/python code/tools/verify_renal_2p5d_program.py`.
Informa corridas incompletas sin tratarlas como experimentos terminados.

La verificación recalcula métricas, intervalos bootstrap y p ajustados a partir
de predicciones locales; no reentrena modelos. Comprueba particiones disjuntas,
emparejamiento, cobertura y reproducción de controles. Las pruebas sintéticas
verifican geometría, muestreo, radiomics sin fondo, gradiente Cox/Breslow,
BatchNorm congelada e invariancia del ajuste Cox/fusión frente a cambios en
desenlaces held-out. Son11 pruebas, ejecutables con
`.venv/bin/python -m unittest discover -s code/tests -p test_renal_2p5d_program.py -v`.
No sustituyen una auditoría clínica ni demuestran ausencia universal de leakage.

El remuestreo descarta los folds sin pares comparables dentro de cada réplica;
usa los mismos pesos por paciente y fold para ambos brazos. La corrección Holm
es interna a cada corrida, no global sobre esta búsqueda adaptativa ni sobre
los experimentos históricos. Por ello incluso diferencias con Holm<0.05 se
interpretan como evidencia exploratoria. No hay conjunto externo intacto.

Entorno observado: `results_vision/renal_2p5d_program_audit/runtime_environment.json`.
Es un inventario de versiones, no un entorno bloqueado. Cachés nuevos locales:
programa renal286MiB, prefijo renal66MiB, imágenes full135MiB y prefijo full56MiB
(tamaños aproximados en disco). El piloto de182.77MiB mide memoria GPU asignada,
no RAM total ni reserva del controlador. No se descargó un nuevo backbone.

Pendientes para cierre: terminar F6, verificar todos los resultados,
registrar decisiones y evaluar si justifican otro seguimiento concreto.
