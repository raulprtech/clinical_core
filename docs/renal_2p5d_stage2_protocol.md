# Segunda etapa 2.5D — protocolo antes de entrenamiento

Fecha: 2026-09-05. Objetivo autorizado: continuar las opciones propuestas y
entregar una conclusión. Se incluyen las tres prioritarias y el piloto DINOv2.
Los resultados anteriores quedan inmutables. Esta etapa sigue siendo exploratoria
en una cohorte reutilizada, no confirmación externa.

## Invariantes

75 CT/20 eventos de la etapa renal; mismos pacientes y outer5x3 seed4049,
inner3. Comparaciones pareadas, C-index medio dentro de outer fold,
bootstrap5000 por paciente, predicciones fijas y Holm por corrida.
Datos, imágenes, pesos adaptados, curvas por paciente y predicciones permanecen
locales. GitHub recibe código, protocolos, informes y métricas agregadas.
No cambiar defaults de producción por una media exploratoria alta.

## S1 — Ventana de entrenamiento y parada interna

Campo completo,16 tokens, prefijo ResNet18 hasta layer3 congelado; layer4
congelado o adaptado, cabeza lineal. AdamW lrhead1e-3, lrlayer4=1e-5,
weight_decay1e-3, BN fija, clipping5 y Cox Breslow con riesgo completo.
Hasta100 épocas, mínimo20 antes de detener, paciencia15 sin mejora del
C-index interno mayor de1e-8. Registrar curvas, mejor época y motivo de parada.
Seleccionar mediana redondeada de las tres mejores épocas internas y reinicializar
para refit outer-train. No elegir época usando outer-test. Es una nueva regla
de selección, no reproducción exacta de F6. Comparación principal adaptado-frozen.
Parada temprana significa saturación de validación según esta regla; no prueba
convergencia matemática ni descarta que otro presupuesto dé otro resultado.

## S2 — Mamba con encoder adaptado

Sustituir cabeza lineal por Mamba128D,estado16,dos bloques,sin posición.
Entrenar conjuntamente layer4 y Mamba dentro de cada inner/outer train; control
idéntico con layer4 congelado. Esto evita construir un cache supervisado global
que filtre desenlaces hacia validación. Mismos16 tokens, pérdidas, optimizadores,
semillas y parada S1. Dropout0 en ambos brazos para que las dos pasadas de
gradiente correspondan al mismo forward; BN fija. No comparar directamente
con Mamba64/dropout0.1 como una ablación pura. Principal adaptado-frozen.
Piloto técnico previo: gradiente correcto, pesos que cambian, BN estable,
riesgos finitos y consumo de memoria. No seleccionar hiperparámetros con piloto.
La ejecución optimizada calcula Mamba en lote de pacientes y propaga el gradiente
de tokens al encoder paciente por paciente; una prueba compara todos los
gradientes con las dos pasadas sin batching. El control congelado puede guardar
tokens pretrained dentro del fit; nunca se reutilizan tokens de un encoder adaptado
entre particiones. Esta optimización no cambia el objetivo ni el protocolo.

## S3 — Campo completo más detalle renal

Usar caches congelados existentes full y renal_letterbox,64 tokens cada uno,
media512D por vista. Cox por modalidad con PCA4/8 y penalizadores0.1/1/10/100,
transformaciones train-only. Riesgos inner cross-fitted y percentiles referidos
a train, pesos convexos en pasos0.1 escogidos en outer-train. Evaluar full,
local y full+local. Contraste principal fusión-full; sin búsqueda de márgenes
ni selección del local según held-out. El selector de hiperparámetros y pesos
comparte validación interna; outer-test nunca interviene.

## S4 — Piloto y evaluación DINOv2 pequeño

DINOv2 ViT-S/14 público (21M parámetros), congelado, sobre las mismas16 imágenes
de campo completo ya guardadas. Comparador ResNet18 congelado reextraído de
esas mismas imágenes, mismas normalización ImageNet y geometría224x224.
No preentrenar desde cero ni usar un modelo de radiografía como si fuera de CT.
Extraer token CLS384D DINO y features512D ResNet, normalizar cada token L2 y
promediar por paciente; PCA4/8 y Cox alpha100/10/1, mismos folds y bootstrap.
Primario DINO-ResNet. Las dimensiones distintas se controlan con la misma
cuadrícula PCA, no con una proyección elegida usando test. Descargar solo código
y pesos oficiales con revisión de licencia/procedencia; ningún dato sale a servicios.
Si no hay acceso o compatibilidad, agotar comprobaciones seguras y documentarlo,
no inventar métricas. Un piloto de memoria no es evidencia de rendimiento.

## Criterio de conclusión

Evaluar las cuatro líneas y sus controles; registrar magnitud, incertidumbre,
estabilidad por repetición y límites de entrenamiento. Una señal favorable
identifica un candidato para validación, no un ganador clínicamente validado.
No ampliar automáticamente arquitecturas o cuadrículas según el mayor outer C-index.
