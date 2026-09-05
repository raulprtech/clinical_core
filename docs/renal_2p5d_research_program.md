# Programa experimental renal 2.5D

Inicio: 2026-09-04. Estado: en ejecución.

## Objetivo y alcance

Evaluar recorte anatómico, adaptación ligera de ResNet18 y radiomics 2D;
identificar oportunidades a partir de los resultados y ejecutar seguimientos
justificados. Documentar protocolos, resultados negativos, decisiones y
limitaciones. No declarar completado el programa hasta evaluar las tres líneas
y resolver los seguimientos registrados.

## Predeclaración inicial

Los experimentos son exploratorios sobre TCGA-KIRC reutilizado. La inferencia
posterior debe reconocer todas las comparaciones realizadas. Una mejora visual
no demuestra aporte incremental sobre tabular ni generalización externa.

1. Recorte: referencia de campo completo, control con selección axial renal
   sin recorte y recorte renal con margen físico fijo de 10 mm. Mantener encoder
   ResNet18 congelado y máximo de 64 tokens, vecinos [-1,0,+1]. Comparar sobre
   la misma intersección de pacientes con máscaras reales verificadas.
2. Adaptación: comparar el último bloque ResNet18 congelado/adaptado dentro de
   cada partición de entrenamiento, usando el mismo recorte y cabeza en ambos
   brazos. BatchNorm congelada. Hacer primero un piloto de memoria y gradientes.
3. Radiomics: estadísticas de intensidad, forma 2D y textura explícitas dentro
   de máscaras renales, resumidas entre cortes; sin relleno aleatorio ni mocks.
   Estandarización, selección/reducción y regularización ajustadas solo en train.

Radiomics inicial: por riñón se usa el corte de mayor área segmentada; intensidad
y textura con ventana [-150,250] HU y 16 niveles fijos, forma con espaciado físico.
Resumir ambos riñones mediante media y diferencia absoluta. No seleccionar lado
usando desenlaces. El wrapper legacy de PyRadiomics no se usa.

Los tres brazos se reconstruyen desde el mismo NIfTI original y su máscara
geométricamente coincidente; el control de campo completo se vuelve a extraer.
El control de selección renal y el recorte comparten los centros axiales.

La ROI renal no es una segmentación tumoral y puede omitir tumor exofítico.
Reutilizar máscaras obtenidas con STU-Net implica un localizador 3D previo;
el encoder pronóstico nuevo sigue siendo 2.5D. Registrar esta dependencia.

## Evaluación y seguimientos

Outer CV estratificada por paciente: 5 folds x 3 repeticiones, semilla 4049;
inner CV 3 folds. Mantener todas las comparaciones pareadas dentro de cada
cohorte. Reportar C-index por fold y repetición, diferencias e intervalos
agrupados por paciente; tratar inferencia pooled OOF como secundaria por la
variación de escala entre modelos de distintos folds. Documentar multiplicidad
de contrastes y no interpretar intervalos exploratorios como confirmación.

Antes de cada seguimiento, registrar hipótesis, evidencia motivadora, cambios
exactos y criterio de decisión. No seleccionar pacientes usando resultados de
supervivencia. Nunca mezclar cortes del mismo paciente entre train y test.

## Registro de ejecución

Adaptación predeclarada antes de entrenamiento: 16 tokens uniformes del recorte,
prefijo ImageNet congelado hasta layer3, cabeza lineal sobre media de features
normalizada. Control idéntico con layer4 congelado. Candidatos de época 1/3/5,
3 inner folds, lr head=1e-3, lr layer4=1e-5, weight decay=1e-3 y BatchNorm fija.
Gradiente Cox Breslow calculado con el conjunto completo de entrenamiento y
repropagado por paciente en dos pasadas; prueba contra gradiente monolítico.
Los dos brazos se reinicializan dentro de cada partición. El piloto mide cambio
real de pesos, memoria y tiempo; no selecciona configuración por C-index.

Auditoría Cox: la pérdida secuencial histórica usa acumulación ordenada sin
agrupación de empates. En los 75 casos actuales no hay tiempos empatados con
eventos, por lo que no afecta a esta comparación Mamba. La adaptación nueva usa
Breslow explícito para admitir correctamente empates en cohortes futuras.

El comparador Cox inicial incluye media de tokens de cada uno de los tres
brazos y radiomics renal de 27 medidas. Pipeline train-only: scaler, PCA 4/8,
scaler de componentes y Cox ridge con alpha 100/10/1; selección en 3 inner folds.
Contrastes: crop-full, renal_slices-full y radiomics-crop. Mamba usa configuración
histórica fija (128D, estado16, dos bloques, dropout0.1, lr=weight_decay=0.001,
200 épocas máximas, paciencia20). Contrastes: crop-full, renal_slices-full,
crop-renal_slices. Incertidumbre: 5,000 bootstraps por paciente de la media de
C-index calculados dentro de cada outer fold, más ajuste Holm dentro de corrida.
El bootstrap condiciona en los modelos entrenados; no estima incertidumbre de
reentrenamiento. Guardar checkpoints de predicción locales por fold para reanudar.

- Inspección inicial: rama codex/resnet-mamba-fastproof limpia; GPU RTX 3050 Ti
  4 GB. Cachés STU-Net existentes por caso. La utilidad legacy de radiomics
  incluye fallback mock y no es apta para estas evaluaciones científicas.
- Pendiente al inicio: auditoría geométrica de máscaras y cobertura, extracción,
  evaluaciones de las tres líneas, seguimientos e informe consolidado.

## Cierre de ejecución — 2026-09-05

Completadas las tres líneas y siete seguimientos motivados: E1-E3 y F1-F7,
además del piloto técnico y el diagnóstico post hoc D1. Protocolos de cada
seguimiento registrados antes de su extracción/entrenamiento, salvo D1 que
se identifica expresamente como reanálisis de predicciones ya disponibles.
Resultados, comandos y revisiones en `renal_2p5d_program_results.md`.
La auditoría final verifica las once salidas y las trece pruebas pasan.
Revisión visual autorizada de cuatro cortes completada; no valida clínicamente
las máscaras ni la cobertura tumoral. No quedan entrenamientos declarados
pendientes. Convergencia extensa y validación independiente son limitaciones
de una etapa futura, no tareas que se presenten como realizadas aquí.
