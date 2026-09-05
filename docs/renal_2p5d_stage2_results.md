# Segunda etapa 2.5D — resultados y seguimiento

Estado: cuatro experimentos completados y auditados; publicación de los últimos
agregados pendiente de autorización explícita para GitHub.
Protocolo: `renal_2p5d_stage2_protocol.md`.
Mismos75 CT/20 eventos, outer5x3 seed4049, inner3. Datos reutilizados:
interpretación exploratoria, no confirmación externa.

## S1 — Parada interna y mayor presupuesto

Completado. Campo completo16 tokens, cabeza lineal, encoder congelado/adaptado.
Máximo100, mínimo20, paciencia15 y selección de mediana de mejores épocas internas.
Esta regla cambia respecto a F6; reportar resultados como nueva corrida.

Adaptado0.7902 frente a congelado0.7399; delta+0.0504,
IC95%[+0.0112,+0.1182], p0.0128 por corrida. La sensibilidad Holm entre los
cuatro contrastes da0.0512: señal exploratoria, no confirmación robusta ni
justificación para cambiar defaults. Los intervalos mostrados son marginales,
no intervalos simultáneos ajustados por multiplicidad.
Las90 curvas internas pararon por paciencia, ninguna alcanzó100 épocas;
las épocas de refit seleccionadas abarcan2–22 adaptado y7–32 congelado.
La auditoría verificó450 filas de predicción y30 selecciones de época.
Dar más presupuesto no aumentó numéricamente el resultado adaptado respecto
a F6 (0.7933), aunque también cambió la regla de selección: no es un contraste
formal ni prueba de equivalencia entre presupuestos.

## S2 — Mamba y encoder adaptado conjuntamente

Completado y auditado, con controles idénticos y entrenamiento estrictamente por partición.
Piloto técnico sin batching:60 pacientes train,10.772s/época,187.629MiB de GPU
asignada; batching:3.858s,187.160MiB. Ambos dan pérdida inicial3.22703576,
cambio máximo de layer4=1.00136e-5 y predicciones held-out finitas.
La prueba de gradientes compara ejecución por lotes con ejecución individual
en brazos congelado y adaptado. No usar el piloto para seleccionar rendimiento.
Las mediciones de tiempo no son benchmarks aislados de carga del sistema.

Resultado: Mamba adaptado0.8199 frente a congelado0.8106;
delta+0.0093, IC95%[-0.0171,+0.0476], p0.5224. No demuestra una ventaja
incremental de adaptar conjuntamente el encoder; no promover este cambio.
El auditor verificó cobertura y etiquetas de las450 filas de predicción,
recalculó métricas y bootstrap, y revisó las90 curvas internas y las30 selecciones
de época. Esta comprobación no vuelve a entrenar los modelos.
Ninguna selección interna alcanzó el máximo100: todas pararon por paciencia.
Esto no prueba convergencia matemática ni descarta otros optimizadores.
Los deltas medios por repetición fueron+0.0111,+0.0012,+0.0158; son positivos
pero las repeticiones comparten pacientes y no son tres replicaciones independientes.
El resultado tampoco demuestra equivalencia ni permite atribuir diferencias
frente al Mamba histórico64 tokens únicamente al encoder (aquí16 tokens).

## S3 — Fusión global-local

Completado. Campo completo0.7818, local renal letterbox0.7276,
fusión0.7982. Diferencia fusión-full+0.0164, IC95%[-0.0461,+0.0673],
p=0.622 (único contraste). No demuestra ventaja incremental; no promover.
El local se fijó antes de la corrida y no se seleccionó mediante outer-test.

## S4 — DINOv2 congelado frente a ResNet18

Piloto superado:16 imágenes del primer caso ordenado, finitud y centros pareados,
1.497s de extracción,162.060MiB de memoria GPU asignada. No se calculó rendimiento
clínico ni se eligió el caso por desenlace. Extracción completa:75/75 casos,
27.576s, mismo pico162.060MiB; fuentes y centros pareados, características finitas.

Evaluación completada: ResNet18 0.7951, DINOv2 0.7915;
delta -0.0035, IC95%[-0.1302,+0.1126], p0.8964. No demuestra ventaja ni
equivalencia: el intervalo permite diferencias relevantes en ambos sentidos.
No promover DINOv2 congelado CLS sobre este comparador ni inferir resultados
para toda la familia DINO, otros pooling o modelos especializados en CT.

Fuente oficial: [DINOv2](https://github.com/facebookresearch/dinov2), revisión
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`. ViT-S/14 sin registros, CLS384D,
pesos pretrained públicos, código y ficha del modelo Apache2.0. Se usa como
extractor de investigación, no modelo clínico validado. Los pesos se cargan con
`weights_only=True`; se registra SHA256 de pesos, código y entradas locales.
[Ficha de modelo](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md).
No está instalado xFormers; la ruta PyTorch estándar superó el piloto sin añadir
dependencias. ResNet y DINO reciben las mismas16 imágenes224x224 y normalización.

## Conclusión de las cuatro opciones

| Contraste pareado | C-index candidato / control | Delta e IC95% marginal | p por corrida | Holm secundario entre4 |
|---|---|---|---|---|
| S1 lineal adaptado / congelado | 0.7902 / 0.7399 | +0.0504 [+0.0112,+0.1182] | 0.0128 | 0.0512 |
| S2 Mamba adaptado / congelado | 0.8199 / 0.8106 | +0.0093 [-0.0171,+0.0476] | 0.5224 | 1.0000 |
| S3 global-local / global | 0.7982 / 0.7818 | +0.0164 [-0.0461,+0.0673] | 0.6220 | 1.0000 |
| S4 DINOv2 / ResNet18 | 0.7915 / 0.7951 | -0.0035 [-0.1302,+0.1126] | 0.8964 | 1.0000 |

La tabla compara cada candidato con su propio control, no establece un ranking
causal entre pipelines distintos. Estimando: media de C-index dentro de los15
folds externos. No comparar estas cifras con C-index OOF agrupado histórico.

Estabilidad: los deltas por repetición de S1 son+0.0317,+0.0715,+0.0479;
S2+0.0111,+0.0012,+0.0158; S3-0.0050,+0.0574,-0.0034;
S4-0.0860,-0.0233,+0.0986. S3/S4 cambian de signo. Las repeticiones comparten
pacientes; ni los tres signos positivos de S1/S2 ni una repetición alta son
validaciones independientes.

**Decisión:** conservar los defaults actuales y el contexto de campo completo.
No se ha demostrado que modificar Mamba mediante adaptación conjunta, agregar
detalle renal o sustituir ResNet por este DINOv2 mejore el comparador respectivo.
La adaptación ligera con cabeza lineal sigue siendo el candidato más claro
para validación adicional, pero no ha demostrado superar el pipeline Mamba
existente; más presupuesto por sí solo no produjo un salto en las cifras.
No lanzar otra búsqueda amplia sobre estos mismos desenlaces ni promover el
mayor promedio como ganador validado.

El siguiente paso informativo es una evaluación independiente con configuración
cerrada y datos compatibles, o una hipótesis distinta de representación (por
ejemplo volumétrica) con su propio control. Esta etapa no demuestra que todas
las opciones2D/2.5D estén agotadas ni garantiza que3D mejore. Ningún nuevo
experimento queda lanzado como parte de esta conclusión.

Límites:75 pacientes y20 eventos, cohorte reutilizada y múltiples búsquedas
previas. El bootstrap5000 remuestrea pacientes con predicciones ya ajustadas;
no incorpora la variación de volver a entrenar ni corrige todas las búsquedas
históricas. La sensibilidad Holm entre4 fue registrada después de S3/S4 y antes
del cierre S1/S2. El valor0.0512 no es un umbral mágico: interpretar magnitud e
incertidumbre, sin afirmar ausencia de efecto ni eficacia clínica.

## Artefactos y reproducción

| ID | Script bajo code/tools/ | Salida bajo results_vision/ |
|---|---|---|
| S1 | evaluate_stage2_joint_adaptation.py --head linear | stage2_linear_convergence_v1 |
| S2 | evaluate_stage2_joint_adaptation.py --head mamba | stage2_joint_mamba_v1 |
| S3 | evaluate_stage2_global_local.py | stage2_global_local_v1 |
| S4 | build_stage2_dino_cache.py; evaluate_stage2_dino.py | stage2_dino_cox_v1 |

Ejecutar con `.venv/bin/python -u` desde la raíz del repositorio.
S1/S2 requieren `--output` correspondiente; los otros argumentos están fijados
por defecto en el protocolo. S3/S4 evaluador tienen salida predeterminada.
Extractor S4: `--output data/embeddings/vision/stage2_dino_v1` después de clonar
la revisión oficial indicada en `data/models/dinov2_source`.
Los scripts rechazan contratos de reanudación distintos; no modificar el código
de una corrida activa. Para variantes, usar una salida nueva y registrar hipótesis.

Revisiones: protocolo/piloto inicial `ca0ceb4`; S1/S2 optimizado y S3 `44d7945`;
extractor/evaluador S4 `331b0b4`. Tests actuales:17, incluyendo13 de la etapa previa.
Paciente, cohorte, splits y checkpoints permanecen fuera de Git. Los registros
de curvas de entrenamiento contienen solo métricas agregadas por fold/época.

Auditoría: `.venv/bin/python code/tools/verify_renal_stage2.py`.
Verifica cobertura, etiquetas, particiones, métricas y bootstrap; también
reproduce la regla de parada a partir de cada curva interna y la selección
de épocas. No sustituye validación clínica independiente.
Al cerrar las cuatro corridas, el auditor añadió una sensibilidad Holm sobre
sus cuatro contrastes primarios. Es un ajuste secundario registrado después
de conocer S3/S4 y antes de terminar S1/S2; no cambia modelos ni reemplaza los
resultados por corrida, y tampoco convierte la cohorte reutilizada en confirmatoria.

Auditoría final: cuatro corridas verificadas,2025 filas de predicción en total,
180 curvas internas y60 selecciones de época de S1/S2, ninguna curva al máximo100.
Las17 pruebas técnicas pasan. La auditoría no vuelve a entrenar los modelos.
Resultados y conclusión conservados localmente; queda pendiente autorización
explícita para publicar los informes y agregados en `raulprtech/clinical_core`,
rama `codex/resnet-mamba-fastproof`. No se publican imágenes, identificadores,
predicciones ni registros individuales de pacientes.
