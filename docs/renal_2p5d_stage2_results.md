# Segunda etapa 2.5D — resultados y seguimiento

Estado: en ejecución. Protocolo: `renal_2p5d_stage2_protocol.md`.
Mismos75 CT/20 eventos, outer5x3 seed4049, inner3. Datos reutilizados:
interpretación exploratoria, no confirmación externa.

## S1 — Parada interna y mayor presupuesto

En ejecución. Campo completo16 tokens, cabeza lineal, encoder congelado/adaptado.
Máximo100, mínimo20, paciencia15 y selección de mediana de mejores épocas internas.
Esta regla cambia respecto a F6; reportar resultados como nueva corrida.

## S2 — Mamba y encoder adaptado conjuntamente

En ejecución, con controles idénticos y entrenamiento estrictamente por partición.
Piloto técnico sin batching:60 pacientes train,10.772s/época,187.629MiB de GPU
asignada; batching:3.858s,187.160MiB. Ambos dan pérdida inicial3.22703576,
cambio máximo de layer4=1.00136e-5 y predicciones held-out finitas.
La prueba de gradientes compara ejecución por lotes con ejecución individual
en brazos congelado y adaptado. No usar el piloto para seleccionar rendimiento.
Las mediciones de tiempo no son benchmarks aislados de carga del sistema.

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

Pendiente: terminar S1/S2, auditar cobertura, emparejamiento, curvas y controles,
consolidar la conclusión y publicar los agregados verificados.
