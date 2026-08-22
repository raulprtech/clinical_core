# Registro continuo de decisiones experimentales

Este documento conserva la evolución de los experimentos de TCGA-KIRC. Es un
registro acumulativo: las entradas nuevas se agregan con fecha y las decisiones
anteriores no se reescriben sin dejar una nota de corrección.

Cada entrada debe indicar pregunta, cohorte, protocolo, artefactos, resultado,
decisión, limitaciones y siguiente prueba. Los informes específicos contienen
el detalle numérico; este archivo mantiene la trazabilidad entre ellos.

## 2026-05-28 — Diagnóstico del VAE mediante concatenación tardía

- **Pregunta:** ¿la caída bimodal provenía del encoder o del VAE Stage A?
- **Protocolo:** concatenar tabular y ClinicalBERT (1,536D) y ajustar una cabeza
  Cox lineal sin VAE; se preparó también la reevaluación de TurboLatent.
- **Evidencia:** `experiment_config_late_fusion_text_n444.yaml` y
  `experiment_config_late_fusion_turbolatent_n444.yaml`.
- **Decisión de entonces:** pasar operacionalmente de VAE Stage A a
  concatenación para aislar la fuente de degradación.
- **Corrección posterior:** esta decisión era diagnóstica, no evidencia de que
  la concatenación fuese el fusionador óptimo.

## Reparación posterior — Comparación pareada de fusión trimodal

- **Pregunta:** ¿la caída trimodal representaba falta de señal o un protocolo
  no comparable y una cabeza sobreparametrizada?
- **Cohorte:** 210 pacientes comunes; unos 168 en outer-train por fold.
- **Hallazgos metodológicos:** las primeras ablaciones mezclaban cohortes; el
  preprocesamiento tabular usaba información de toda la cohorte; concatenar
  2,307 entradas producía `p >> n` y forzaba modalidades débiles.
- **Resultado:** tabular 0.7990, texto 0.6358, ResNet18 2.5D 0.6397, fusión
  igualitaria 0.7758 y fusión convexa train-only 0.8111. La convexa asignó peso
  cero a texto y 0.1--0.3 a visión.
- **Comparadores:** concatenación proyectada 0.7516, atención cruzada 0.7915,
  jerárquica residual 0.8066 y ortogonal+concatenación 0.7483.
- **Decisión vigente:** mantener la convexa como referencia principal y la
  jerárquica residual como comparador secundario. No regresar a concatenación
  cruda sin una justificación y un control explícito de dimensionalidad.
- **Detalle:** `docs/trimodal_fusion_repair.md`.

## Pilotos STU-Net — representación y cuantización

- **Representación FP32:** pooling de bounding box y riñones izquierdo/derecho,
  3 x 256D, seguido de normalización L2. El piloto completó 10/10 casos, con
  mediana de 186.2 s y coseno de repetición 0.99999975.
- **Restricción de cohorte:** de 190 CT, 76 tenían geometría uniforme y 114
  requerían atención de geometría; toda comparación final debe usar la misma
  intersección QC.
- **Supervivencia exploratoria:** en 50 pacientes y 13 eventos, STU-Net obtuvo
  0.6318 frente a 0.4796 de ResNet18 2.5D. Los intervalos fueron amplios y el
  resultado no es comparable con el baseline oficial de 214 pacientes.
- **Cuantización:** TurboConv W4A8 superó PTQ en fidelidad relativa, pero no
  pasó las compuertas predeclaradas y fue rechazado. W6A8 quedó como candidato.
  El ensayo con fake quantization no demuestra aceleración o ahorro de memoria.
- **Detalle:** `docs/stunet_fp32_pilot.md`,
  `docs/stunet_vs_resnet18_2p5d_preliminary_survival.md` y
  `docs/stunet_turboconv_preliminary.md`.

## 2026-08-21 — ResNet18 axial + attention/Mamba Fast Proof

- **Pregunta:** ¿modelar el stack axial completo mejora el resumen de tres
  vistas centrales sin requerir STU-Net/Colab?
- **Cohorte:** 214 pacientes, 64 eventos; cinco hold-outs pareados de 43
  pacientes y 13 eventos.
- **Protocolo:** mismo cache de tokens ResNet18 congelados para attention y
  Mamba; held-out ciego y selección exclusivamente en train.
- **Resultado:** PCA+Cox oficial 0.7170 ± 0.0987, attention 0.7252 ± 0.0539 y
  Mamba 0.7563 ± 0.0127. Mamba ganó 4/5 semillas frente al oficial.
- **Limitación:** ningún IC bootstrap pareado por semilla excluyó cero. El
  baseline reajusta con todo outer-train; las redes reservan validación para
  early stopping y no hacen un reajuste simétrico.
- **Decisión:** avanzar a nested repeated CV con uso simétrico de outer-train.
  Después, integrar el riesgo visual Mamba cross-fitted en el comparador de
  fusión convexa, sin reutilizar embeddings aprendidos con el held-out.
- **Detalle:** `docs/resnet18_sequence_mamba_fastproof.md` y
  `docs/resnet18_sequence_mamba_preliminary_results.md`.
- **Artefactos:** `results_vision/resnet18_attention_mamba_fastproof/`.

## 2026-08-21 — Confirmación interna con nested repeated CV

- **Pregunta:** ¿la señal de Mamba sobrevive a outer CV completa y reajuste
  simétrico de todos los modelos?
- **Protocolo:** 5 outer folds x 3 repeticiones, 3 inner folds, selección de
  época por mediana interna y reajuste con todo outer-train; 5,000 bootstraps
  agrupados por paciente.
- **Resultado OOF:** PCA+Cox 0.6432 ± 0.0264, attention 0.6957 ± 0.0242 y Mamba
  0.7030 ± 0.0194.
- **Comparación pareada:** Mamba - PCA+Cox +0.0598, IC95%
  [+0.0156, +0.1050], p=0.0100. Mamba - attention +0.0073, IC95%
  [-0.0136, +0.0284], p=0.5076.
- **Decisión:** conservar Mamba y attention como candidatos secuenciales. No
  afirmar que Mamba supera a attention; la evidencia sólida es a favor de la
  tubería axial secuencial frente al resumen oficial de tres vistas.
- **Siguiente prueba:** generar riesgos secuenciales cross-fitted dentro del
  evaluador trimodal y medir aporte incremental frente a la fusión convexa.
- **Detalle:** docs/resnet18_sequence_mamba_confirmatory_results.md.
- **Artefactos agregados:** results_vision/resnet18_sequence_nested_cv/.

## 2026-08-21 — Integración Mamba en fusión convexa

- **Pregunta:** ¿la mejora visual secuencial añade señal a la fusión trimodal?
- **Cohorte/protocolo:** mismos 210 pacientes y cinco hold-outs históricos;
  riesgos Mamba cross-fitted dentro del outer-train y reajuste separado para el
  held-out.
- **Resultado:** visión Mamba 0.7265 frente a ResNet 0.6397; fusión Mamba 0.8180
  frente a fusión ResNet 0.8111 y tabular 0.7990.
- **Estabilidad:** Mamba visual ganó 5/5; fusión Mamba frente a ResNet tuvo
  2 victorias, 1 empate y 2 derrotas. Ningún IC individual de esa comparación
  excluyó cero.
- **Decisión:** preferir Mamba como representación visual, pero mantener 0.8111
  como referencia formal de fusión. Registrar 0.8180 como candidata
  diagnóstica.
- **Detalle:** docs/trimodal_mamba_fusion_results.md.
- **Siguiente prueba local:** repeated outer CV de fusión para estabilidad de
  pesos. STU-Net y validación externa requieren cómputo/datos adicionales.

## 2026-08-21 — Outer repeated CV de fusión

- **Pregunta:** ¿la ganancia diagnóstica 0.8180 vs 0.8111 es estable?
- **Protocolo:** 5 outer folds x 3 repeticiones, Mamba cross-fitted de forma
  anidada y 5,000 bootstraps agrupados por paciente.
- **Resultado:** tabular 0.7892, visión ResNet 0.6327, visión Mamba 0.6774,
  fusión ResNet 0.7841 y fusión Mamba 0.7866.
- **Incertidumbre:** Mamba visual +0.0447, IC95% [-0.0076, +0.0995],
  p=0.0988. Fusión Mamba - ResNet +0.0025, IC95%
  [-0.0117, +0.0169], p=0.7328.
- **Corrección de decisión:** el 0.8180 queda estrictamente exploratorio. No se
  confirma ventaja de Mamba en fusión ni ventaja de la fusión sobre tabular.
  Mamba queda como candidata visual, no como baseline formal.
- **Pesos:** Mamba asignó en promedio 0.747 tabular, 0.080 texto y 0.173 visión.
- **Siguiente paso local:** estudiar el outlier, sensibilidad de tokens y
  calibración/regularización de pesos sin aumentar complejidad.
- **Detalle:** docs/trimodal_sequence_nested_cv_results.md.

## 2026-08-21 — Diagnóstico de estabilidad de época Mamba

- **Problema:** un outer fold cayó a C-index 0.3611 con 19 épocas.
- **Composición:** no se encontró una anomalía suficiente en CT/MR, número de
  tokens o geometría.
- **Inicialización:** diez seeds a 19 épocas dieron 0.4170 ± 0.0367; el problema
  persiste y no es sólo una seed.
- **Épocas:** con la seed original, 3 épocas dieron 0.5451 y 5 dieron 0.5313.
- **Sensibilidad global post hoc:** cap 5 mejoró el promedio por fold +0.0173,
  pero tuvo 7 victorias, 2 empates y 6 pérdidas.
- **Decisión:** conservar el outlier, no adoptar el cap con la misma cohorte y
  registrar cap 5 únicamente como hipótesis predeclarable.
- **Detalle:** docs/mamba_epoch_stability_diagnostic.md.

## 2026-08-21 — Ablación factorial de arquitectura, tokens y posición

- **Pregunta:** ¿el resultado secuencial depende de attention/Mamba, 32/64
  tokens o de coordenadas axiales explícitas?
- **Protocolo:** ocho configuraciones en los mismos 5 outer folds x 3
  repeticiones, 3 inner folds, reajuste completo y 5,000 bootstraps agrupados.
- **Resultado:** Mamba-64 sin posición encabezó la media global con 0.6868 y
  CT con 0.7076. Attention-32 sin posición quedó segundo con 0.6751. MR tuvo
  0.5676 para la primera, pero sólo hay 24 casos y 11 eventos.
- **Incertidumbre:** ninguno de 12 contrastes excluyó cero. Mamba vs attention
  a 64 sin posición fue +0.0130, IC95% [-0.0161, +0.0405]; 64 vs 32 para Mamba
  sin posición fue +0.0146, IC95% [-0.0076, +0.0365].
- **Decisión:** adoptar Mamba-64 sin posición como configuración operativa
  predeclarable, no como ganador confirmado. Mantener attention-32 sin posición
  como control compacto y no hacer afirmaciones sobre MR.
- **Relación con la fusión:** no revierte la decisión histórica. La
  concatenación tardía fue un diagnóstico del VAE y la concatenación cruda
  quedó desaconsejada por `p >> n`; la fusión convexa sigue como referencia.
- **Detalle:** docs/sequence_factorial_ablation_results.md.
- **Artefactos agregados:** results_vision/sequence_factorial_ablation/.

## 2026-08-21 — Mamba bidireccional con pesos compartidos

- **Pregunta:** ¿promediar recorridos axiales directo e inverso mejora
  Mamba-64 sin posición sin aumentar parámetros?
- **Protocolo:** misma semilla y splits de la ablación factorial; 5 outer folds
  x 3 repeticiones, 3 inner folds, reajuste completo y 5,000 bootstraps.
- **Control de reproducción:** la rama unidireccional reprodujo exactamente las
  predicciones Mamba-64 sin posición anteriores.
- **Resultado:** unidireccional 0.6868 y bidireccional 0.6731; delta -0.0136,
  IC95% [-0.0313, +0.0033], p=0.1080. Ambas tienen 319,041 parámetros.
- **Estabilidad:** la bidireccional ganó 8 folds, empató 2 y perdió 5, pero una
  pérdida de -0.1281 dominó sus ganancias pequeñas. El outlier usó menos épocas
  y tuvo validación interna similar, por lo que no se atribuye a una selección
  simplemente más larga.
- **Decisión:** rechazar la bidireccional como configuración operativa y
  conservar Mamba-64 sin posición unidireccional. No aumentar capacidad.
- **Detalle:** docs/mamba_bidirectional_ablation_results.md.
- **Artefactos agregados:** results_vision/mamba_bidirectional_ablation/.

## 2026-08-21 — Ensamble fijo Mamba + attention

- **Pregunta:** ¿un promedio 50/50 aprovecha errores complementarios de
  Mamba-64 y attention-32, ambos sin posición?
- **Escala:** el análisis principal promedió rangos percentiles por held-out
  fold; riesgo crudo y z-score fueron sensibilidades predefinidas.
- **Resultado:** rangos 0.6997 frente a Mamba 0.6868; delta +0.0129, IC95%
  [-0.0116, +0.0364], p=0.2952. Z-score llegó a 0.7040 y crudo a 0.6903.
- **Estabilidad:** rangos ganó 6 folds, empató 1 y perdió 8 frente a Mamba. La
  mejora agregada depende principalmente de una repetición.
- **Límite:** rangos/z-score usan la distribución no etiquetada del held-out y
  son transductivos; no constituyen una estimación desplegable.
- **Decisión:** no reemplazar Mamba ni seleccionar z-score post hoc. Conservar
  la complementariedad como hipótesis para una evaluación con escala derivada
  exclusivamente del outer-train.
- **Detalle:** docs/fixed_sequence_ensemble_results.md.
- **Artefactos agregados:** results_vision/fixed_sequence_ensemble/.

## 2026-08-21 — Ensamble con escala exclusiva de outer-train

- **Pregunta:** ¿la señal del ensamble post hoc persiste con una transformación
  aplicable sin usar la distribución held-out?
- **Protocolo:** reajuste nested de Mamba-64 y attention-32; ECDF y z-score
  estimados sólo con riesgos de outer-train; pesos fijos 0.5/0.5.
- **Reproducción:** ambos riesgos base coincidieron bit por bit con la ablación
  factorial.
- **Resultado:** percentil-train 0.7069 frente a Mamba 0.6868; delta +0.0201,
  IC95% [-0.0053, +0.0457], p=0.1256. Ganó 10/15 folds. Z-score-train obtuvo
  0.7042 y crudo 0.6903.
- **Subgrupos:** CT 0.7234 frente a 0.7076; MR 0.5484 frente a 0.5676, con sólo
  24 casos/11 eventos.
- **Decisión:** promover percentil-train 50/50 a mejor candidato visual interno,
  mantener Mamba como referencia simple y predeclarar el ensamble para una
  validación nueva. No afirmar superioridad frente a Mamba.
- **Detalle:** docs/train_scaled_sequence_ensemble_results.md.
- **Artefactos agregados:** results_vision/train_scaled_sequence_ensemble/.

## Lista para la revisión final

- [ ] Cada cifra agregada apunta a un CSV/JSON versionado; los artefactos a
  nivel paciente permanecen locales y excluidos de Git.
- [ ] Cohortes, eventos y reglas de intersección están explícitos.
- [ ] Todo scaler, PCA, selector y modelo se ajusta únicamente con train.
- [ ] Las comparaciones usan los mismos pacientes y splits o declaran por qué
  no son comparables.
- [ ] Se distingue exploración, confirmación y validación externa.
- [ ] Los cambios de decisión conservan la razón histórica y la evidencia que
  los motivó.
- [ ] La integración Mamba-fusión usa riesgos cross-fitted por outer split.
- [ ] Se revisan estabilidad, intervalos y no sólo el C-index promedio.
