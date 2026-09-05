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

## 2026-08-22 — CT multi-window 2.5D

- **Pregunta:** ¿tres ventanas HU fijas mejoran los tokens sin pasar a 3D?
- **Intervención:** `[-150,250]`, `[-73,304]` y `[-200,500]`; ResNet18
  congelada, media equiponderada de features 512D y MRI sin cambio material.
- **Auditoría:** 214 pacientes/13,526 tokens; mismas series, longitudes y
  posiciones; 0 fallos. Coseno CT medio 0.9908.
- **Protocolo:** mismos 5 folds x 3 repeticiones, 3 inner folds y seeds del
  ensamble percentil-train.
- **Resultado primario:** ensamble 0.7139 vs 0.7069 single-window; +0.0069,
  IC95% [-0.0037,+0.0182], p=0.2052. CT +0.0073.
- **Secundario:** attention +0.0129, IC95% [+0.0018,+0.0241], p=0.0228;
  comparación no primaria, múltiples lecturas y magnitud menor a +0.02.
- **Decisión:** no promover multi-window ni ajustar ventanas post hoc. Mantener
  el ensamble single-window como referencia y pasar al encoder médico 2D.
- **Detalle:** `docs/resnet18_multiwindow3_results.md`.
- **Artefactos:** `results_vision/resnet18_multiwindow3_train_scaled_ensemble/`
  y `results_vision/resnet18_multiwindow3_vs_single/`.

## 2026-08-22 — ResNet50 RadImageNet 2.5D

- **Pregunta:** ¿el preentrenamiento radiológico mejora la representación
  secuencial antes de pasar a contexto 3D?
- **Control:** ResNet50 ImageNet frente a ResNet50 RadImageNet; misma cohorte,
  tokens, proyección fija 2048->512, poolers, folds, seeds y evaluación.
- **Auditoría:** 214 pacientes/13,526 tokens por encoder, mismas series,
  longitudes y posiciones; cero fallos.
- **Resultado primario:** ResNet18 0.7069, ImageNet50 0.6719 y RadImageNet50
  0.6903. Rad-ImageNet50 +0.0183, IC95% [-0.0405,+0.0746], p=0.5372;
  Rad-ResNet18 -0.0167, IC95% [-0.0666,+0.0316], p=0.4972.
- **Subgrupos:** RadImageNet50 CT 0.7193, prácticamente igual a ResNet18
  0.7234; MR 0.4707 con sólo 24 casos/11 eventos.
- **Decisión:** no promover ResNet50 ni ajustar el encoder post hoc. Mantener
  ResNet18 single-window y cerrar la búsqueda de encoders 2D congelados grandes.
- **Detalle:** `docs/resnet50_radimagenet_results.md`.
- **Artefactos:** `results_vision/resnet50_imagenet_train_scaled_ensemble/`,
  `results_vision/resnet50_radimagenet_train_scaled_ensemble/` y comparadores
  `results_vision/resnet50_*_vs_*/`.

## 2026-08-21 — Auditoría externa MMIST/CPTAC-CCRCC

- **Pregunta:** ¿MMIST-ccRCC permite una validación externa independiente del
  candidato visual entrenado con TCGA-KIRC?
- **Enlace:** los 189 casos CPTAC de MMIST correspondieron con GDC CPTAC-3;
  167 tuvieron OS derivable y 31 eventos.
- **Intersección visual:** CT quedó en 50 casos evaluables y 4 eventos; MRI en
  5 casos y 0 eventos. CT y MRI sólo se solapan en un paciente.
- **Corrección del endpoint:** el target MMIST a 12 meses ocultaba tres muertes
  CT posteriores a ese horizonte. GDC permite construir tiempo-a-evento, pero
  cuatro eventos siguen siendo insuficientes para confirmación.
- **Inventario TCIA posterior:** la cohorte CT conocida subió a 54 casos, 53
  endpoints válidos y los mismos 4 eventos; no aumentó la potencia.
- **Decisión:** excluir todo caso TCGA del test; conservar CPTAC CT sólo como
  prueba exploratoria de transporte y no descargar DICOM con la evidencia actual.
- **Detalle:** `docs/mmist_cptac_external_validation_audit.md`.
- **Auditores:** `code/tools/audit_mmist_cptac_survival.py` y
  `code/tools/audit_tcia_cptac_inventory.py`.
- **Agregados:** `results_external/mmist_cptac_survival_audit/`.

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

## 2026-08-27 — Pooling volumétrico STU-Net y fusión trimodal

- **Pregunta:** ¿conservar dispersión volumétrica del bottleneck renal mejora
  STU-Net y su aporte multimodal frente a tres medias regionales?
- **Representación:** `renal_moments_512` = media256 + desviación estándar256
  en ROI renal; pesos STU-Net-S congelados.
- **Protocolo visual:** 75 casos/20 eventos, 5 folds x 5 repeticiones, nested
  PCA+ridge Cox.
- **Resultado visual inicial:** 0.6911 vs 0.5379 por rangos OOF; delta +0.1531,
  IC95% [+0.0250,+0.2922].
- **Protocolo trimodal:** intersección de 72 casos/20 eventos; mismos 5x5 outer
  folds, 3 inner folds, riesgos cross-fitted y pesos convexos train-only.
- **Resultado trimodal:** visión 0.7531 vs 0.5734; +0.1797, IC95%
  [+0.0747,+0.2800]. Fusión 0.7825 vs 0.7567; +0.0258, IC95%
  [-0.0060,+0.0599]. Fusión moments vs tabular +0.0061, IC95%
  [-0.0372,+0.0540].
- **Decisión:** promover `renal_moments_512` como representación visual, pero no
  afirmar mejora multimodal ni superioridad sobre tabular.
- **Ablación bimodal predeclarada:** tabular+moments obtuvo 0.7890 frente a
  0.7825 trimodal; +0.0065, IC95% [-0.0075,+0.0209]. Ganó 4 folds, empató 20 y
  perdió 1. Se acepta como simplificación parsimoniosa sin superioridad confirmada.
- **Detalle:** `docs/stunet_volumetric_pooling_pilot.md` y
  `docs/stunet_trimodal_pooling_results.md`.
- **Artefactos agregados:** `results_vision/stunet_volumetric_pooling_nested_75/`
  `results_vision/stunet_trimodal_pooling_nested_72/` y
  `results_vision/stunet_bimodal_moments_nested_72/`.

## 2026-08-27 — Separación del contexto ResNet18 2.5D

- **Pregunta predeclarada:** ¿separar los canales vecinos `±2` o `±4` cortes
  mejora Mamba frente al contexto adyacente `±1`?
- **Control:** mismas 214 series, 64 eventos, ResNet18 congelada, ventana renal,
  64 tokens, Mamba-64 sin posición, splits y semillas.
- **Reproducción:** las 642 predicciones OOF de `span1` coincidieron bit por bit
  con la ablación factorial histórica.
- **Resultado:** `span1` 0.6868, `span2` 0.6536 y `span4` 0.6466.
  `span2-span1` -0.0332, IC95% [-0.0675,+0.0006], p=0.0548;
  `span4-span1` -0.0402, IC95% [-0.0840,+0.0020], p=0.0632.
- **Estabilidad:** `span2` ganó 5/15 folds y `span4` 6/15; la separación amplia
  aumentó la DE entre repeticiones a 0.0448.
- **Decisión:** conservar `[-1,0,+1]` y cerrar variaciones adicionales de
  separación 2.5D en esta cohorte. No afirmar daño confirmatorio porque ambos
  intervalos rozan cero.
- **Detalle:** `docs/resnet18_2p5d_context_span_protocol.md` y
  `docs/resnet18_2p5d_context_span_results.md`.
- **Artefactos agregados:**
  `results_vision/resnet18_2p5d_context_span_nested/`.

## 2026-09-04/05 — Programa renal 2.5D, adaptación y radiomics

- **Alcance autorizado:** ejecutar recorte anatómico, adaptación ligera de
  ResNet18 y radiomics2D; registrar y probar seguimientos motivados por resultados.
- **Cohorte inicial:** 75 CT/20 eventos con máscaras reales de STU-Net ya
  disponibles. Geometría/hashes y centros pareados verificados. Siete máscaras
  tienen un único riñón; no asumir lateralidad tumoral ni nefrectomía.
- **Recorte:** Mamba campo completo0.8187, tramo renal sin recorte0.8242,
  recorte0.6990. Recorte vs mismos centros sin recorte: -0.1252,
  IC95% [-0.2208,-0.0281], Holm0.0372. Conservar contexto; no promover recorte.
- **Radiomics:** 27 medidas explícitas, sin mocks; 0.6655 con Cox.
  Fusión tabular+radiomics0.7372 vs tabular0.7319, delta+0.0053,
  IC95% [-0.0599,+0.1066]. No ventaja incremental demostrada.
- **Adaptación inicial:** último bloque adaptado0.7161 vs congelado0.6799,
  delta+0.0362, IC95% [-0.0207,+0.0958]. Selección en máximo5 épocas en
  13/15 y15/15 folds: seguimiento predeclarado hasta20 épocas.
- **Adaptación extendida F5:** adaptado0.7545 vs congelado0.7474,
  delta+0.0071, IC95% [-0.0460,+0.0778], p0.7368. La ventaja inicial se reduce
  al permitir entrenar más al control; no promover fine-tuning por esta cifra.
  Máximo20 seleccionado en11/15 y8/15: convergencia no establecida.
- **Oportunidad F6:** repetir la comparación adaptado-congelado con campo
  completo, dado el daño del recorte observado en E2. Protocolo registrado
  antes de entrenamiento, mismos75 casos y cuadrícula1/3/5/10/20.
- **F6 completado:** adaptado 0.7933 vs congelado 0.7443, delta +0.0490,
  IC95% [+0.0027,+0.1085], p0.0396. Señal exploratoria favorable; ambos brazos
  eligen máximo20 en12/15 folds, sin convergencia establecida.
- **D1 post hoc:** adaptado frente a Mamba de campo completo 0.8187:
  delta -0.0254, IC95% [-0.1004,+0.0487]; frente a Cox 0.7829:
  delta +0.0104, IC95% [-0.0705,+0.1021]. No demuestra superioridad.
- **F7, oportunidad de inspección visual:** el usuario autorizó revisar cuatro
  ejemplos locales. El recorte se estiraba a224x224. Padding cuadrado sin
  deformación obtiene 0.7366 vs recorte0.6990, delta +0.0376,
  IC95% [-0.0537,+0.1385], Holm0.4256. No supera campo completo ni demuestra
  mejora; las máscaras siguen sin validación tumoral experta.
- **Momentos ResNet:** mejora media pequeña en75 que no se sostiene al ampliar
  a214/64 eventos: media0.6771 vs momentos0.6515, delta-0.0256,
  IC95% [-0.0586,+0.0057]. Fusión con momentos tampoco supera fusión con media.
- **Interpretación:** cifras de esta etapa son medias de C-index dentro de fold;
  no compararlas directamente con OOF pooled histórico ni entre cohortes75/214.
  Todos los seguimientos son exploratorios, con modelos/preprocesamiento en train,
  CV5x3, bootstrap por paciente y multiplicidad explícita dentro de corrida.
- **Registro completo:** `docs/renal_2p5d_research_program.md` y
  `docs/renal_2p5d_program_results.md`; incluye protocolos, revisiones y artefactos.
- **Decisión final:** conservar contexto completo y los defaults actuales.
  Adaptación full queda como candidato exploratorio, no reemplazo validado.
  No promover radiomics, recorte o momentos por esta búsqueda. No afirmar
  agotamiento de todo2D/2.5D ni mejora garantizada por3D.
- **Estado:** diez experimentos y D1 completos y verificados; trece pruebas
  pasan. Código, informes y agregados preparados para GitHub; datos clínicos
  por paciente e imágenes excluidos. Detalles de redondeo CSV en el informe.

## 2026-09-05 — Segunda etapa 2.5D

- **Alcance:** continuar las tres opciones prioritarias y el piloto DINOv2,
  según `docs/renal_2p5d_stage2_protocol.md`, registrado antes de entrenamiento.
- **S1/S2:** parada interna hasta100 épocas, mínimo20 y paciencia15;
  cabezas lineal/Mamba, controles congelado/adaptado y17 pruebas técnicas.
  Entrenamientos en ejecución; no se dispone aún de conclusión global.
- **S3 global-local:**0.7982 vs full0.7818, delta+0.0164,
  IC95%[-0.0461,+0.0673], p0.622. No ventaja incremental demostrada.
- **S4 DINOv2:**0.7915 vs ResNet0.7951 sobre las mismas16 imágenes/caso,
  delta-0.0035, IC95%[-0.1302,+0.1126], p0.8964. No ventaja demostrada;
  no afirmar equivalencia ni extrapolar a todos los encoders autosupervisados.
- **Detalle:** `docs/renal_2p5d_stage2_results.md`, con fuentes oficiales,
  revisiones, límites y comandos de reproducción.
