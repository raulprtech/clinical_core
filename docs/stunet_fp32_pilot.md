# Piloto técnico STU-Net-S congelada

## Objetivo

Validar procesamiento automático por cohorte, ROI reproducible, embeddings,
reanudación, memoria, tiempos y repetibilidad antes de implementar PTQ o
TurboConv. Esta etapa no utiliza desenlaces ni particiones de supervivencia.

## Definición congelada del embedding

El checkpoint oficial STU-Net-S produce un bottleneck de 256 canales. El
adaptador construye el vector canónico de 768 dimensiones sin una proyección
aprendida:

1. pooling del bottleneck dentro del bounding box renal con margen de 30 mm;
2. pooling dentro de la máscara de riñón izquierdo (etiqueta 38);
3. pooling dentro de la máscara de riñón derecho (etiqueta 39);
4. concatenación de los tres vectores 256D y normalización L2.

Las máscaras se reducen al bottleneck mediante pooling fraccional. PTQ y
TurboConv deben reutilizar exactamente esta función para que la deriva de
embeddings sea pareada.

## Inferencia de baja memoria

El modelo conserva sus pesos FP32 y usa AMP en el piloto. Las probabilidades
de las 105 clases se acumulan en un `memmap` FP16 temporal y el argmax se
calcula por bloques. Así, el tamaño del volumen incrementa el uso de disco,
pero no obliga a mantener todo el tensor de probabilidades en RAM.

Antes de inferencia se comprueba la geometría DICOM usando
`ImagePositionPatient` e `ImageOrientationPatient`. Series con posiciones
duplicadas, huecos mayores a 1.5 veces la mediana o pasos menores a la mitad
se registran como fallos y no producen embeddings.

## Ejecución resumible

```bash
.venv/bin/python -u code/tools/build_stunet_embeddings.py \
  --limit 10 \
  --precision amp \
  --output-root data/embeddings/vision/stunet_fp32_pilot
```

Cada paciente solo se considera completo cuando existen la máscara y
`cases/<CASE_ID>/complete.json`. Una nueva ejecución omite esos casos. Los
archivos agregados son:

- `stunet_s_fp32_embeddings_768.csv`;
- `metrics.csv`;
- `failures.csv`;
- `pilot_cohort.csv`;
- `provenance.json`;
- máscaras NIfTI por paciente.

## Resultado del piloto de diez CT

- 10/10 casos QC-pass procesados;
- embeddings: `10 x 768`, finitos y norma L2 aproximadamente 1;
- 9 casos con ambos riñones y uno con una sola máscara renal;
- tiempo por caso: 73.6–257.9 s, mediana 186.2 s;
- RSS máximo: 3.62 GiB;
- memoria CUDA máxima asignada: 2.18 GiB;
- acumulador temporal máximo: 6.84 GiB en disco;
- reanudación completa: 4.3 s sin inferencia;
- repetición independiente de un caso:
  - coseno de embeddings: 0.99999975;
  - Dice izquierdo/derecho: 0.99978 / 0.99936;
  - acuerdo de etiquetas global: 99.981 %.

El preflight sobre las 190 series CT seleccionadas encontró 76 uniformes y
114 con problemas geométricos según estos criterios. La comparación final con
ResNet debe reconstruirse sobre la intersección CT que supere el QC y produzca
embeddings válidos en todas las variantes.
