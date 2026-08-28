# Protocolo predeclarado: separación del contexto 2.5D

Fecha de predeclaración: 2026-08-27, antes de extraer las variantes nuevas y
antes de consultar sus resultados de supervivencia.

## Pregunta

¿El contexto entre canales de cada token 2.5D es demasiado estrecho cuando se
usan cortes inmediatamente adyacentes? La referencia actual codifica cada token
axial como `[-1, 0, +1]`. Se comparará contra dos separaciones simétricas:

- referencia: `[-1, 0, +1]`;
- contexto intermedio: `[-2, 0, +2]`;
- contexto amplio: `[-4, 0, +4]`.

Los desplazamientos se expresan en índices de corte. Por tanto, esta ablación
evalúa una regla 2.5D pragmática y no equivale a una distancia física constante
entre escáneres. Esta limitación se conservará en la interpretación.

## Diseño fijo

- Mismas series TCIA-KIRC, desenlaces y cohorte común que la ablación secuencial.
- ResNet18 ImageNet1K V1 congelada, ventana renal `[-150, 250]`, imagen 224 px.
- Máximo de 64 tokens axiales uniformes, conservando extremos.
- Único modelo de supervivencia: Mamba-64 unidireccional, dos bloques, sin
  posición explícita; no se volverá a seleccionar arquitectura o capacidad.
- Outer CV de 5 folds por 3 repeticiones, inner CV de 3 folds para seleccionar
  época y reajuste sobre todo el outer-train.
- Particiones compartidas entre las tres separaciones.
- Contrastes principales pareados: `span2 - span1` y `span4 - span1`.
- Incertidumbre mediante 5,000 bootstraps agrupados por paciente.

## Regla de decisión

Una variante sólo reemplazará la referencia si su delta global tiene límite
inferior del IC95% mayor que cero. Las medias, el análisis CT/MR y el contraste
`span4 - span2` serán diagnósticos. Si ambos contrastes principales incluyen
cero, se cerrará la búsqueda local de separaciones 2.5D; no se elegirá la mayor
media de forma post hoc.

Esta es una ablación interna exploratoria sobre una cohorte reutilizada. Aun si
una variante supera el umbral, necesitará validación externa antes de una
afirmación confirmatoria.
