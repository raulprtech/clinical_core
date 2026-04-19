# Reproducibility Guide

> Objetivo: reproducir los números del reporte cuatrimestral v3 de CLINICAL-CORE desde un clone limpio del repositorio. Este documento es el contrato operacional entre el experimento documentado y quien lo ejecute.

## 0. Números objetivo

Lo que debes ver al final de una corrida exitosa. Corrida de referencia: `20260419_011106_c5ecc7cd`.

| Componente | Métrica | Valor esperado | Tolerancia |
|---|---|---|---|
| Phase 1 (imputación) | C-index de KNN k=5 | 0.805 | ± 0.010 |
| Phase 2 (linear_compact) | C-index | 0.809 | ± 0.015 |
| Phase 2 (ft_transformer) | C-index | 0.790 | ± 0.020 |
| Phase 2 external (RSF) | C-index | 0.806 | ± 0.015 |
| Phase 5 trimodal | C-index | 0.785 | ± 0.010 |
| Phase 6 (VAE generativo) | Stage A epochs | 100 | exacto |
| Phase 6 d_latent | 128 | exacto |
| Phase 7 Hadamard INT4 | C-index | 0.829 | ± 0.008 |
| Phase 7 Hadamard INT3 | C-index | 0.827 | ± 0.008 |
| Phase 7 SVD FP32 rotado | C-index | 0.781 | ± 0.030 |
| Phase 8 Cox mediana | C-index | 0.827 | ± 0.008 |
| Phase 8 Weibull mediana | C-index | 0.819 | ± 0.015 |
| Phase 8 Weibull media | C-index | 0.796 | ± 0.040 (outlier) |

Si algún número cae fuera de tolerancia, revisa la sección **"Troubleshooting"** al final.

---

## 1. Prerrequisitos

### Hardware mínimo

- CPU x86-64 con 8+ GB de RAM. GPU opcional (CUDA acelera Phase 6 en ~2×).
- 3 GB de espacio en disco (repo + datos TCGA-KIRC + outputs).
- La corrida de referencia se ejecutó en WSL2 + Python 3.12.3 + PyTorch 2.11.0+cu130 + NumPy 2.4.4. Cualquier entorno equivalente en versiones debería producir los mismos números dentro de la tolerancia.

### Datos de entrada

El pipeline consume los XMLs clínicos de TCGA-KIRC del NCI GDC. Éstos **no se versionan en el repo** por tamaño y DUA. Debes tenerlos accesibles en una ruta local, típicamente `~/data/tcga-kirc-xml/`.

Para verificar que tu directorio tiene los XMLs correctos:

```bash
ls ~/data/tcga-kirc-xml/ | head -5
# Debe mostrar archivos como: nationwidechildrens.org_clinical.TCGA-XX-XXXX.xml
ls ~/data/tcga-kirc-xml/ | wc -l
# Debe devolver ~537
```

### Entorno Python

```bash
git clone https://github.com/raulprtech/clinical_core.git
cd clinical_core
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si `requirements.txt` no existe o es parcial, las dependencias críticas son:

```
torch>=2.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
lifelines>=0.27
pyyaml>=6.0
scikit-survival>=0.22  # para RSF baseline
tabpfn>=2.6            # opcional, Phase 2 external
```

---

## 2. Corrida de referencia (todas las fases)

Desde la raíz del repo:

```bash
cd code
python core/experiment_runner.py experiments/experiment_config.yaml
```

Tiempo aproximado en hardware moderno sin GPU: **~55 minutos**. Con GPU CUDA para Phase 6: **~50 minutos** (la mayor parte del tiempo se va en Phases 2 y 4, que son CPU-bound).

Al terminar, el comando imprime algo como:

```
EXPERIMENT COMPLETE in 3130.29s
Results: /home/<user>/clinical_core/results/20260XXX_XXXXXX_XXXXXXXX
Errors:  1
```

Un solo error esperado en esta versión: Phase 4 stress test reporta un directorio faltante. Es un bug conocido de infraestructura que no afecta los resultados principales — ver Troubleshooting.

---

## 3. Estructura del run directory

Cada corrida genera un directorio único con timestamp + hash del config. Su contenido:

```
results/20260419_011106_c5ecc7cd/
├── experiment_config.yaml      # copia literal del config usado
├── feature_config.yaml         # copia literal del schema de features
├── run_metadata.json           # timestamp, hash, versiones, componentes registrados
├── summary.json                # resultados agregados de todas las fases
├── raw_features.csv            # salida del extractor XML
├── raw_targets.csv             # targets de supervivencia
│
├── phase1_imputation.csv       # benchmark imputación
├── phase2_variants.csv         # por fold × seed
├── phase2_variants_summary.csv # agregado
├── phase2_external_baselines.csv
├── phase3_efficiency.csv
├── phase4_stress.csv           # (puede faltar por el bug conocido)
├── phase5_multimodal_ablation.csv
├── phase_6_fusion_proc.csv
├── phase_7_turbolatent.csv
├── phase_8_prognosis_benchmark.csv
│
└── artifacts/                  # artefactos persistidos para reutilización
    ├── phase_6_latent_z.npz    # Z, T, E, case_ids, train/val idx
    ├── phase_6_vae_checkpoint.pt
    └── phase_6_vae_history.json
```

Para verificar los números del reporte, el archivo clave es `summary.json`. Los otros CSVs permiten análisis más fino (por ejemplo, identificar qué fold produjo el outlier de Weibull seed 1024).

---

## 4. Corridas parciales (iteración rápida)

El runner soporta activar/desactivar fases vía YAML. Esto es el camino recomendado cuando ya tienes una corrida completa previa y solo quieres re-ejecutar parte del pipeline.

### Ejemplo: re-correr solo TurboLatent con distintos bit widths

Edita `code/experiments/experiment_config.yaml` y pon `enabled: false` en `phase_1_imputation`, `phase_2_variants`, `phase_2_external_baselines`, `phase_3_efficiency`, `phase_4_stress`, `phase_5_multimodal`, `phase_6_fusion_proc`, `phase_8_prognosis_benchmark`. Deja `enabled: true` solo en `phase_7_turbolatent` y ajusta `bit_widths: [2, 3, 4, 6, 8, 10, 12]`.

Corre el runner normalmente. Phase 7 buscará automáticamente el último artefacto `phase_6_latent_z.npz` de una corrida anterior mediante la función `resolve_artifact_path`. **No necesitas re-entrenar el VAE** — se reutiliza el Z ya producido.

Tiempo estimado: ~60 segundos (frente a 55 min de la corrida completa).

### Ejemplo: probar un nuevo PROGNOSIS-PROC sobre el mismo Z

1. Implementa tu nuevo procesador y regístralo (ver `components_guide.md`).
2. Añádelo a la lista `models:` de `phase_8_prognosis_benchmark`.
3. Desactiva Phases 1-7, deja solo Phase 8 activa.
4. Corre el runner. Phase 8 resolverá el Z artefacto de la corrida previa y evaluará Cox + Weibull + tu nuevo procesador lado a lado.

### Punteando a un artefacto específico

Si tienes varias corridas y quieres consumir una específica (por ejemplo, el Z producido con una configuración del VAE distinta), setea la ruta explícita en el YAML:

```yaml
phase_7_turbolatent:
  enabled: true
  source_artifact_path: "/home/user/clinical_core/results/20260419_011106_c5ecc7cd/artifacts/phase_6_latent_z.npz"
```

El mecanismo de resolución sigue esta precedencia:

1. Artefacto en el run_dir actual (si Phase 6 corrió en la misma sesión).
2. `source_artifact_path` explícito del YAML (si lo especificas).
3. Artefacto más reciente en los últimos 10 runs del `output.base_dir`.

---

## 5. Verificación numérica

Después de una corrida, para verificar que los números coinciden con el reporte:

```bash
cd results/<tu-run-id>
python -c "
import json
s = json.load(open('summary.json'))

# Phase 6: VAE
p6 = s['phases']['phase_6'][0]
assert p6['d_latent'] == 128, f\"d_latent={p6['d_latent']}\"
assert p6['stage_a_epochs'] == 100

# Phase 7: Hadamard INT4 debe igualar baseline
p7 = {(r['variant'], r['bits']): r for r in s['phases']['phase_7']}
baseline = p7[('baseline', 'fp32')]['cindex_mean']
had_int4 = p7[('hadamard', 4)]['cindex_mean']
delta = abs(baseline - had_int4)
print(f'Baseline FP32: {baseline:.4f}')
print(f'Hadamard INT4: {had_int4:.4f}')
print(f'Delta: {delta:.4f} (tolerancia: 0.008)')
assert delta < 0.008, 'Hadamard INT4 deberia estar dentro de 0.008 del baseline'

# Phase 8: Cox estable, Weibull con outlier
import numpy as np
p8 = s['phases']['phase_8']
cox = np.array([r['cindex_mean_folds'] for r in p8 if r['model'].endswith('cox')])
wei = np.array([r['cindex_mean_folds'] for r in p8 if r['model'].endswith('weibull_head')])
print(f'Cox: mean={cox.mean():.4f}, std={cox.std():.4f}')
print(f'Weibull: mean={wei.mean():.4f}, std={wei.std():.4f}, median={np.median(wei):.4f}')
print('\\nTodos los checks pasaron.')
"
```

Si algún assert falla, ejecuta la sección de Troubleshooting.

---

## 6. Troubleshooting

### "Phase 4 stress error: directory does not exist"

**Síntoma:** `summary.json` tiene una entrada en `errors` mencionando `_stress_clean` no existe.

**Causa:** bug conocido en Phase 4. El directorio que la fase intenta escribir no se crea antes.

**Impacto:** Ninguno sobre los números principales. Phase 4 solo mide degradación bajo ruido sintético, no afecta Phase 2/5/6/7/8.

**Fix temporal:** desactivar Phase 4 (`phase_4_stress.enabled: false`) en el YAML hasta que se resuelva.

### "tabpfn_external failed: unexpected keyword argument 'token'"

**Síntoma:** `phase_2_external_baselines.csv` tiene `cindex = NaN` para `tabpfn_external`.

**Causa:** cambio de API en TabPFN ≥ 2.6. El wrapper de `tabpfn_external.py` usa un kwarg obsoleto.

**Impacto:** Solo se pierde el baseline TabPFN. El baseline RSF sigue funcionando.

**Fix temporal:** desactivar el baseline TabPFN:

```yaml
phase_2_external_baselines:
  baselines:
    tabpfn_external:
      enabled: false
```

### C-index de Cox sobre Z muy por debajo de 0.82 (~0.71)

**Síntoma:** Phase 8 Cox reporta ~0.71 en lugar de ~0.83.

**Causa:** Phase 6 está usando la proyección random aleatoria en lugar del encoder `linear_compact` entrenado. Esta fue la versión inicial que se corrigió con la Opción A.

**Verificación:** busca en el log de Phase 6 la línea `Training linear_compact embedder (19 → 768)...`. Si no aparece, estás corriendo la versión vieja.

**Fix:** actualiza `code/core/experiment_runner.py` a la versión que entrena el encoder antes del VAE. El commit correcto debe incluir el bloque:

```python
encoder = VariantC_LinearEncoder(input_dim=X_tab.shape[1], ...)
train_variant_c(encoder, X_tab_t[enc_train_idx], ...)
```

en la función `phase_6_fusion_proc`.

### Weibull seed 1024 da ~0.71 en lugar de ~0.82

**Síntoma:** 4 semillas de Weibull dan 0.81-0.82 pero seed 1024 cae a 0.71.

**Causa:** comportamiento esperado y documentado en el reporte §6. La parametrización Weibull es sensible a la partición de validación; ciertas particiones inicializan los heads en regiones subóptimas.

**No es un bug.** La mediana por semilla (0.819) es la métrica robusta que debe reportarse.

### Números levemente distintos pero dentro de tolerancia

**Síntoma:** tu Phase 7 Hadamard INT4 da 0.826 en lugar de 0.829.

**Causa probable:** diferencias de versión en PyTorch o NumPy producen variaciones de orden 10⁻³ en el C-index por cambios en el orden de operaciones aritméticas de punto flotante.

**Acción:** Verifica que tus versiones son compatibles con las de referencia (ver prerrequisitos). Si están dentro de la tolerancia tabulada en §0, el número es aceptable.

### El runner crashea con `ModuleNotFoundError: components.processors.fusion.models.vae_generative`

**Causa:** el módulo del VAE generativo no está presente o el `__init__.py` no lo expone.

**Fix:** verifica que existe `components/processors/fusion/models/vae_generative.py` con la clase `FusionVAEGenerative` y la config `VAEGenTrainConfig`. Si falta, el componente no fue pusheado en el commit del sprint de FUSION-PROC — revisa con `git log` los commits recientes.

---

## 7. Preguntas que no resuelve este documento

Si después de seguir esta guía los números siguen fuera de tolerancia y Troubleshooting no aplica:

- Guarda `summary.json`, `run_metadata.json`, y el log completo del runner.
- Compara `run_metadata.json.environment` con las versiones de referencia (§1).
- Abre un issue en el repositorio adjuntando esos tres archivos.

Los números del reporte v3 provienen del commit que puede identificarse como `git log --grep "phase_6_option_a"` en el repositorio. Si tu checkout es anterior, verifica en qué commit estás con `git log -1 --format='%H %s'`.

---

## 8. Extensión a corridas de producción

Este documento cubre reproducción de los números del reporte. Para producir una corrida con nueva configuración (nuevo hiperparámetro, nueva semilla, nuevo componente), sigue en cambio `components_guide.md` para el protocolo de extensión. Este documento es **solo** sobre reproducción.