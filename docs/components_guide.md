# Developer Guide: Adding Components

> Protocolo para extender CLINICAL-CORE con nuevos componentes conformes al estándar. Cubre los tres tipos de extensión más comunes: un nuevo adaptador de ingestión, un nuevo procesador (fusión o pronóstico), y una nueva fase del runner.

---

## 1. El contrato al que tu componente debe conformarse

Cada tipo de componente satisface un contrato distinto. Implementar tu componente es equivalente a cumplir las garantías del contrato correspondiente.

### Contrato de ingestión (adaptadores IN)

Un adaptador IN recibe datos crudos de una modalidad y produce un embedding de dimensión fija con una métrica de confianza. Las garantías que debe cumplir son:

- **Dimensionalidad:** el embedding tiene exactamente `output_dim` dimensiones (canónicamente 768). La responsabilidad de la proyección al tamaño objetivo es del adaptador, nunca del fusionador.
- **Normalización:** los embeddings están L2-normalizados antes de ser devueltos.
- **Confianza:** cada caso produce un score en `[0.0, 1.0]` que refleja qué tan completos o confiables son los datos de entrada. Cero cuando los datos están ausentes; uno cuando todos los campos están presentes y pasan validaciones.

La interfaz en Python:

```python
class MyTabularAdapter:
    name = "my_tabular_adapter"

    def __init__(self, input_dim: int, output_dim: int = 768, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # ... arquitectura interna

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        # x: (batch, input_dim) — datos preprocesados
        # mask: (batch, input_dim) — 1 si presente, 0 si imputado
        # Retorna: (embedding [batch, output_dim], confidence [batch, 1])
        ...
```

Para datos no tabulares (imagen, texto), la signatura cambia pero el contrato de salida se mantiene idéntico.

### Contrato de procesamiento (procesadores PROC)

Un procesador PROC consume embeddings de uno o más adaptadores y produce una salida discriminativa. Hay dos roles canónicos:

**FUSION-PROC** integra embeddings de múltiples modalidades en un espacio latente compartido. Las garantías:

- **Dimensionalidad fija** en su salida (el valor exacto es configurable pero debe ser determinístico).
- **Confianza compuesta** que agrega las confianzas de los adaptadores fuente.
- **Robustez ante modalidades ausentes:** si una modalidad falta (confianza = 0 en sus embeddings), el procesador debe producir salida con confianza reducida pero sin crashear.

**PROGNOSIS-PROC** consume el output de FUSION-PROC y produce un score de supervivencia o clasificación. Las garantías:

- **Discriminatividad** verificable empíricamente (C-index > 0.5 sobre validación).
- **Confianza calibrada** en la salida (varianza del intervalo predictivo, no solo valor puntual).
- **Acoplamiento débil** con el espacio latente: debe ser entrenable sobre Z congelado sin modificar el fusionador.

### Contrato de explicabilidad (EXPLAIN)

Post-predicción, un componente de explicabilidad produce justificaciones legibles de la salida del core. Pendiente de implementación en el cuatrimestre actual. La especificación está en el protocolo v12 §7.2.

### Contrato de canal (OUT)

Los adaptadores de canal (OUT) comunican los resultados del core al mundo exterior (reporte clínico, coordenadas quirúrgicas, export a EHR). Pendiente de implementación; ver protocolo v12 §7.2.

### Contrato de monitoreo (MONITOR)

Componentes de observación longitudinal que operan sobre distribuciones agregadas (drift, fairness, audit). Pendientes; ver protocolo v12 §7.2.

---

## 2. Añadir un adaptador de ingestión

Ejemplo: añadir una nueva variante de `TABULAR-IN`.

### Paso 1 — Implementa el componente

Crea `code/components/adapters/ingestion/tabular/models/my_variant.py`:

```python
import torch
import torch.nn as nn

class MyVariantEncoder(nn.Module):
    name = "my_variant"

    def __init__(self, input_dim: int, output_dim: int = 768, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        embedding = self.net(x)
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        confidence = mask.float().mean(dim=1, keepdim=True)
        return embedding, confidence
```

### Paso 2 — Regístralo

Edita `code/core/registry.py`:

```python
from components.adapters.ingestion.tabular.models.my_variant import MyVariantEncoder

VARIANT_REGISTRY = {
    'cox_baseline': ...,
    'linear_compact': ...,
    'ft_transformer': ...,
    'my_variant': lambda input_dim, output_dim, **kw: MyVariantEncoder(
        input_dim=input_dim, output_dim=output_dim, **kw
    ),
}
```

### Paso 3 — Añádelo al YAML

Edita `code/experiments/experiment_config.yaml`:

```yaml
phase_2_variants:
  variants:
    - cox_baseline
    - linear_compact
    - ft_transformer
    - my_variant      # ← nuevo
  variant_params:
    my_variant:
      epochs: 200
      lr: 1.0e-3
      patience: 20
```

### Paso 4 — Verifica

Corre el runner con solo Phase 2 activa. Tu variante aparecerá en `phase2_variants_summary.csv` con su C-index y la columna `contract_satisfied`. Si `contract_satisfied = False`, revisa:

- La dimensionalidad del embedding (debe ser `output_dim`).
- La normalización L2.
- Que la función `verify_ingestion_contract` reconozca el output.

---

## 3. Añadir un procesador de pronóstico

Ejemplo: añadir un PROGNOSIS-PROC basado en Weibull discreto.

### Paso 1 — Implementa

Crea `code/components/processors/prognosis/my_weibull_discrete.py`:

```python
import torch
import torch.nn as nn

class PrognosisWeibullDiscrete(nn.Module):
    name = "prognosis_weibull_discrete"

    def __init__(self, fused_dim: int, **kwargs):
        super().__init__()
        self.shape_head = nn.Linear(fused_dim, 1)
        self.scale_head = nn.Linear(fused_dim, 1)
        # inicialización del reporte: scale directo, no log
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, 1000.0)

    def fit(self, Z_tr, T_tr, E_tr, Z_va, T_va, E_va,
            epochs=200, patience=20, verbose=False, **kwargs):
        # training loop... ver weibull_head.py como referencia
        return {'best_val_cindex': ci_value, 'risk_head': self}

    def predict_risk(self, Z):
        return -(self.scale_head(Z)).squeeze(-1)  # mayor riesgo = menor scale
```

### Paso 2 — Regístralo

```python
# code/core/registry.py
from components.processors.prognosis.my_weibull_discrete import PrognosisWeibullDiscrete

PROGNOSIS_PROC_REGISTRY = {
    'prognosis_baseline_linear_cox': ...,
    'prognosis_weibull_head': ...,
    'prognosis_weibull_discrete': lambda fused_dim, **kw: PrognosisWeibullDiscrete(
        fused_dim=fused_dim, **kw
    ),
}
```

### Paso 3 — Añádelo al benchmark de Phase 8

```yaml
phase_8_prognosis_benchmark:
  enabled: true
  models:
    - name: "prognosis_baseline_linear_cox"
    - name: "prognosis_weibull_head"
      params:
        init_scale: 1000.0
    - name: "prognosis_weibull_discrete"   # ← nuevo
      params:
        epochs: 200
```

### Paso 4 — Compara con baselines

Corre el runner con solo Phase 8 activa. Phase 8 resuelve el Z de la corrida previa de Phase 6 y evalúa tu procesador lado a lado con Cox y Weibull continuos. Los resultados caen en `phase_8_prognosis_benchmark.csv` con una fila por (modelo × semilla).

---

## 4. Añadir un procesador de fusión

El patrón es idéntico al de pronóstico pero con signatura distinta. Ver `fusion_vae_generative.py` como referencia completa. Puntos críticos:

- El constructor recibe `modalities` (lista de nombres) y `modality_dims` (dict con dim por modalidad).
- El método `fit` recibe `X_train`, `conf_train`, `T_train`, `E_train` y sus contrapartes de validación.
- El método `extract_latent_space(X, conf)` debe existir y devolver `(Z, conf_fused)` para alimentar al PROGNOSIS-PROC downstream.

Para comparar tu fusionador con los existentes, el patrón correcto es ejecutar Phase 6 con tu componente, persistir el Z como artefacto, y luego correr Phase 7/8 con el mismo `source_artifact_path` para cada fusionador. De esa forma la comparación es controlada: mismo predictor downstream, misma cuantización, única variable el fusionador.

---

## 5. Añadir una fase nueva al runner

Cuando quieras introducir un experimento completamente nuevo (por ejemplo, una fase de validación clínica A/B de explicaciones), el patrón es:

### Signatura de la fase

```python
def phase_N_my_experiment(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
    # opcional: inputs adicionales desde fases anteriores
    best_imputation: str = None,
) -> Optional[pd.DataFrame]:
    phase_cfg = config.get('phase_N_my_experiment', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE N] DISABLED")
        return None

    log("\n[PHASE N] My experiment")

    # Si necesitas artefactos de fases anteriores:
    artifact_path = resolve_artifact_path(
        artifact_name='phase_6_latent_z.npz',
        current_run_dir=run_dir,
        phase_cfg=phase_cfg,
        output_base_dir=Path(config['output']['base_dir']),
    )
    if artifact_path is None:
        log("[PHASE N] SKIPPED — no upstream artifact")
        return None

    # ... experimento ...

    results_df = pd.DataFrame(rows)
    results_df.to_csv(run_dir / "phase_N_my_experiment.csv", index=False)
    return results_df
```

### Registro en el orquestador

Al final de `run_experiment()`, después del último `try:` existente y antes del `# ---- Final summary ----`:

```python
try:
    phN = phase_N_my_experiment(df_features, df_targets, config, run_dir)
    if phN is not None:
        summary['phases']['phase_N'] = phN.to_dict(orient='records')
except Exception as e:
    summary['errors'].append({'phase': 'N', 'error': str(e)})
    if fail_fast: raise
```

### Bloque YAML

En `experiment_config.yaml`:

```yaml
phase_N_my_experiment:
  enabled: false          # default off para que no se active sin intención
  source_artifact_path: null
  # ... tus hiperparámetros
```

---

## 6. El modelo de artefactos (para fases desacopladas)

Cuando una fase produce datos que otras fases consumirán (como Phase 6 produciendo el Z del VAE), el patrón canónico es persistir un artefacto y permitir que fases posteriores lo resuelvan mediante `resolve_artifact_path`.

### Producir un artefacto

```python
artifacts_dir = get_artifacts_dir(run_dir)
np.savez(
    artifacts_dir / "phase_N_my_output.npz",
    array1=data1,
    array2=data2,
)
```

Conviene usar un nombre canónico consistente entre fases productoras y consumidoras.

### Consumir un artefacto

```python
artifact_path = resolve_artifact_path(
    artifact_name='phase_N_my_output.npz',
    current_run_dir=run_dir,
    phase_cfg=phase_cfg,
    output_base_dir=Path(config['output']['base_dir']),
)
if artifact_path is None:
    log("[PHASE M] SKIPPED — no upstream artifact")
    return None

data = np.load(artifact_path, allow_pickle=True)
```

### Orden de precedencia

1. Artefacto en el run_dir actual (fase productora corrió en esta sesión).
2. Path explícito en el YAML bajo `source_artifact_path`.
3. Artefacto más reciente en los últimos 10 runs del `output.base_dir`.

Esto permite iteración rápida: re-correr solo la fase consumidora sobre el artefacto de una corrida previa sin re-ejecutar la fase productora.

---

## 7. Checklist antes de hacer commit

Antes de integrar tu componente al repositorio:

- [ ] Tu componente satisface el contrato correspondiente (ver §1).
- [ ] Está registrado en `core/registry.py`.
- [ ] Tiene un bloque en `experiments/experiment_config.yaml` con defaults sensatos y `enabled: false`.
- [ ] Corriste el runner con solo tu fase activa y el CSV de salida tiene el formato esperado.
- [ ] `contract_satisfied = True` (si aplica) en el output.
- [ ] Documentaste las decisiones no triviales (hiperparámetros, supuestos de entrada) en el docstring de la clase.
- [ ] Si tu componente produce artefactos, el nombre del archivo es canónico y documentado.
- [ ] Si tu componente consume artefactos, usa `resolve_artifact_path` (nunca rutas hardcodeadas).
- [ ] Si añadiste una fase nueva, `reproducibility_guide.md` se actualiza con los números objetivo de esa fase.

---

## 8. Patrones a evitar

**No hardcodear rutas absolutas** en el código del componente. Todas las rutas vienen del `config` o de `run_dir`.

**No modificar el runner** para acomodar tu componente. Si necesitas un nuevo patrón que el runner no soporta, discútelo antes de hacer el cambio — probablemente se puede expresar mediante el mecanismo de artefactos.

**No asumir que Phase N corrió antes** cuando tu fase consume datos. Usa `resolve_artifact_path` y maneja el caso `None` con un skip explícito, no con un crash.

**No re-benchmarkear en una fase de producción** de artefactos. Por ejemplo, Phase 6 no hace validación cruzada del encoder linear_compact — ya lo hace Phase 2. Phase 6 entrena **una** instancia del encoder como embedder de producción. Separar roles entre fases es esencial para que los resultados sean interpretables.

**No reportar resultados sin intervalos de confianza o semillas múltiples.** El runner usa 5 semillas por defecto; tu componente debe producir resultados reproducibles bajo esas semillas.
