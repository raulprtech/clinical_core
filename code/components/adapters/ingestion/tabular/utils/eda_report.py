"""
CLINICAL-CORE: Dossier de Inteligencia Epsilon (Advanced EDA)
====================================================
Genera visualizaciones fundamentales para justificar las decisiones
del pipeline TABULAR-CONN sobre la cohorte TCGA-KIRC.

Nuevas secciones v2:
  1. Análisis de Censura (Kaplan-Meier)
  2. Mapa de Valores Faltantes (Missingness)
  3. Correlación de Variables Clínicas (Spearman)
  4. Distribuciones base (Edad)
  5. Ranking de Poder Prognóstico Univariante (C-index)
  6. Análisis MNAR (Missingness vs Supervivencia)
  7. Análisis de Outliers y Escalamiento
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from pathlib import Path

# ==============================================================================
# RESOLUCIÓN DE RUTAS
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from utils.extractor import TCGAExtractor
except ImportError:
    from extractor import TCGAExtractor

# Configuración estética Epsilon
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def generate_eda_report(config_path: str, xml_dir: str, output_dir: str):
    print("=" * 60)
    print("Invocando Dossier Epsilon: Protocolo de Justificación Arquitectónica")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extracción de Datos
    print("[1/8] Asimilando cohorte...")
    extractor = TCGAExtractor(config_path=config_path)
    df_features, df_targets = extractor.extract_cohort(xml_dir)
    
    # Pre-procesamiento básico para análisis (event indicator)
    df_targets['event'] = df_targets['event'].fillna(0).astype(int)
    valid_surv = (df_targets['survival_days'] > 0) & df_targets['survival_days'].notna()
    df_f = df_features[valid_surv].copy()
    df_t = df_targets[valid_surv].copy()
    
    # 2. Análisis de Censura
    print("[2/8] Graficando supervivencia global...")
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    kmf.fit(df_t['survival_days'], event_observed=df_t['event'], label="TCGA-KIRC")
    kmf.plot_survival_function(linewidth=2.5, color='#2c3e50')
    plt.title("Curva de Supervivencia de la Cohorte", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_kaplan_meier_global.png"), dpi=300)
    plt.close()

    # 3. Mapa de Missingness
    print("[3/8] Analizando densidad de información...")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_f.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Densidad de Información (Missingness Map)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_missingness_map.png"), dpi=300)
    plt.close()

    # 4. Correlación de Spearman
    print("[4/8] Estudiando redundancia de variables...")
    numeric_cols = df_f.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df_f[numeric_cols].corr(method='spearman')
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title("Correlación de Spearman", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_correlation_matrix.png"), dpi=300)
        plt.close()

    # 5. Ranking de Poder Prognóstico Univariante
    print("[5/8] Calculando Poder Prognóstico (C-index ranking)...")
    c_indices = []
    for col in df_f.columns:
        valid_mask = df_f[col].notna()
        if valid_mask.sum() < 20: continue
        
        try:
            # Simple Cox para ranking univariante
            cph = CoxPHFitter(penalizer=0.1)
            temp_df = pd.DataFrame({
                'feature': df_f.loc[valid_mask, col],
                'T': df_t.loc[valid_mask, 'survival_days'],
                'E': df_t.loc[valid_mask, 'event']
            })
            # Manejo de varianza cero
            if temp_df['feature'].std() < 1e-4: continue
            
            cph.fit(temp_df, duration_col='T', event_col='E')
            ci = cph.concordance_index_
            # El C-index puede ser < 0.5 (efecto protector), lo normalizamos a [0.5, 1.0] para el ranking de "importancia"
            c_indices.append({'feature': col, 'c_index': max(ci, 1-ci)})
        except:
            continue
    
    if c_indices:
        df_ci = pd.DataFrame(c_indices).sort_values('c_index', ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='c_index', y='feature', data=df_ci, palette='Reds_r')
        plt.axvline(0.5, color='black', linestyle='--')
        plt.xlim(0.45, 1.0)
        plt.title("Ranking de Poder Prognóstico Univariante (C-index)", fontsize=14, fontweight='bold')
        plt.xlabel("C-index (valor absoluto de predicción)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "05_feature_ranking.png"), dpi=300)
        plt.close()

    # 6. Análisis MNAR (Missingness vs Supervivencia)
    print("[6/8] Test de MNAR (¿Faltar dato es señal clínica?)...")
    plt.figure(figsize=(10, 6))
    # Creamos un score de missingness (cuántas variables le faltan al paciente)
    df_f['null_count'] = df_f.isnull().sum(axis=1)
    # Dividimos en 2 grupos: Pocos nulos vs Muchos nulos (mediana)
    cutoff = df_f['null_count'].median()
    low_miss = df_f['null_count'] <= cutoff
    high_miss = df_f['null_count'] > cutoff
    
    kmf = KaplanMeierFitter()
    if low_miss.any():
        kmf.fit(df_t.loc[low_miss, 'survival_days'], df_t.loc[low_miss, 'event'], label=f"Bajo Nivel de Nulos (<= {cutoff})")
        kmf.plot_survival_function(color='blue')
    if high_miss.any():
        kmf.fit(df_t.loc[high_miss, 'survival_days'], df_t.loc[high_miss, 'event'], label=f"Alto Nivel de Nulos (> {cutoff})")
        kmf.plot_survival_function(color='red')
        
    plt.title("Impacto del 'Missingness' en la Supervivencia", fontsize=14, fontweight='bold')
    plt.ylabel("Probabilidad de Supervivencia")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_mnar_analysis.png"), dpi=300)
    plt.close()

    # 7. Análisis de Outliers y Escalamiento
    print("[7/8] Analizando distribuciones para justificar escalamiento...")
    important_cont = df_ci['feature'].head(3).tolist() if c_indices else []
    if important_cont:
        plt.figure(figsize=(12, 6))
        # Seleccionamos las 3 variables más importantes para mostrar su distribución
        data_plot = df_f[important_cont].melt()
        sns.violinplot(x='variable', y='value', data=data_plot, inner='quartile')
        plt.title("Distribución y Outliers de Variables Top", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "07_outlier_analysis.png"), dpi=300)
        plt.close()

    # 8. Categorical Balance
    print("[8/8] Analizando balance de categorías...")
    cat_cols = df_f.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        # Analizamos solo la primera o más relevante
        top_cat = cat_cols[0]
        plt.figure(figsize=(8, 5))
        sns.countplot(y=top_cat, data=df_f, order=df_f[top_cat].value_counts().index)
        plt.title(f"Balance de Clases: {top_cat}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "08_categorical_balance.png"), dpi=300)
        plt.close()

    print("\n¡Protocolo Finalizado! Decodificaciones guardadas en:", output_dir)
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar Reporte EDA Avanzado para CLINICAL-CORE")
    parser.add_argument('--config', type=str, default='code/configs/config.yaml', help='Ruta al config.yaml')
    parser.add_argument('--xml_dir', type=str, default='/content/drive/MyDrive/data_tesis/clinicalsupplement', help='Directorio de los XMLs')
    parser.add_argument('--out_dir', type=str, default='./eda_results_advanced', help='Directorio para guardar los gráficos')
    args = parser.parse_args()
    
    generate_eda_report(args.config, args.xml_dir, args.out_dir)