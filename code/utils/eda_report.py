"""
CLINICAL-CORE: Dossier de Inteligencia Epsilon (EDA)
====================================================
Genera visualizaciones fundamentales para justificar las decisiones
del pipeline TABULAR-CONN sobre la cohorte TCGA-KIRC.

Métricas extraídas:
  1. Análisis de Censura (Kaplan-Meier)
  2. Mapa de Valores Faltantes (Missingness)
  3. Correlación de Variables Clínicas
  4. Distribuciones base (Edad)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
# import yaml
from pathlib import Path

# ==============================================================================
# [Ajuste Epsilon] Resolución de rutas para importar desde el directorio padre
# ==============================================================================
# Calculamos la ruta absoluta de la carpeta 'code' (padre de 'utils')
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Importamos nuestro cirujano de datos
from extractor import TCGAExtractor

# Configuración estética al estilo Epsilon (Oscuro, elegante y profesional)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def generate_eda_report(config_path: str, xml_dir: str, output_dir: str):
    print("=" * 60)
    print("Iniciando Protocolo de Exploración: Dossier Epsilon")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extracción de Datos
    print("[1/5] Invocando TCGAExtractor para asimilar XMLs...")
    extractor = TCGAExtractor(config_path=config_path)
    df_features, df_targets = extractor.extract_cohort(xml_dir)
    
    print(f"      Pacientes asimilados: {len(df_features)}")
    
    # 2. Análisis de Censura (Kaplan-Meier)
    print("[2/5] Calculando curvas de supervivencia (Kaplan-Meier)...")
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    
    # Curva global
    kmf.fit(durations=df_targets['survival_days'], 
            event_observed=df_targets['vital_status'], 
            label="Cohorte General (TCGA-KIRC)")
    ax = kmf.plot_survival_function(linewidth=2.5, color='#2c3e50')
    
    plt.title("Probabilidad de Supervivencia Global (ccRCC)", fontsize=14, fontweight='bold')
    plt.xlabel("Tiempo (Días)", fontsize=12)
    plt.ylabel("Probabilidad de Supervivencia", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_kaplan_meier_global.png"), dpi=300)
    plt.close()

    # 3. Mapa del Vacío (Missingness)
    print("[3/5] Mapeando el vacío (Valores Faltantes)...")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_features.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Mapa de Densidad de Datos (Variables Clínicas)", fontsize=14, fontweight='bold')
    plt.xlabel("Características Clínicas", fontsize=12)
    plt.ylabel("Pacientes", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_missingness_map.png"), dpi=300)
    plt.close()

    # 4. Matriz de Correlación (Excluyendo categóricas no ordinales si es necesario)
    print("[4/5] Tejiendo la red de correlaciones...")
    plt.figure(figsize=(10, 8))
    # Para correlación, tomamos variables numéricas y calculamos matriz de Spearman (ideal para datos médicos)
    numeric_features = df_features.select_dtypes(include=[np.number])
    corr_matrix = numeric_features.corr(method='spearman')
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title("Matriz de Correlación de Spearman (Características)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_correlation_matrix.png"), dpi=300)
    plt.close()

    # 5. Distribuciones Clínicas (Edad como ejemplo)
    print("[5/5] Analizando distribuciones demográficas...")
    if 'age' in df_features.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df_features['age'], kde=True, color='#e74c3c', bins=30)
        plt.title("Distribución de Edad al Diagnóstico", fontsize=14, fontweight='bold')
        plt.xlabel("Edad Normalizada (Z-score)" if extractor.config['features']['age'].get('normalization') else "Edad", fontsize=12)
        plt.ylabel("Frecuencia", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "04_age_distribution.png"), dpi=300)
        plt.close()

    print("\n¡Misión cumplida, mi señor! Gráficos guardados en:", output_dir)
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar Reporte EDA para CLINICAL-CORE")
    parser.add_argument('--config', type=str, default='config_v2.yaml', help='Ruta al config.yaml')
    parser.add_argument('--xml_dir', type=str, default='/content/drive/MyDrive/data_tesis/clinicalsupplement', help='Directorio de los XMLs')
    parser.add_argument('--out_dir', type=str, default='./eda_results', help='Directorio para guardar los gráficos')
    args = parser.parse_args()
    
    generate_eda_report(args.config, args.xml_dir, args.out_dir)