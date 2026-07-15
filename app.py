import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew, gaussian_kde
from scipy.linalg import eigh
import re

# Importação de suporte para KMO e Bartlett se disponíveis
try:
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FA_AVAILABLE = True
except ImportError:
    FA_AVAILABLE = False

# Configuração da Página
st.set_page_config(page_title="Plataforma Estatística Avançada", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #2C3E50; }
    .stAlert { margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Plataforma de Análise Estatística Avançada")
st.markdown("Faça o upload da sua base de dados, configure os parâmetros e gere relatórios científicos completos.")

# --- FUNÇÕES MATEMÁTICAS INTERNAS ---

def calcular_descritiva(df, cols):
    stats = []
    for c in cols:
        s = df[c].dropna()
        mean = s.mean()
        std = s.std()
        stats.append({
            'Variável': c,
            'Média': mean,
            'Mediana': s.median(),
            'Moda': s.mode().iloc[0] if not s.mode().empty else np.nan,
            'Desv Padrão': std,
            'Variância': s.var(),
            'CV (%)': (std / mean * 100) if mean != 0 else np.nan,
            'Mínimo': s.min(),
            'Máximo': s.max(),
            'Amplitude': s.max() - s.min(),
            'Q1': s.quantile(0.25),
            'Q3': s.quantile(0.75)
        })
    return pd.DataFrame(stats).set_index('Variável')

def analisar_assimetria(s):
    if s.nunique() <= 1: return "Constante"
    sk = skew(s.dropna())
    if sk > 0.5: return "Assimétrica Positiva"
    elif sk < -0.5: return "Assimétrica Negativa"
    return "Relativamente Simétrica"

def format_p_value(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"

def formatar_texto_latex(texto):
    txt = str(texto).replace('%', '\\%').replace('$', '\\$').replace('_', '\\_')
    return f"\\text{{{txt}}}"

def recuperar_nota_corrompida(val):
    val_str = str(val).strip()
    match = re.match(r'^2026[-/](\d{2})[-/](\d{2})', val_str)
    if match:
        mes = int(match.group(1))
        dia = int(match.group(2))
        return float(f"{dia}.{mes}") if dia <= 5 else float(f"{mes}.{dia}")
    
    match_br = re.match(r'^(\d{2})[-/](\d{2})[-/](2026|\d{2})', val_str)
    if match_br:
        d = int(match_br.group(1))
        m = int(match_br.group(2))
        if d <= 5: return float(f"{d}.{m}")
        if m <= 5: return float(f"{m}.{d}")
        
    limpo = val_str.replace(',', '.')
    limpo = re.sub(r'[^\d\.\-]+', '', limpo)
    try: return float(limpo) if limpo else np.nan
    except: return np.nan

# Rotações de Fatores em NumPy Puro
def varimax_rotation(
