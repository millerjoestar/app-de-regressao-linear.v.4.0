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
def varimax_rotation(Phi, gamma=1.0, max_iter=500, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(max_iter):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(np.dot(Phi.T, Lambda**3 - (gamma / p) * np.dot(Lambda, np.diag(np.sum(Lambda**2, axis=0)))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and (d - d_old) / d < tol: break
    return np.dot(Phi, R), R

def promax_rotation(Phi, m=4):
    L_varimax, R_varimax = varimax_rotation(Phi)
    P = np.abs(L_varimax)**m / L_varimax
    coef = np.linalg.lstsq(L_varimax, P, rcond=None)[0]
    u, s, vh = np.linalg.svd(coef)
    T = np.dot(u, vh)
    return np.dot(L_varimax, T)

# Cálculo do Alfa de Cronbach
def calcular_cronbach(df_vars):
    if df_vars.shape[1] < 2:
        return np.nan
    df_clean = df_vars.dropna()
    k = df_clean.shape[1]
    variancias_itens = df_clean.var(ddof=1).sum()
    variancia_total = df_clean.sum(axis=1).var(ddof=1)
    if variancia_total == 0:
        return 0.0
    alfa = (k / (k - 1)) * (1 - (variancias_itens / variancia_total))
    return alfa

# Interface Lateral (Sidebar)
with st.sidebar:
    st.header("⚙️ Painel de Controle")
    uploaded_file = st.file_uploader("1. Carregar Base de Dados", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            
            # Limpeza corretiva contra o bug de datas do Excel
            for col in df.columns:
                if str(col).lower() not in ['obs', 'obs.', 'id', 'identificação', 'unidade', 'região']:
                    df[col] = df[col].apply(recuperar_nota_corrompida)

            all_numeric_cols = [c for c in df.select_dtypes(include=np.number).columns.tolist() if str(c).lower() not in ['obs', 'obs.', 'id']]
            all_numeric_cols = [c for c in all_numeric_cols if df[c].notna().sum() > 0]

            if len(all_numeric_cols) < 2:
                st.error("A base de dados precisa conter ao menos 2 colunas numéricas válidas.")
                st.stop()
            
            df_num = df[all_numeric_cols].dropna()
            
            st.markdown("---")
            tipo_analise = st.radio("2. Tipo de Análise Técnica", ["📈 Regressão Linear Múltipla", "🧬 Análise Fatorial Exploratória (AFE)"])
            
            st.markdown("---")
            if "Regressão" in tipo_analise:
                valid_targets = [c for c in all_numeric_cols if df[c].nunique() > 1]
                target_col = st.selectbox("3. Variável Dependente (Y)", valid_targets)
                independent_cols = [c for c in all_numeric_cols if c != target_col]
            else:
                opcoes_fa = [c for c in all_numeric_cols if df_num[c].nunique() > 1]
                independent_cols = st.multiselect("3. Selecionar Itens para Fatoração", opcoes_fa, default=opcoes_fa)
                
                st.markdown("**Configurações da AFE:**")
                metodo_fatores = st.radio("Critério de Extração",
