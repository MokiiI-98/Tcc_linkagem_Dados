import sys
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from Classificador_Supervisionado import ClassificadorSupervisionado
from Classificador_Probabilistico import ClassificadorProbabilistico

# ===============================
# 1. Carregar os datasets (Usando os limpos gerados pelo filtro)
# ===============================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df_a = pd.read_csv(os.path.join(base_dir, "Dados", "SINASC", "DN24OPEN_limpo.csv"), sep=";", encoding="utf-8")
df_b = pd.read_csv(os.path.join(base_dir, "Dados", "SIM", "DO24OPEN_limpo.csv"), sep=";", encoding="utf-8")
true_matches = pd.read_csv(os.path.join(base_dir, "Dados", "matches_2024_teste.csv"), sep=";", encoding="utf-8")

# ===============================
# 2. Testar Classificador Machine Learning (Supervisionado)
# ===============================
print("\n" + "="*50)
print("🚀 INICIANDO TESTE: RANDOM FOREST (SUPERVISIONADO)")
print("="*50)

classificador_ml = ClassificadorSupervisionado(df_a, df_b, true_matches)

# Mapeamento dos índices
map_sim = {old: new for new, old in enumerate(classificador_ml.df_b.index)}
map_sinasc = {old: new for new, old in enumerate(classificador_ml.df_a.index)}

df_matches_idx = true_matches.iloc[:, :2].copy()
df_matches_idx.columns = ["sinasc_index", "sim_index"]
df_matches_idx["sinasc_index"] = df_matches_idx["sinasc_index"].map(map_sinasc)
df_matches_idx["sim_index"] = df_matches_idx["sim_index"].map(map_sim)
df_matches_idx = df_matches_idx.dropna().astype(int)

classificador_ml.true_matches = pd.MultiIndex.from_frame(df_matches_idx[["sinasc_index", "sim_index"]])
modelo_ml = classificador_ml.treinar_e_avaliar()

# ===============================
# 3. Testar Classificador Probabilístico
# ===============================
print("\n" + "="*50)
print("🧮 INICIANDO TESTE: PROBABILÍSTICO (PESOS MATEMÁTICOS)")
print("="*50)

classificador_prob = ClassificadorProbabilistico(df_a, df_b, true_matches)

# O classificador probabilístico também precisa do mesmo mapeamento se for usar as métricas internas
classificador_prob.true_matches = pd.MultiIndex.from_frame(df_matches_idx[["sinasc_index", "sim_index"]])

predict_matches, scores = classificador_prob.calcular_scores_probabilisticos()

print("\n✅ Ambos os pipelines foram executados com sucesso!")
