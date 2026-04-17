import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from classificadores.Classificador_Supervisionado import ClassificadorSupervisionado
from classificadores.Classificador_Probabilistico import ClassificadorProbabilistico
from classificadores.Classificador_Descritivo import ClassificadorDescritivo
import os
import sys
sys.stdout = open('safe_output.txt', 'w', encoding='utf-8')

print("\n" + "="*50)
print(" INICIANDO TESTE COM DADOS MOCKADOS")
print("="*50)

df_a = pd.read_csv("sinasc_100.csv", sep=";", encoding="utf-8", dtype=str)
df_b = pd.read_csv("sim_100.csv", sep=";", encoding="utf-8", dtype=str)
true_matches = pd.read_csv("matches_100.csv", sep=";", encoding="utf-8", dtype=str)

print("\n" + "="*50)
print(" RANDOM FOREST (SUPERVISIONADO)")
print("="*50)

classificador_ml = ClassificadorSupervisionado(df_a, df_b, true_matches)

# Mapeamento dos índices
map_sim = {old: new for new, old in enumerate(classificador_ml.df_b.index)}
map_sinasc = {old: new for new, old in enumerate(classificador_ml.df_a.index)}

df_matches_idx = true_matches.iloc[:, :2].copy()
df_matches_idx = df_matches_idx.astype(int)
df_matches_idx.columns = ["sinasc_index", "sim_index"]
df_matches_idx["sinasc_index"] = df_matches_idx["sinasc_index"].map(map_sinasc)
df_matches_idx["sim_index"] = df_matches_idx["sim_index"].map(map_sim)
df_matches_idx = df_matches_idx.dropna().astype(int)

classificador_ml.true_matches = pd.MultiIndex.from_frame(df_matches_idx[["sinasc_index", "sim_index"]])
print("DEBUG true_matches:", classificador_ml.true_matches[:5])
modelo_ml = classificador_ml.treinar_e_avaliar()

print("\n" + "="*50)
print(" PROBABILÍSTICO (PESOS MATEMÁTICOS)")
print("="*50)

classificador_prob = ClassificadorProbabilistico(df_a, df_b, true_matches)
classificador_prob.true_matches = pd.MultiIndex.from_frame(df_matches_idx[["sinasc_index", "sim_index"]])
predict_matches, scores = classificador_prob.calcular_scores_probabilisticos()

print("\n" + "="*50)
print(" DESCRITIVO (DETERMINÍSTICO - REGRAS DE BANCO)")
print("="*50)

classificador_desc = ClassificadorDescritivo(df_a, df_b, true_matches)
classificador_desc.true_matches = pd.MultiIndex.from_frame(df_matches_idx[["sinasc_index", "sim_index"]])
matches_desc = classificador_desc.linkar()

print("\n✅ Todos os 3 pipelines mockados foram executados com sucesso!")
