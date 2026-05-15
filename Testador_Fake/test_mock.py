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

df_a = pd.read_csv("sinasc_800.csv", sep=";", encoding="utf-8", dtype=str)
df_b = pd.read_csv("sim_800.csv", sep=";", encoding="utf-8", dtype=str)
true_matches = pd.read_csv("matches_800.csv", sep=";", encoding="utf-8", dtype=str)

# Correção de tipos (float vs int) que quebravam o recordlinkage
import numpy as np

def to_str_int(col):
    return col.apply(lambda x: str(int(float(x))) if pd.notna(x) else np.nan)

campos_numericos = ['DTNASC', 'SEXO', 'CODMUNRES', 'CODMUNNASC',
                    'RACACOR', 'GESTACAO', 'GRAVIDEZ', 'PARTO', 'PESO']

for col in campos_numericos:
    if col in df_b.columns: df_b[col] = to_str_int(df_b[col])
    if col in df_a.columns: df_a[col] = to_str_int(df_a[col])

from ruido_dados import adicionar_ruido

# Aplica ruído ANTES de passar para os classificadores (todos eles usarão dados sujos)
df_a = adicionar_ruido(df_a, seed=42)
df_b = adicionar_ruido(df_b, seed=99)


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

print("\n" + "=" * 50)
print(" VERIFICAÇÃO DE INTEGRIDADE DO GABARITO (AMOSTRA)")
print("=" * 50)

# Pega os 5 primeiros pares do gabarito para verificação
for idx_sinasc, idx_sim in classificador_ml.true_matches[:5]:
    nome_sinasc = classificador_ml.df_a.loc[idx_sinasc, "NOME"]
    nome_sim = classificador_ml.df_b.loc[idx_sim, "NOME"]

    mae_sinasc = classificador_ml.df_a.loc[idx_sinasc, "NOMEMAE"]
    mae_sim = classificador_ml.df_b.loc[idx_sim, "NOMEMAE"]

    print(f"🔹 Par de Match (SINASC: {idx_sinasc} <-> SIM: {idx_sim})")
    print(f"   [SINASC] NOME: {nome_sinasc: <25} | MÃE: {mae_sinasc}")
    print(f"   [SIM]    NOME: {nome_sim: <25} | MÃE: {mae_sim}")
    print("-" * 50)
# ---------------------------------------------------------


modelo_ml = classificador_ml.treinar_e_avaliar()

print("\n" + "="*50)
print(" PROBABILÍSTICO (PESOS MATEMÁTICOS)")
print("="*50)

# Fora da classe, antes de instanciar

# Fora da classe, antes de instanciar
from sklearn.model_selection import train_test_split
true_matches_multi = pd.MultiIndex.from_frame(df_matches_idx[["sinasc_index", "sim_index"]])
pares_lista = list(true_matches_multi)
train_pairs, test_pairs = train_test_split(pares_lista, test_size=0.3, random_state=42)

true_matches_train = pd.MultiIndex.from_tuples(train_pairs, names=["sinasc_index", "sim_index"])
true_matches_test  = pd.MultiIndex.from_tuples(test_pairs, names=["sinasc_index", "sim_index"])

# Passa o train para calibrar o threshold, guarda o test para avaliar
classificador_prob = ClassificadorProbabilistico(df_a, df_b, true_matches=true_matches_train)
predict_matches, scores, best_threshold, unmatched_a, unmatched_b = classificador_prob.calcular_scores_probabilisticos()

# Avalia honestamente no test
classificador_prob._avaliar_externo(predict_matches, scores, true_matches_test)

print("\n" + "="*50)
print(" DESCRITIVO (DETERMINÍSTICO - REGRAS DE BANCO)")
print("="*50)

classificador_desc = ClassificadorDescritivo(df_a, df_b, true_matches)
classificador_desc.true_matches = pd.MultiIndex.from_frame(df_matches_idx[["sinasc_index", "sim_index"]])
matches_desc, unmatched_a_desc, unmatched_b_desc = classificador_desc.linkar()

print("\n✅ Todos os 3 pipelines mockados foram executados com sucesso!")
