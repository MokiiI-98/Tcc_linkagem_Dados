import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Classificador_Supervisionado import ClassificadorSupervisionado
from recordlinkage import LogisticRegressionClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 1. Carregar dados pequenos
df_sim = pd.read_csv("../amostras/amostras_SIM.csv", sep=";", dtype=str)
df_sinasc = pd.read_csv("../amostras/amostras_SINASC.csv", sep=";", dtype=str)

# 2. Renomear colunas
df_sim = df_sim.rename(columns={
    'DTNASC': 'data_nascimento',
    'SEXO': 'sexo',
    'RACACOR': 'raca'
})
df_sinasc = df_sinasc.rename(columns={
    'DTNASC': 'data_nascimento',
    'SEXO': 'sexo',
    'RACACOR': 'raca'
})

# 3. Reindexar
df_sim = df_sim.reset_index(drop=True)
df_sinasc = df_sinasc.reset_index(drop=True)

# 4. Criar classificador
classificador = ClassificadorSupervisionado(df_sim, df_sinasc)

# 5. Gerar pares
pares = classificador.indexar()
print(f"[INFO] Total de pares gerados: {len(pares)}")

# 6. Carregar pares verdadeiros do CSV
df_matches = pd.read_csv("../Dados/matches_2024_teste.csv", sep=";", dtype=str)
print("[DEBUG] Colunas no matches_2024_teste.csv →", df_matches.columns.tolist())

# Pegar só as duas primeiras colunas com índices originais
df_matches_idx = df_matches.iloc[:, :2].astype(int)

# Criar mapeamento dos índices originais → novos índices (após reset_index)
map_sim = {old: new for new, old in enumerate(df_sim.index)}
map_sinasc = {old: new for new, old in enumerate(df_sinasc.index)}

# Renomear colunas
df_matches_idx = df_matches_idx.rename(columns={
    df_matches_idx.columns[0]: "sim_index",
    df_matches_idx.columns[1]: "sinasc_index"
})

# Aplicar mapeamento
df_matches_idx["sim_index"] = df_matches_idx["sim_index"].map(map_sim)
df_matches_idx["sinasc_index"] = df_matches_idx["sinasc_index"].map(map_sinasc)

# Remover linhas inválidas
df_matches_idx = df_matches_idx.dropna().astype(int)

# Criar MultiIndex final
true_matches = pd.MultiIndex.from_frame(df_matches_idx[["sim_index", "sinasc_index"]])
print(f"[INFO] Total de true_matches após reindexação: {len(true_matches)}")

# 7. Criar vetor de features
features = classificador.comparar(pares)

# 8. Balanceamento
y_true_series = pd.Series(0, index=features.index)

# Interseção: só pega pares que existem nos dois
true_matches_validos = true_matches.intersection(features.index)

print(f"[INFO] True matches válidos após interseção: {len(true_matches_validos)}")

# Marca apenas os válidos
y_true_series.loc[true_matches_validos] = 1


matches_idx = y_true_series[y_true_series == 1].index
nonmatches_idx = y_true_series[y_true_series == 0].index

n_matches = len(matches_idx)
nonmatches_sample = nonmatches_idx.to_series().sample(
    n=min(2 * n_matches, len(nonmatches_idx)), random_state=42
).index

final_index = matches_idx.union(nonmatches_sample)
features_bal = features.loc[final_index]
y_true_bal = y_true_series.loc[final_index]

print(f"[INFO] Conjunto balanceado → {len(matches_idx)} matches + {len(nonmatches_sample)} não-matches")

# 9. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    features_bal, y_true_bal, test_size=0.3, random_state=42, stratify=y_true_bal
)
print(f"[INFO] Divisão treino/teste → {len(X_train)} treino | {len(X_test)} teste")

# 10. Treinar classificador
clf = LogisticRegressionClassifier()
y_train_idx = y_train[y_train == 1].index
clf.fit(X_train, y_train_idx)

# 11. Avaliar
y_pred = clf.predict(X_test)
y_pred_series = pd.Series(0, index=X_test.index)
y_pred_series.loc[y_pred] = 1

print("\n[RELATÓRIO DE CLASSIFICAÇÃO - TESTE]")
print(classification_report(y_test, y_pred_series))
