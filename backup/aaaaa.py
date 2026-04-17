# Testador2.py
import pandas as pd
from Classificador_Supervisionado import ClassificadorSupervisionado

# === Carregar amostras
df_sim = pd.read_csv("amostras/amostras_SIM.csv", sep=";", dtype=str)
df_sinasc = pd.read_csv("amostras/amostras_SINASC.csv", sep=";", dtype=str)

# Renomear p/ o classificador
ren = {'DTNASC':'data_nascimento','SEXO':'sexo','RACACOR':'raca'}
df_sim = df_sim.rename(columns=ren).reset_index(drop=True)
df_sinasc = df_sinasc.rename(columns=ren).reset_index(drop=True)

# === Carregar matches verdadeiros (primeiras 2 colunas são os índices)
df_matches = pd.read_csv("Dados/matches_2024_teste.csv", sep=";", dtype=str)
print("[DEBUG] Colunas no matches_2024_teste.csv →", df_matches.columns.tolist())
df_matches_idx = df_matches.iloc[:, :2].astype(int)
df_matches_idx.columns = ["sim_index", "sinasc_index"]
true_matches = pd.MultiIndex.from_frame(df_matches_idx)
print(f"[INFO] True matches (arquivo): {len(true_matches)}")

# Helper para rodar uma condição e reportar interseção
def rodar_experimento(tag, **indexar_kwargs):
    print(f"\n=== {tag} ===")
    cls = ClassificadorSupervisionado(df_sim, df_sinasc)
    pares = cls.indexar(**indexar_kwargs)
    inter = true_matches.intersection(pares)
    print(f"[INFO] Interseção com true_matches: {len(inter)} de {len(true_matches)}")
    return cls, pares, inter

# V1: blocking data_nascimento + sexo (com cap de pares para teste)
cls_b, pares_b, inter_b = rodar_experimento(
    "V1: Blocking (data_nascimento + sexo)",
    blocking=["data_nascimento","sexo"],
    max_pairs=200000  # ajuste se quiser
)

# V2: Full index (cap para não explodir)
cls_f, pares_f, inter_f = rodar_experimento(
    "V2: FullIndex",
    strategy="full",
    max_pairs=200000  # ajuste conforme máquina
)

# Se quiser seguir para features/treino, só quando há pelo menos 1 match válido:
if len(inter_b) > 0:
    feats_b = cls_b.comparar(pares_b)
    print("[OK] Features (blocking):", feats_b.shape)
else:
    print("[WARN] 0 matches na versão com blocking — revise blocking/dados.")

if len(inter_f) > 0:
    feats_f = cls_f.comparar(pares_f)
    print("[OK] Features (full):", feats_f.shape)
else:
    print("[WARN] 0 matches na versão full — verifique índices do CSV de matches.")
