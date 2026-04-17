import pandas as pd
import os
from pathlib import Path

# ============================
# Carregar TODOS os arquivos filtrados
# ============================
base_dir = Path(__file__).resolve().parent.parent / "Dados"

# SIM → concatena todos
sim_files = sorted(base_dir.glob("SIM/DO24OPEN_filtrado_teste_v3.csv"))
sim = pd.concat([pd.read_csv(f, sep=";", dtype=str) for f in sim_files], ignore_index=True)

# SINASC → concatena todos
sinasc_files = sorted(base_dir.glob("SINASC/DN24OPEN_filtrado_teste_v3.csv"))
sinasc = pd.concat([pd.read_csv(f, sep=";", dtype=str) for f in sinasc_files], ignore_index=True)

print(f"✅ SIM carregado: {len(sim)} linhas de {len(sim_files)} arquivos")
print(f"✅ SINASC carregado: {len(sinasc)} linhas de {len(sinasc_files)} arquivos")

# ============================
# Padronizar datas (DDMMAAAA → YYYY-MM-DD)
# ============================
for df in [sim, sinasc]:
    if "DTNASC" in df.columns:
        df["DTNASC"] = pd.to_datetime(df["DTNASC"], errors="coerce", format="%d%m%Y").dt.strftime("%Y-%m-%d")

# ============================
# Criar amostras
# ============================
n_sim = 1500
n_sinasc = 1500
num_amostras = 40

amostras_sim, amostras_sinasc = [], []

for i in range(1, num_amostras + 1):
    amostra_sim = sim.sample(n=min(n_sim, len(sim)), random_state=i)
    amostra_sim["id_amostra"] = i
    amostras_sim.append(amostra_sim)

    amostra_sinasc = sinasc.sample(n=min(n_sinasc, len(sinasc)), random_state=i)
    amostra_sinasc["id_amostra"] = i
    amostras_sinasc.append(amostra_sinasc)

amostras_sim = pd.concat(amostras_sim, ignore_index=True)
amostras_sinasc = pd.concat(amostras_sinasc, ignore_index=True)

# ============================
# Salvar
# ============================
os.makedirs("amostras", exist_ok=True)
amostras_sim.to_csv("amostras/amostras_SIM_v3.csv", sep=";", index=False)
amostras_sinasc.to_csv("amostras/amostras_SINASC_v3.csv", sep=";", index=False)

print("📂 amostras salvas em 'amostras/'")
