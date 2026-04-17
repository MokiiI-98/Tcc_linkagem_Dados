import pandas as pd
import recordlinkage

df_sinasc = pd.read_csv(
    r"C:\Users\luanb\PycharmProjects\tcc\Dados\SINASC\DN24OPEN.csv",
    sep=";", dtype=str
)

df_sim = pd.read_csv(
    r"C:\Users\luanb\PycharmProjects\tcc\Dados\SIM\DO24OPEN.csv",
    sep=";", dtype=str
)


# ============================
# Conversão de datas
# ============================
df_sinasc["DTNASC"] = pd.to_datetime(df_sinasc["DTNASC"], format="%d%m%Y", errors="coerce")
df_sim["DTNASC"] = pd.to_datetime(df_sim["DTNASC"], format="%d%m%Y", errors="coerce")
df_sim["DTOBITO"] = pd.to_datetime(df_sim["DTOBITO"], format="%d%m%Y", errors="coerce")

# ============================
# Conversão de colunas numéricas
# ============================
cols_to_numeric = ["PESO", "IDADEMAE", "QTDFILVIVO", "QTDFILMORT"]
for col in cols_to_numeric:
    if col in df_sinasc.columns:
        df_sinasc[col] = pd.to_numeric(df_sinasc[col], errors="coerce")
    if col in df_sim.columns:
        df_sim[col] = pd.to_numeric(df_sim[col], errors="coerce")

# ============================
# Filtrar registros de 2024
# ============================
df_sinasc = df_sinasc[df_sinasc["DTNASC"].dt.year == 2024]
df_sim = df_sim[df_sim["DTOBITO"].dt.year == 2024]

# ============================
# Filtrar apenas por um Munícipio Específico (Campo Grande - MS)
# ============================
# Garante que os registros sejam da mesma cidade, aumentando muito a chance de encontrar os True Matches
codigo_campo_grande = "500270"
if "CODMUNRES" in df_sinasc.columns:
    df_sinasc = df_sinasc[df_sinasc["CODMUNRES"] == codigo_campo_grande]
if "CODMUNRES" in df_sim.columns:
    df_sim = df_sim[df_sim["CODMUNRES"] == codigo_campo_grande]

# ============================
# Criar colunas auxiliares para bloqueio
# ============================
df_sinasc["ANO_NASC"] = df_sinasc["DTNASC"].dt.year
df_sim["ANO_NASC"] = df_sim["DTNASC"].dt.year

# ============================
# Limpeza de Dados Essenciais (Missing & Inconsistencias)
# ============================
# Conforme a metodologia: "eliminando registros com campos essenciais ausentes ou inconsistentes"
colunas_chave_sinasc = [c for c in ["DTNASC", "SEXO"] if c in df_sinasc.columns]
colunas_chave_sim = [c for c in ["DTNASC", "SEXO"] if c in df_sim.columns]

if colunas_chave_sinasc:
    df_sinasc = df_sinasc.dropna(subset=colunas_chave_sinasc)
if colunas_chave_sim:
    df_sim = df_sim.dropna(subset=colunas_chave_sim)

print("Total de registros limpos (SINASC 2024):", len(df_sinasc))
print("Total de registros limpos (SIM 2024):", len(df_sim))

# ============================
# Caminho de salvamento
# ============================
df_sinasc.to_csv(r"C:\Users\luanb\PycharmProjects\tcc\Dados\SINASC\DN24OPEN_limpo.csv",
                 sep=";", index=False, encoding="utf-8-sig")

df_sim.to_csv(r"C:\Users\luanb\PycharmProjects\tcc\Dados\SIM\DO24OPEN_limpo.csv",
              sep=";", index=False, encoding="utf-8-sig")

print("✅ Dados filtrados e limpos foram salvos para a etapa de Classificação!")
