import pandas as pd
import zipfile
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent

# Caminhos dos zips
sim_zip = base_path / "SIM.zip"
sinasc_zip = base_path / "SINASC.zip"

# Pastas de destino
sim_folder = base_path / "Dados" / "SIM"
sinasc_folder = base_path / "Dados" / "SINASC"

# Extrair se necessário
if not (sim_folder / "DO24OPEN.csv").exists():
    with zipfile.ZipFile(sim_zip, 'r') as zip_ref:
        zip_ref.extractall(sim_folder)

if not (sinasc_folder / "DN24OPEN.csv").exists():
    with zipfile.ZipFile(sinasc_zip, 'r') as zip_ref:
        zip_ref.extractall(sinasc_folder)

# Ler colunas
colunas_sim = pd.read_csv(sim_folder / "DO24OPEN.csv", nrows=0, sep=";").columns
colunas_sinasc = pd.read_csv(sinasc_folder / "DN24OPEN.csv", nrows=0, sep=";").columns

# Exibir no formato semelhante ao Pandas
print("\n📌 Colunas do SIM:")
print(colunas_sim)

print("\n📌 Colunas do SINASC:")
print(colunas_sinasc)
