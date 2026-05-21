import os
import pandas as pd

# Resolvendo caminhos relativos ao local do script
base_dir = os.path.dirname(os.path.abspath(__file__))
dados_in_dir = os.path.join(base_dir, "..", "Dados")
matches_in_dir = os.path.join(base_dir, "..", "Matches")

# Load the original 100-row datasets
df_sinasc = pd.read_csv(os.path.join(dados_in_dir, "sinasc_100.csv"), sep=";", dtype=str)
df_sim = pd.read_csv(os.path.join(dados_in_dir, "sim_100.csv"), sep=";", dtype=str)
matches_100 = pd.read_csv(os.path.join(matches_in_dir, "matches_100.csv"), sep=";", dtype=str)

# Generate 800 rows by repeating the dataset 8 times and adding suffixes
df_sinasc_800_list = []
df_sim_800_list = []
matches_800_list = []

surnames = ["", " Oliveira", " Santos", " Souza", " Rodrigues", " Ferreira", " Alves", " Pereira"]
for i in range(8):
    temp_sinasc = df_sinasc.copy()
    temp_sim = df_sim.copy()
    
    suffix = surnames[i]
    if 'NOME' in temp_sinasc.columns:
        temp_sinasc['NOME'] = temp_sinasc['NOME'].apply(lambda x: str(x) + suffix if pd.notna(x) else x)
    if 'NOME' in temp_sim.columns:
        temp_sim['NOME'] = temp_sim['NOME'].apply(lambda x: str(x) + suffix if pd.notna(x) else x)
    if 'NOMEMAE' in temp_sinasc.columns:
        temp_sinasc['NOMEMAE'] = temp_sinasc['NOMEMAE'].apply(lambda x: str(x) + suffix if pd.notna(x) else x)
    if 'NOMEMAE' in temp_sim.columns:
        temp_sim['NOMEMAE'] = temp_sim['NOMEMAE'].apply(lambda x: str(x) + suffix if pd.notna(x) else x)
    
    # Garantir que a combinação de dados seja única alterando o ano de nascimento
    if 'DTNASC' in temp_sinasc.columns:
        temp_sinasc['DTNASC'] = temp_sinasc['DTNASC'].apply(lambda x: str(x)[:-1] + str(i) if pd.notna(x) and len(str(x)) >= 4 else x)
    if 'DTNASC' in temp_sim.columns:
        temp_sim['DTNASC'] = temp_sim['DTNASC'].apply(lambda x: str(x)[:-1] + str(i) if pd.notna(x) and len(str(x)) >= 4 else x)
    
    df_sinasc_800_list.append(temp_sinasc)
    df_sim_800_list.append(temp_sim)

    temp_matches = matches_100.copy()
    temp_matches['sinasc_index'] = temp_matches['sinasc_index'].astype(int) + (i * 100)
    temp_matches['sim_index'] = temp_matches['sim_index'].astype(int) + (i * 100)
    matches_800_list.append(temp_matches)

df_sinasc_800 = pd.concat(df_sinasc_800_list, ignore_index=True)
df_sim_800 = pd.concat(df_sim_800_list, ignore_index=True)
matches_800 = pd.concat(matches_800_list, ignore_index=True)

# Alinhar colunas comuns para garantir casamento perfeito antes do ruído
common_cols = [c for c in df_sinasc_800.columns if c in df_sim_800.columns]
for col in common_cols:
    df_sim_800.loc[50:, col] = df_sinasc_800.loc[50:, col].values

new_matches = pd.DataFrame({
    'sinasc_index': range(50, 800),
    'sim_index': range(50, 800)
})

# Drop duplicates from matches
matches_800 = pd.concat([matches_800, new_matches]).drop_duplicates(subset=['sinasc_index', 'sim_index']).reset_index(drop=True)

# Shuffle the SIM dataset to decouple indices
df_sim_800_shuffled = df_sim_800.sample(frac=1, random_state=42).reset_index(drop=False).rename(columns={'index': 'old_sim_index'})

# Create mapping from old sim_index to the new shuffled index
map_old_to_new_sim = {row['old_sim_index']: new_idx for new_idx, row in df_sim_800_shuffled.iterrows()}

# Update sim_index in matches_800
matches_800['sim_index'] = matches_800['sim_index'].map(map_old_to_new_sim)

# Drop temporary index column
df_sim_800_final = df_sim_800_shuffled.drop(columns=['old_sim_index'])

# Drop specific columns requested by the user
cols_to_drop = ['ESCMAE', 'SERIESCMAE', 'ESCMAE2010', 'VERSAOSIST', 'NUMEROLOTE']
df_sinasc_800 = df_sinasc_800.drop(columns=[c for c in cols_to_drop if c in df_sinasc_800.columns])
df_sim_800_final = df_sim_800_final.drop(columns=[c for c in cols_to_drop if c in df_sim_800_final.columns])

# Insert id_registro column at the beginning of both datasets
df_sinasc_800.insert(0, 'id_registro', range(len(df_sinasc_800)))
df_sim_800_final.insert(0, 'id_registro', range(len(df_sim_800_final)))

# Save to CSV in Dados and Matches directories
output_dir = os.path.join(base_dir, "..", "Dados")
matches_dir = os.path.join(base_dir, "..", "Matches")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(matches_dir, exist_ok=True)
df_sinasc_800.to_csv(os.path.join(output_dir, "sinasc_800.csv"), sep=";", index=False)
df_sim_800_final.to_csv(os.path.join(output_dir, "sim_800.csv"), sep=";", index=False)
matches_800.to_csv(os.path.join(matches_dir, "matches_800.csv"), sep=";", index=False)

print("Generated 800-row mock datasets (shuffled) with id_registro column inside Dados_fake/Dados directory.")
print(f"Total True Matches: {len(matches_800)}")
