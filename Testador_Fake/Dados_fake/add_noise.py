import pandas as pd
import numpy as np
import random
import string

def inject_noise(df):
    df = df.copy()
    n = len(df)
    
    # 1. Dados Ausentes (Missing data) em colunas que costumam falhar no SUS (20% de chance)
    cols_to_null = ['ESTCIVMAE', 'GESTACAO', 'PARTO', 'RACACOR', 'CEP', 'LOGRADOURO', 'PESO']
    for col in cols_to_null:
        if col in df.columns:
            mask = np.random.rand(n) < 0.20
            df.loc[mask, col] = "" # Simula campo não preenchido
            
    # 2. Erros de digitação (Typos) em Nomes (15% de chance)
    for col in ['NOME', 'NOMEMAE']:
        if col in df.columns:
            for i in range(n):
                if random.random() < 0.15:
                    val = str(df.at[i, col])
                    if len(val) > 5:
                        # Substitui uma letra no meio do nome por uma letra aleatória
                        idx = random.randint(1, len(val)-2)
                        df.at[i, col] = val[:idx] + random.choice(string.ascii_lowercase) + val[idx+1:]
                elif random.random() < 0.10:
                    val = str(df.at[i, col])
                    if len(val.split()) > 1:
                        # Remove o ultimo sobrenome
                        df.at[i, col] = " ".join(val.split()[:-1])

    # 3. Pequenos erros de formato de data (Trocar dia com mês) (10% chance)
    if 'DTNASC' in df.columns:
        for i in range(n):
            if random.random() < 0.10:
                val = str(df.at[i, 'DTNASC'])
                if len(val) == 8:
                    # Inverte dia e mês (ex: 05012020 vira 01052020)
                    df.at[i, 'DTNASC'] = val[2:4] + val[:2] + val[4:]
                    
    # 4. Erro de Código de Município (5% chance de ser preenchido errado)
    for col in ['CODMUNRES', 'CODMUNNASC']:
        if col in df.columns:
             for i in range(n):
                 if random.random() < 0.05:
                     df.at[i, col] = "999999"

    return df

print("Carregando CSVs originais...")
df_sim = pd.read_csv("sim_test.csv", sep=";", dtype=str)
df_sinasc = pd.read_csv("sinasc_test.csv", sep=";", dtype=str)

print("Injetando ruídos realistas em toda a base...")
df_sim_noisy = inject_noise(df_sim)
df_sinasc_noisy = inject_noise(df_sinasc)

df_sim_noisy.to_csv("sim_test.csv", sep=";", index=False)
df_sinasc_noisy.to_csv("sinasc_test.csv", sep=";", index=False)
print("Ruídos realistas (Campos vazios, erros de digitação e datas invertidas) aplicados e salvos!")
