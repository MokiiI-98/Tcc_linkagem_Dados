import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def random_date(start, end):
    return (start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())),
    )).strftime('%d%m%Y')

start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

sim_columns = ['NOME', 'NOMEMAE', 'LOGRADOURO', 'contador', 'ORIGEM', 'TIPOBITO', 'DTOBITO', 'HORAOBITO', 'NATURAL',
       'CODMUNNATU', 'DTNASC', 'IDADE', 'SEXO', 'RACACOR', 'ESTCIV', 'ESC',
       'ESC2010', 'SERIESCFAL', 'OCUP', 'CODMUNRES', 'LOCOCOR', 'CODESTAB',
       'CODMUNOCOR', 'IDADEMAE', 'ESCMAE', 'ESCMAE2010', 'SERIESCMAE',
       'OCUPMAE', 'QTDFILVIVO', 'QTDFILMORT', 'GRAVIDEZ', 'SEMAGESTAC',
       'GESTACAO', 'PARTO', 'OBITOPARTO', 'PESO', 'TPMORTEOCO', 'OBITOGRAV',
       'OBITOPUERP', 'ASSISTMED', 'EXAME', 'CIRURGIA', 'NECROPSIA', 'LINHAA',
       'LINHAB', 'LINHAC', 'LINHAD', 'LINHAII', 'CAUSABAS', 'CB_PRE',
       'COMUNSVOIM', 'DTATESTADO', 'CIRCOBITO', 'ACIDTRAB', 'FONTE',
       'NUMEROLOTE', 'DTINVESTIG', 'DTCADASTRO', 'ATESTANTE', 'STCODIFICA',
       'CODIFICADO', 'VERSAOSIST', 'VERSAOSCB', 'FONTEINV', 'DTRECEBIM',
       'ATESTADO', 'DTRECORIGA', 'OPOR_DO', 'CAUSAMAT', 'ESCMAEAGR1',
       'ESCFALAGR1', 'STDOEPIDEM', 'STDONOVA', 'DIFDATA', 'NUDIASOBCO',
       'DTCADINV', 'TPOBITOCOR', 'DTCONINV', 'FONTES', 'TPRESGINFO',
       'TPNIVELINV', 'DTCADINF', 'MORTEPARTO', 'DTCONCASO', 'ALTCAUSA',
       'CAUSABAS_O', 'TPPOS', 'TP_ALTERA', 'CB_ALT']

sinasc_columns = ['NOME', 'NOMEMAE', 'LOGRADOURO', 'contador', 'ORIGEM', 'CODESTAB', 'CODMUNNASC', 'LOCNASC', 'IDADEMAE',
       'ESTCIVMAE', 'ESCMAE', 'CODOCUPMAE', 'QTDFILVIVO', 'QTDFILMORT',
       'CODMUNRES', 'GESTACAO', 'GRAVIDEZ', 'PARTO', 'CONSULTAS', 'DTNASC',
       'HORANASC', 'SEXO', 'APGAR1', 'APGAR5', 'RACACOR', 'PESO', 'IDANOMAL',
       'DTCADASTRO', 'CODANOMAL', 'NUMEROLOTE', 'VERSAOSIST', 'DTRECEBIM',
       'DIFDATA', 'OPORT_DN', 'DTRECORIGA', 'NATURALMAE', 'CODMUNNATU',
       'CODUFNATU', 'ESCMAE2010', 'SERIESCMAE', 'DTNASCMAE', 'RACACORMAE',
       'QTDGESTANT', 'QTDPARTNOR', 'QTDPARTCES', 'IDADEPAI', 'DTULTMENST',
       'SEMAGESTAC', 'TPMETESTIM', 'CONSPRENAT', 'MESPRENAT', 'TPAPRESENT',
       'STTRABPART', 'STCESPARTO', 'TPNASCASSI', 'TPFUNCRESP', 'TPDOCRESP',
       'DTDECLARAC', 'ESCMAEAGR1', 'STDNEPIDEM', 'STDNNOVA', 'CODPAISRES',
       'TPROBSON', 'PARIDADE', 'KOTELCHUCK']

def generate_mock_data(columns, n=300):
    first_names = ["Jose", "Maria", "Joao", "Ana", "Carlos", "Paulo", "Pedro", "Lucas", "Luiz", "Marcos"]
    last_names = ["Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Alves", "Pereira", "Lima", "Gomes"]
    logradouros = ["Rua Quinze de Novembro", "Avenida Paulista", "Rua do Comercio", "Avenida Brasil", "Rua Direita"]

    data = {}
    for col in columns:
        if col == 'contador':
            data[col] = range(1, n + 1)
        elif 'DT' in col:
            data[col] = [random_date(start_date, end_date) for _ in range(n)]
        elif col in ['NOME', 'NOMEMAE']:
            data[col] = [f"{random.choice(first_names)} {random.choice(last_names)} {random.choice(last_names)}" for _ in range(n)]
        elif col == 'LOGRADOURO':
            data[col] = [f"{random.choice(logradouros)}, {random.randint(10, 9999)}" for _ in range(n)]
        elif col in ['SEXO']:
            data[col] = np.random.choice([1, 2], n)
        elif col in ['PESO']:
            data[col] = np.random.randint(500, 4500, n)
        elif col in ['IDADE', 'IDADEMAE', 'IDADEPAI']:
            data[col] = np.random.randint(15, 50, n)
        elif col in ['RACACOR', 'RACACORMAE']:
            data[col] = np.random.choice([1, 2, 3, 4, 5], n)
        elif col in ['CAUSABAS', 'CAUSABAS_O', 'CB_ALT', 'CB_PRE']:
            data[col] = ['A' + str(np.random.randint(10, 99)) for _ in range(n)]
        else:
            data[col] = np.random.randint(0, 100, n)
    return pd.DataFrame(data)

sim_df = generate_mock_data(sim_columns, 300)
sinasc_df = generate_mock_data(sinasc_columns, 300)

num_matches = 50
common_cols = list(set(sim_columns).intersection(set(sinasc_columns)))
for i in range(num_matches):
    for col in common_cols:
        if col != 'contador':
            sim_df.at[i, col] = sinasc_df.at[i, col]
            
    # Apply severe noise / variations to SIM names and exact columns
    # so accuracy drops below 1.00 and we test the model's breaking points.
    for col in common_cols:
        if col != 'contador':
            # Add severe noise to exact matching columns occasionally (20% chance)
            if col in ['RACACOR', 'ESTCIVMAE', 'GESTACAO', 'PARTO', 'PESO', 'SEXO'] and random.random() < 0.2:
                sim_df.at[i, col] = np.random.randint(0, 5) # Incorrect category
            
            # 15% chance to set to empty (Missing Data)
            if random.random() < 0.15:
                sim_df.at[i, col] = ""

    for col in ['NOME', 'NOMEMAE']:
        original_name = str(sim_df.at[i, col])
        if not original_name.strip(): continue
        parts = original_name.split()
        
        # Heavy name noise
        dice = random.random()
        if dice < 0.20 and len(parts) > 1:
             sim_df.at[i, col] = " ".join(parts[:-1]) # Omit last name
        elif dice < 0.50 and len(parts) > 1:
             # Swap 
             sim_df.at[i, col] = " ".join([parts[1], parts[0]] + parts[2:])
        elif dice < 0.65:
             # Add typo (character replacement)
             chars = list(original_name)
             if chars:
                 chars[random.randint(0, len(chars)-1)] = random.choice("abcdefghijklmnopqrstuvwxyz")
                 sim_df.at[i, col] = "".join(chars)
             
    original_log = str(sim_df.at[i, 'LOGRADOURO'])
    if original_log.strip() and random.random() < 0.4:
         # Typo in street numbers or name
         sim_df.at[i, 'LOGRADOURO'] = original_log[:-1] + str(random.randint(1,9))

    # Break blocking keys for 10% of the matches (creates False Negatives because they will not be indexed!)
    if random.random() < 0.1:
        sim_df.at[i, 'DTNASC'] = "01011999"
        sim_df.at[i, 'CODMUNRES'] = 999999
        sim_df.at[i, 'CODMUNNASC'] = 999999

# Add Hard Negatives (twins and homonyms) to confuse the model
# These are NOT in matches_test.csv, so the model must learn to REJECT them!
for i in range(50, 70):
    for col in common_cols:
        if col != 'contador':
            sim_df.at[i, col] = sinasc_df.at[i, col]
    
    if i < 60:
        # Twins: Same mother, town, birth, but different name and sex
        sim_df.at[i, 'SEXO'] = 1 if sinasc_df.at[i, 'SEXO'] == 2 else 2
        sim_df.at[i, 'NOME'] = "Gemeo " + str(sim_df.at[i, 'NOME'])
    else:
        # Homonyms: Same name and town, but different birth date and mother
        sim_df.at[i, 'DTNASC'] = "01011980"
        sim_df.at[i, 'NOMEMAE'] = "Mae Diferente Silva"

true_matches = pd.DataFrame({
    'sinasc_index': range(num_matches),
    'sim_index': range(num_matches)
})

sim_df.to_csv('sim_test.csv', index=False, sep=';')
sinasc_df.to_csv('sinasc_test.csv', index=False, sep=';')
true_matches.to_csv('matches_test.csv', index=False, sep=';')

print("Files sim_test.csv, sinasc_test.csv, and matches_test.csv generated successfully.")
