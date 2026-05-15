import pandas as pd
import numpy as np
import random
import string

def _trocar_letra(texto: str) -> str:
    """Substitui um caractere aleatório por outro."""
    if len(texto) < 3:
        return texto
    i = random.randint(1, len(texto) - 2)
    novo = random.choice(string.ascii_lowercase)
    return texto[:i] + novo + texto[i+1:]

def _transpor_letras(texto: str) -> str:
    """Troca dois caracteres adjacentes (erro de digitação comum)."""
    if len(texto) < 4:
        return texto
    i = random.randint(1, len(texto) - 3)
    lst = list(texto)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return "".join(lst)

def adicionar_ruido(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Aplica degradações realistas nos campos de um DataFrame.
    Simula erros típicos das bases SIM/SINASC reais.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    
    # Guarda as colunas originais e converte para minúsculo temporariamente
    orig_cols = df.columns
    df.columns = [str(c).lower() for c in df.columns]

    # --- Campos de texto: nome e nomemae ---
    for col in ["nome", "nomemae"]:
        if col not in df.columns:
            continue

        mascara = rng.random(n)

        # 2% → campo vazio (ausência de preenchimento)
        df.loc[mascara < 0.02, col] = np.nan

        # 10% → troca de letra (typo simples)
        idx_typo = df.index[
            (mascara >= 0.02) & (mascara < 0.12) & df[col].notna()
        ]
        df.loc[idx_typo, col] = df.loc[idx_typo, col].apply(
            lambda x: _trocar_letra(str(x)) if isinstance(x, str) else x
        )

        # 5% → transposição de letras adjacentes
        idx_transp = df.index[
            (mascara >= 0.12) & (mascara < 0.17) & df[col].notna()
        ]
        df.loc[idx_transp, col] = df.loc[idx_transp, col].apply(
            lambda x: _transpor_letras(str(x)) if isinstance(x, str) else x
        )

    # --- Data de nascimento ---
    if "dtnasc" in df.columns:
        mascara = rng.random(n)

        # 3% → campo ausente
        df.loc[mascara < 0.03, "dtnasc"] = np.nan

        # 10% → dia errado (±1 a ±5 dias)
        idx_data = df.index[(mascara >= 0.03) & (mascara < 0.13) & df["dtnasc"].notna()]
        try:
            # Tenta parsear no formato DDMMAAAA
            datas_parseadas = pd.to_datetime(df.loc[idx_data, "dtnasc"], format="%d%m%Y", errors="coerce")
            datas_alteradas = datas_parseadas + pd.to_timedelta(rng.integers(-5, 6, size=len(idx_data)), unit="D")
            # Devolve para o formato original string DDMMAAAA
            df.loc[idx_data, "dtnasc"] = datas_alteradas.dt.strftime("%d%m%Y")
        except Exception:
            pass

    # --- CEP ---
    if "cep" in df.columns:
        mascara = rng.random(n)

        # 15% → ausente
        df.loc[mascara < 0.15, "cep"] = np.nan

        # 15% → último dígito errado
        idx_cep = df.index[
            (mascara >= 0.15) & (mascara < 0.30) & df["cep"].notna()
        ]
        df.loc[idx_cep, "cep"] = df.loc[idx_cep, "cep"].apply(
            lambda x: str(x)[:-1] + str(rng.integers(0, 10))
            if isinstance(x, str) and len(str(x)) > 3 else x
        )

    # --- Município de residência ---
    if "codmunres" in df.columns:
        mascara = rng.random(n)
        # 10% → ausente
        df.loc[mascara < 0.10, "codmunres"] = np.nan

    # --- Campos opcionais: ausência mais frequente ---
    for col in ["racacor", "estcivmae", "gestacao", "gravidez", "parto"]:
        if col in df.columns:
            mascara = rng.random(n)
            df.loc[mascara < 0.25, col] = np.nan  # 25% ausente

    # Restaura as colunas originais (maiúsculas)
    df.columns = orig_cols
    return df
