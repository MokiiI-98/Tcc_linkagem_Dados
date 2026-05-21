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
    Retorna uma cópia do DataFrame. O ruído real já é gerado e salvo
    no disco por generate_800.py para que os arquivos SIM_800 e SINASC_800
    contenham bastante ruído visível.
    """
    return df.copy()

