import pandas as pd
from pathlib import Path


def filtrar_csv(caminho_csv, coluna_data, ano_inicio=2020, ano_fim=2024, meses=4, saida="dados_filtrados.csv"):
    """
    Filtra um CSV pelas colunas de data entre anos específicos e primeiros meses.

    :param caminho_csv: Caminho para o CSV de entrada
    :param coluna_data: Nome da coluna com a data (ex: 'DTNASC' ou 'DTOBITO')
    :param ano_inicio: Ano inicial do filtro (default=2020)
    :param ano_fim: Ano final do filtro (default=2024)
    :param meses: Até qual mês filtrar (default=4 -> primeiros 4 meses)
    :param saida: Caminho do arquivo CSV de saída
    """

    # Lê o CSV
    df = pd.read_csv(caminho_csv,sep=";")

    if coluna_data not in df.columns:
        raise ValueError(
            f"⚠️ A coluna '{coluna_data}' não existe no arquivo. Colunas disponíveis: {df.columns.tolist()}")

    # Converte para datetime
    df[coluna_data] = pd.to_datetime(df[coluna_data], format="%d%m%Y", errors="coerce")

    # Filtro de ano e meses
    df_filtrado = df[
        (df[coluna_data].dt.year >= ano_inicio) &
        (df[coluna_data].dt.year <= ano_fim) &
        (df[coluna_data].dt.month <= meses)
        ]

    # Salva resultado
    Path(saida).parent.mkdir(parents=True, exist_ok=True)  # garante que a pasta existe
    df_filtrado.to_csv(saida, index=False, encoding="utf-8-sig")

    print(f"✅ Arquivo filtrado salvo em: {saida} ({len(df_filtrado)} registros)")
    return df_filtrado


# ================== EXEMPLOS DE USO ==================

# Filtrar SINASC
filtrar_csv(
    caminho_csv="Dados/SINASC/filtrado_um_ano_part_1.csv",
    coluna_data="DTNASC",
    saida="Dados/SINASC/dados_filtrados_2020_2024.csv"
)

# Filtrar SIM
filtrar_csv(
    caminho_csv="Dados/SIM/DO240PEN.csv",
    coluna_data="DTOBITO",
    saida="Dados/SIM/dados_filtrados_2020_2024.csv"
)
