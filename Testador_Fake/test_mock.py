import pandas as pd
import numpy as np
import warnings
import random
import sys
import os

# Adiciona o diretório raiz ao path para importar classificadores
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split
from Testador_Fake.Dados_fake.Ruidos.ruido_dados import adicionar_ruido
from classificadores.Classificador_Probabilistico import ClassificadorProbabilistico
from classificadores.Classificador_Descritivo import ClassificadorDescritivo
from classificadores.Classificador_Supervisionado import ClassificadorSupervisionado

sys.stdout = open('safe_output.txt', 'w', encoding='utf-8')

# ------------------------------------------------------------------
# Seed global — garante que toda execução produza os mesmos números
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==================================================================
# UTILITÁRIO — Exporta pares reidentificados em CSV legível
# ==================================================================

def exportar_pares_reidentificados(
    predict_matches: pd.MultiIndex,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    true_matches: pd.MultiIndex,
    nome_arquivo: str,
    estagio: str,
    colunas_exibir: list = None,
):
    """
    Gera um CSV com os pares reidentificados, indicando se são TP ou FP.

    Parâmetros
    ----------
    predict_matches : MultiIndex com pares preditos (sinasc_index, sim_index)
    df_a            : DataFrame SINASC (com índice alinhado ao MultiIndex)
    df_b            : DataFrame SIM    (com índice alinhado ao MultiIndex)
    true_matches    : MultiIndex com gabarito completo (para marcar TP/FP)
    nome_arquivo    : Caminho de saída do CSV
    estagio         : Nome do estágio, ex: "Descritivo", "Probabilístico"
    colunas_exibir  : Lista de colunas a incluir no CSV (None = auto)
    """

    if len(predict_matches) == 0:
        print(f"[EXPORT] {estagio}: nenhum par para exportar.")
        return

    # Normaliza colunas dos DataFrames para lowercase
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a.columns = [c.strip().lower() for c in df_a.columns]
    df_b.columns = [c.strip().lower() for c in df_b.columns]

    # Colunas prioritárias a exibir (filtra as que existem)
    colunas_candidatas = colunas_exibir or [
        "nome", "nomemae", "dtnasc", "sexo",
        "codmunres", "codmunnasc", "cep", "peso",
        "racacor", "gestacao", "gravidez", "parto",
    ]

    cols_a = [c for c in colunas_candidatas if c in df_a.columns]
    cols_b = [c for c in colunas_candidatas if c in df_b.columns]

    registros = []
    for sinasc_idx, sim_idx in predict_matches:
        is_tp = (sinasc_idx, sim_idx) in true_matches

        row = {
            "estagio"      : estagio,
            "sinasc_index" : sinasc_idx,
            "sim_index"    : sim_idx,
            "resultado"    : "TP" if is_tp else "FP",
        }

        # Campos do SINASC (prefixo _sinasc)
        if sinasc_idx in df_a.index:
            for col in cols_a:
                row[f"sinasc_{col}"] = df_a.at[sinasc_idx, col]
        else:
            for col in cols_a:
                row[f"sinasc_{col}"] = np.nan

        # Campos do SIM (prefixo _sim)
        if sim_idx in df_b.index:
            for col in cols_b:
                row[f"sim_{col}"] = df_b.at[sim_idx, col]
        else:
            for col in cols_b:
                row[f"sim_{col}"] = np.nan

        registros.append(row)

    df_out = pd.DataFrame(registros)

    # Reordena: TP primeiro, depois FP
    df_out = df_out.sort_values("resultado", ascending=True)  # FP < TP alfabeticamente; inverte:
    df_out["_ord"] = df_out["resultado"].map({"TP": 0, "FP": 1})
    df_out = df_out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)

    output_dir = os.path.join("Dados_fake", "Pares_Reidentificados")
    os.makedirs(output_dir, exist_ok=True)
    caminho_completo = os.path.join(output_dir, nome_arquivo)
    df_out.to_csv(caminho_completo, sep=";", index=False, encoding="utf-8")

    tp_count = (df_out["resultado"] == "TP").sum()
    fp_count = (df_out["resultado"] == "FP").sum()
    print(f"[EXPORT] {estagio}: {len(df_out)} pares exportados "
          f"(TP={tp_count}, FP={fp_count}) → {caminho_completo}")


# ==================================================================
# 1. CARREGAMENTO
# ==================================================================
print("\n" + "="*60)
print(" INICIANDO TESTE COM DADOS MOCKADOS")
print("="*60)

import shutil
dados_dir = os.path.join("Dados_fake", "Dados")
matches_dir = os.path.join("Dados_fake", "Matches")
os.makedirs(dados_dir, exist_ok=True)
os.makedirs(matches_dir, exist_ok=True)

# Mover sinasc e sim
for arq in ["sinasc_800.csv", "sim_800.csv"]:
    if os.path.exists(arq):
        dest = os.path.join(dados_dir, arq)
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(arq, dest)
        print(f"[MOVIDO] {arq} -> {dest}")

# Mover matches_800
matches_fontes = ["matches_800.csv", os.path.join(dados_dir, "matches_800.csv")]
for src in matches_fontes:
    if os.path.exists(src):
        dest = os.path.join(matches_dir, "matches_800.csv")
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(src, dest)
        print(f"[MOVIDO] {src} -> {dest}")

df_a = pd.read_csv(os.path.join(dados_dir, "sinasc_800.csv"), sep=";", encoding="utf-8", dtype=str)
df_b = pd.read_csv(os.path.join(dados_dir, "sim_800.csv"),    sep=";", encoding="utf-8", dtype=str)
true_matches_raw = pd.read_csv(os.path.join(matches_dir, "matches_800.csv"), sep=";", encoding="utf-8", dtype=str)


# Normalização de tipos: evita comparação int vs float no recordlinkage
def to_str_int(col):
    return col.apply(
        lambda x: str(int(float(x))) if pd.notna(x) and x not in ('', 'nan') else np.nan
    )

campos_numericos = ['DTNASC', 'SEXO', 'CODMUNRES', 'CODMUNNASC',
                    'RACACOR', 'GESTACAO', 'GRAVIDEZ', 'PARTO', 'PESO']
for col in campos_numericos:
    if col in df_a.columns: df_a[col] = to_str_int(df_a[col])
    if col in df_b.columns: df_b[col] = to_str_int(df_b[col])

# Gabarito como MultiIndex (índices 0-based, alinhados com o CSV)
df_matches_idx = true_matches_raw.iloc[:, :2].astype(int).copy()
df_matches_idx.columns = ["sinasc_index", "sim_index"]
true_matches_completo = pd.MultiIndex.from_frame(df_matches_idx)

print(f"SINASC: {len(df_a)} registros | SIM: {len(df_b)} registros")
print(f"Gabarito: {len(true_matches_completo)} pares verdadeiros")

# ==================================================================
# 2. RUÍDO — aplicado apenas nos dados do Probabilístico
# ==================================================================
df_a_ruidoso = adicionar_ruido(df_a, seed=SEED)
df_b_ruidoso = adicionar_ruido(df_b, seed=SEED + 1)

# ==================================================================
# 3. ESTÁGIO 1 — DESCRITIVO
# ==================================================================
print("\n" + "="*60)
print(" ESTÁGIO 1 — DESCRITIVO (DETERMINÍSTICO)")
print("="*60)

clf_desc = ClassificadorDescritivo(df_a, df_b, true_matches_completo)
matches_desc, unmatched_a_desc, unmatched_b_desc = clf_desc.linkar()

# --- Exporta pares do Descritivo ---
exportar_pares_reidentificados(
    predict_matches = matches_desc,
    df_a            = clf_desc.df_a,   # já normalizado internamente
    df_b            = clf_desc.df_b,
    true_matches    = true_matches_completo,
    nome_arquivo    = "pares_reidentificados_descritivo.csv",
    estagio         = "Descritivo",
)

# ==================================================================
# 4. ESTÁGIO 2 — PROBABILÍSTICO
# ==================================================================
print("\n" + "="*60)
print(" ESTÁGIO 2 — PROBABILÍSTICO (PESOS EMPÍRICOS)")
print("="*60)

pares_lista = list(true_matches_completo)
train_pairs, test_pairs = train_test_split(
    pares_lista, test_size=0.3, random_state=SEED
)

true_matches_train = pd.MultiIndex.from_tuples(
    train_pairs, names=["sinasc_index", "sim_index"]
)
true_matches_test = pd.MultiIndex.from_tuples(
    test_pairs, names=["sinasc_index", "sim_index"]
)

print(f"Gabarito treino : {len(true_matches_train)} pares  (calibração do threshold)")
print(f"Gabarito teste  : {len(true_matches_test)} pares  (avaliação honesta)")

clf_prob = ClassificadorProbabilistico(
    df_a_ruidoso,
    df_b_ruidoso,
    true_matches=true_matches_train
)

predict_matches, scores, best_threshold, unmatched_a_prob, unmatched_b_prob = \
    clf_prob.calcular_scores_probabilisticos()

# Avaliação honesta no conjunto de TESTE
print("\n--- Avaliação no conjunto de TESTE (holdout 30%) ---")
clf_prob._avaliar_externo(predict_matches, scores, true_matches_test)

# --- Exporta pares do Probabilístico ---
exportar_pares_reidentificados(
    predict_matches = predict_matches,
    df_a            = clf_prob.df_a,
    df_b            = clf_prob.df_b,
    true_matches    = true_matches_completo,   # gabarito completo para marcar TP/FP
    nome_arquivo    = "pares_reidentificados_probabilistico.csv",
    estagio         = "Probabilístico",
)

# ==================================================================
# 4.5 ESTÁGIO 3 — RANDOM FOREST
# ==================================================================
print("\n" + "="*60)
print(" ESTÁGIO 3 — RANDOM FOREST (CLASSIFICADOR SUPERVISIONADO)")
print("="*60)

# Passando o gabarito completo e os matches anteriores (Descritivo + Probabilístico)
matches_anteriores = matches_desc.union(predict_matches)

clf_rf = ClassificadorSupervisionado(
    df_a=unmatched_a_prob,
    df_b=unmatched_b_prob,
    true_matches_completo=true_matches_completo,
    matches_anteriores=matches_anteriores
)

predict_matches_rf = clf_rf.treinar_e_avaliar()

# Mapeando os índices de volta para os originais (já que o ClassificadorSupervisionado faz reset_index)
inv_map_a_rf = {v: k for k, v in clf_rf._map_a.items()}
inv_map_b_rf = {v: k for k, v in clf_rf._map_b.items()}

predict_matches_rf_orig = pd.MultiIndex.from_tuples(
    [(inv_map_a_rf[s], inv_map_b_rf[d]) for s, d in predict_matches_rf],
    names=["sinasc_index", "sim_index"]
) if len(predict_matches_rf) > 0 else pd.MultiIndex.from_tuples([], names=["sinasc_index", "sim_index"])

# Registros não-linkados após o Random Forest
linked_sinasc_rf = predict_matches_rf_orig.get_level_values("sinasc_index")
linked_sim_rf = predict_matches_rf_orig.get_level_values("sim_index")

unmatched_a_rf = unmatched_a_prob[~unmatched_a_prob.index.isin(linked_sinasc_rf)]
unmatched_b_rf = unmatched_b_prob[~unmatched_b_prob.index.isin(linked_sim_rf)]

# --- Exporta pares do Random Forest ---
exportar_pares_reidentificados(
    predict_matches = predict_matches_rf_orig,
    df_a            = df_a,
    df_b            = df_b,
    true_matches    = true_matches_completo,
    nome_arquivo    = "pares_reidentificados_rf.csv",
    estagio         = "Random Forest",
)

# ==================================================================
# 5. RESUMO DA CASCATA
# ==================================================================
print("\n" + "="*60)
print(" RESUMO DA CASCATA — DESCRITIVO + PROBABILÍSTICO + RANDOM FOREST")
print("="*60)

total = len(df_b)

tp_desc = len(true_matches_completo.intersection(matches_desc))
fp_desc = len(matches_desc) - tp_desc
nao_link_desc = len(unmatched_a_desc)

# Para o Probabilístico, avaliamos sobre o total já que prediz na base toda
tp_prob = len(true_matches_completo.intersection(predict_matches))
fp_prob = len(predict_matches) - tp_prob
nao_link_prob = len(unmatched_a_prob)

# Para o Random Forest, mostramos o resultado acumulado (Probabilístico + Random Forest)
matches_rf_cum = predict_matches.union(predict_matches_rf_orig)
tp_rf = len(true_matches_completo.intersection(matches_rf_cum))
fp_rf = len(matches_rf_cum) - tp_rf
nao_link_rf = len(unmatched_a_rf)

print(f"\n  {'Estágio':<22} {'Matches':>8} {'TP':>6} {'FP':>6} {'Não-link %':>12}")
print("  " + "-"*58)
print(f"  {'Descritivo':<22} {len(matches_desc):>8} {tp_desc:>6} {fp_desc:>6} {nao_link_desc/total:>11.1%}")
print(f"  {'Probabilístico':<22} {len(predict_matches):>8} {tp_prob:>6} {fp_prob:>6} {nao_link_prob/total:>11.1%}")
print(f"  {'Random Forest (Acumulado)':<22} {len(matches_rf_cum):>8} {tp_rf:>6} {fp_rf:>6} {nao_link_rf/total:>11.1%}")

print(f"\n  Taxa de reidentificação — Descritivo     : {tp_desc/total:.2%}  ({tp_desc}/{total})")
print(f"  Taxa de reidentificação — Probabilístico : {tp_prob/total:.2%}  ({tp_prob}/{total})")
print(f"  Taxa de reidentificação — Random Forest  : {tp_rf/total:.2%}  ({tp_rf}/{total})")
print(f"\n  Threshold probabilístico usado : {best_threshold:.2f}")
print(f"  Registros ainda sem link       : {nao_link_rf} ({nao_link_rf/total:.1%})")

# ==================================================================
# 6. EXPORTAÇÃO CONSOLIDADA — todos os pares de todos os estágios
# ==================================================================
print("\n" + "="*60)
print(" EXPORTANDO CONSOLIDADO — TODOS OS PARES REIDENTIFICADOS")
print("="*60)

# Lê os CSVs gerados e une num único arquivo
output_dir = os.path.join("Dados_fake", "Pares_Reidentificados")
arquivos_estagio = [
    "pares_reidentificados_descritivo.csv",
    "pares_reidentificados_probabilistico.csv",
    "pares_reidentificados_rf.csv",
]

dfs_consolidados = []
for arq in arquivos_estagio:
    caminho_arq = os.path.join(output_dir, arq)
    try:
         dfs_consolidados.append(pd.read_csv(caminho_arq, sep=";", encoding="utf-8", dtype=str))
    except FileNotFoundError:
         print(f"[AVISO] Arquivo não encontrado para consolidação: {caminho_arq}")

if dfs_consolidados:
    df_consolidado = pd.concat(dfs_consolidados, ignore_index=True)
    # Converter para int para evitar inconsistência de tipo na contagem/filtros
    df_consolidado["sinasc_index"] = df_consolidado["sinasc_index"].astype(float).astype(int)
    df_consolidado["sim_index"] = df_consolidado["sim_index"].astype(float).astype(int)
    
    caminho_todos = os.path.join(output_dir, "pares_reidentificados_TODOS.csv")
    df_consolidado.to_csv(
        caminho_todos,
        sep=";", index=False, encoding="utf-8"
    )
    total_tp = (df_consolidado["resultado"] == "TP").sum()
    total_fp = (df_consolidado["resultado"] == "FP").sum()
    print(f"[EXPORT] Consolidado: {len(df_consolidado)} pares "
          f"(TP={total_tp}, FP={total_fp}) → {caminho_todos}")

    # Contagem de registros únicos reidentificados corretamente no total consolidado
    df_tp = df_consolidado[df_consolidado["resultado"] == "TP"]
    reid_sinasc_unicos = df_tp["sinasc_index"].nunique()
    reid_sim_unicos = df_tp["sim_index"].nunique()

    # Estágio 1 - Descritivo
    tp_desc_pairs = true_matches_completo.intersection(matches_desc)
    tp_desc_df = pd.DataFrame(list(tp_desc_pairs), columns=["sinasc_index", "sim_index"]) if len(tp_desc_pairs) > 0 else pd.DataFrame(columns=["sinasc_index", "sim_index"])
    reid_sinasc_desc = tp_desc_df["sinasc_index"].nunique()
    reid_sim_desc = tp_desc_df["sim_index"].nunique()

    # Estágio 2 - Probabilístico (Geral e Holdout Teste 30%)
    tp_prob_total_pairs = true_matches_completo.intersection(predict_matches)
    tp_prob_total_df = pd.DataFrame(list(tp_prob_total_pairs), columns=["sinasc_index", "sim_index"]) if len(tp_prob_total_pairs) > 0 else pd.DataFrame(columns=["sinasc_index", "sim_index"])
    reid_sinasc_prob_total = tp_prob_total_df["sinasc_index"].nunique()
    reid_sim_prob_total = tp_prob_total_df["sim_index"].nunique()

    tp_prob_test_pairs = true_matches_test.intersection(predict_matches)
    tp_prob_test_df = pd.DataFrame(list(tp_prob_test_pairs), columns=["sinasc_index", "sim_index"]) if len(tp_prob_test_pairs) > 0 else pd.DataFrame(columns=["sinasc_index", "sim_index"])
    reid_sinasc_prob_test = tp_prob_test_df["sinasc_index"].nunique()
    reid_sim_prob_test = tp_prob_test_df["sim_index"].nunique()

    # Estágio 3 - Random Forest (Acumulado)
    tp_rf_pairs = true_matches_completo.intersection(matches_rf_cum)
    tp_rf_df = pd.DataFrame(list(tp_rf_pairs), columns=["sinasc_index", "sim_index"]) if len(tp_rf_pairs) > 0 else pd.DataFrame(columns=["sinasc_index", "sim_index"])
    reid_sinasc_rf = tp_rf_df["sinasc_index"].nunique()
    reid_sim_rf = tp_rf_df["sim_index"].nunique()

    # Total de registros no gabarito
    gabarito_sinasc_unicos = df_matches_idx["sinasc_index"].nunique()
    gabarito_sim_unicos = df_matches_idx["sim_index"].nunique()
    
    # Total na base de dados
    total_sinasc = len(df_a)
    total_sim = len(df_b)
    
    print("\n" + "="*60)
    print(" CONTAGEM DE REIDENTIFICAÇÃO POR ESTÁGIO (ÚNICOS)")
    print("="*60)
    
    print(" ESTÁGIO 1 — DESCRITIVO (DETERMINÍSTICO):")
    print("   SINASC:")
    print(f"     • Reidentificados : {reid_sinasc_desc} de {gabarito_sinasc_unicos} no gabarito ({reid_sinasc_desc/gabarito_sinasc_unicos:.1%})")
    print(f"     • Na base completa: {reid_sinasc_desc} de {total_sinasc} registros ({reid_sinasc_desc/total_sinasc:.1%})")
    print("   SIM:")
    print(f"     • Reidentificados : {reid_sim_desc} de {gabarito_sim_unicos} no gabarito ({reid_sim_desc/gabarito_sim_unicos:.1%})")
    print(f"     • Na base completa: {reid_sim_desc} de {total_sim} registros ({reid_sim_desc/total_sim:.1%})")

    print("\n ESTÁGIO 2 — PROBABILÍSTICO (PESOS EMPÍRICOS):")
    print("   [Conjunto de Teste/Avaliação (Holdout 30%)]")
    print("     SINASC:")
    print(f"       • Reidentificados : {reid_sinasc_prob_test} de {len(true_matches_test)} no gabarito de teste ({reid_sinasc_prob_test/len(true_matches_test):.1%})")
    print("     SIM:")
    print(f"       • Reidentificados : {reid_sim_prob_test} de {len(true_matches_test)} no gabarito de teste ({reid_sim_prob_test/len(true_matches_test):.1%})")
    
    print("   [Geral / Base Completa (100%)]")
    print("     SINASC:")
    print(f"       • Reidentificados : {reid_sinasc_prob_total} de {gabarito_sinasc_unicos} no gabarito total ({reid_sinasc_prob_total/gabarito_sinasc_unicos:.1%})")
    print(f"       • Na base completa: {reid_sinasc_prob_total} de {total_sinasc} ({reid_sinasc_prob_total/total_sinasc:.1%})")
    print("     SIM:")
    print(f"       • Reidentificados : {reid_sim_prob_total} de {gabarito_sim_unicos} no gabarito total ({reid_sim_prob_total/gabarito_sim_unicos:.1%})")
    print(f"       • Na base completa: {reid_sim_prob_total} de {total_sim} ({reid_sim_prob_total/total_sim:.1%})")

    print("\n ESTÁGIO 3 — RANDOM FOREST (CLASSIFICADOR SUPERVISIONADO - ACUMULADO):")
    print("   SINASC:")
    print(f"     • Reidentificados : {reid_sinasc_rf} de {gabarito_sinasc_unicos} no gabarito ({reid_sinasc_rf/gabarito_sinasc_unicos:.1%})")
    print(f"     • Na base completa: {reid_sinasc_rf} de {total_sinasc} registros ({reid_sinasc_rf/total_sinasc:.1%})")
    print("   SIM:")
    print(f"     • Reidentificados : {reid_sim_rf} de {gabarito_sim_unicos} no gabarito ({reid_sim_rf/gabarito_sim_unicos:.1%})")
    print(f"     • Na base completa: {reid_sim_rf} de {total_sim} registros ({reid_sim_rf/total_sim:.1%})")

    print("\n TOTAL CONSOLIDADO (DESCRITIVO + PROBABILÍSTICO + RANDOM FOREST):")
    print("   SINASC:")
    print(f"     • Reidentificados : {reid_sinasc_unicos} de {gabarito_sinasc_unicos} no gabarito ({reid_sinasc_unicos/gabarito_sinasc_unicos:.1%})")
    print(f"     • Na base completa: {reid_sinasc_unicos} de {total_sinasc} ({reid_sinasc_unicos/total_sinasc:.1%})")
    print("   SIM:")
    print(f"     • Reidentificados : {reid_sim_unicos} de {gabarito_sim_unicos} no gabarito ({reid_sim_unicos/gabarito_sim_unicos:.1%})")
    print(f"     • Na base completa: {reid_sim_unicos} de {total_sim} ({reid_sim_unicos/total_sim:.1%})")
    print("="*60 + "\n")

print("\n✅ Execução concluída com sucesso!")
print("📁 Arquivos gerados:")
print("   • safe_output.txt                        — log completo da execução")
print("   • Dados_fake/Pares_Reidentificados/pares_reidentificados_descritivo.csv   — pares do Estágio 1")
print("   • Dados_fake/Pares_Reidentificados/pares_reidentificados_probabilistico.csv — pares do Estágio 2")
print("   • Dados_fake/Pares_Reidentificados/pares_reidentificados_rf.csv           — pares do Estágio 3")
print("   • Dados_fake/Pares_Reidentificados/pares_reidentificados_TODOS.csv        — consolidado de todos os estágios")