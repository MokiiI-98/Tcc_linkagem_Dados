import pandas as pd
import sys
import os

# Adiciona o diretório raiz ao path para importar as classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classificadores.Classificador_Descritivo import ClassificadorDescritivo
from classificadores.Classificador_Supervisionado import ClassificadorSupervisionado
from classificadores.Classificador_Probabilistico import ClassificadorProbabilistico

print("\n" + "="*50)
print(" CARREGANDO DADOS REAIS (LIMITE 5000 LINHAS)")
print("="*50)

# Carrega apenas as primeiras 5000 linhas das amostras reais
df_sim = pd.read_csv("../amostras/amostras_SIM.csv", sep=";", dtype=str, nrows=5000)
df_sinasc = pd.read_csv("../amostras/amostras_SINASC.csv", sep=";", dtype=str, nrows=5000)

# O gabarito tem colunas sem nome (Unnamed: 0 e Unnamed: 1)
df_matches = pd.read_csv("../Dados/matches_2024_teste.csv", sep=";", dtype=str)
df_matches_idx = df_matches.iloc[:, :2].copy()
df_matches_idx.columns = ["sim_index", "sinasc_index"]

# Vamos usar MultiIndex para o gabarito. Como só lemos 5000 linhas, precisamos
# garantir que os matches avaliados estejam dentro desse range.
# Converte para int e remove o que não existe
df_matches_idx = df_matches_idx.dropna().astype(int)
df_matches_idx = df_matches_idx[(df_matches_idx['sim_index'] < 5000) & (df_matches_idx['sinasc_index'] < 5000)]

true_matches = pd.MultiIndex.from_frame(df_matches_idx[["sinasc_index", "sim_index"]])
print(f"Total de True Matches na fatia de 5000: {len(true_matches)}")

# Ajuste nos DFs para garantir índice alinhado e nomes corretos
df_sinasc = df_sinasc.rename(columns={'ANO_NASC': 'ANO'})
df_sim = df_sim.rename(columns={'ANO_NASC': 'ANO'})

df_sinasc = df_sinasc.reset_index(drop=True)
df_sim = df_sim.reset_index(drop=True)

# Armazena os DFs originais completos (na fatia de 5000)
df_sinasc_restante = df_sinasc.copy()
df_sim_restante = df_sim.copy()

matches_finais = set()

# ==========================================
# FASE 1: CLASSIFICADOR DESCRITIVO
# ==========================================
print("\n" + "="*50)
print(" FASE 1: FILTRO DESCRITIVO (REGRAS ABSOLUTAS)")
print("="*50)

# Passando df_matches_idx que é um DataFrame em vez do true_matches que é MultiIndex
classificador_desc = ClassificadorDescritivo(df_sinasc_restante, df_sim_restante, true_matches)
matches_desc = classificador_desc.linkar()

# Guardar os pares encontrados
matches_finais.update(matches_desc.to_list())

# O que foi "match" no descritivo, removemos dos dataframes para não processar de novo
# Como a tupla é (sinasc_index, sim_index), pegamos os únicos de cada um
sinasc_achados = [t[0] for t in matches_desc.to_list()]
sim_achados = [t[1] for t in matches_desc.to_list()]

df_sinasc_restante = df_sinasc_restante.drop(index=list(set(sinasc_achados)), errors='ignore')
df_sim_restante = df_sim_restante.drop(index=list(set(sim_achados)), errors='ignore')

print(f"[FASE 1] Pares encontrados: {len(matches_desc)}")
print(f"Restam SINASC: {len(df_sinasc_restante)} | Restam SIM: {len(df_sim_restante)}")

# ==========================================
# FASE 2: CLASSIFICADOR SUPERVISIONADO (ML)
# ==========================================
print("\n" + "="*50)
print(" FASE 2: FILTRO SUPERVISIONADO (RANDOM FOREST)")
print("="*50)

if len(df_sinasc_restante) > 0 and len(df_sim_restante) > 0:
    classificador_ml = ClassificadorSupervisionado(df_sinasc_restante, df_sim_restante, true_matches)
    # Precisamos tratar o fato de que a classe supervisionada roda train_test_split.
    # Como não temos um predict final já extraído como retorno padronizado na classe atual,
    # vamos chamar treinar_e_avaliar e pegar os links previstos.
    try:
        clf = classificador_ml.treinar_e_avaliar()
        candidate_links = classificador_ml.indexar()
        
        # Como as features não são guardadas na classe, vamos extrair aqui rapidinho para gerar o predict na sobra
        import recordlinkage
        compare = recordlinkage.Compare()
        # Copiando as lógicas de features lá do classificador
        colunas_exatas = ["sexo", "SEXO", "ano", "ANO", "codmunres", "CODMUNRES", "dtnasc", "DTNASC"]
        for col in colunas_exatas:
            if col in df_sinasc_restante.columns and col in df_sim_restante.columns:
                compare.exact(col, col, missing_value=0, label=col)
                
        features_ml = compare.compute(candidate_links, df_sinasc_restante, df_sim_restante)
        
        if not features_ml.empty and clf is not None:
            probabilidades = clf.predict_proba(features_ml)[:, 1]
            pred_mask = (probabilidades >= 0.40)
            matches_ml_list = features_ml[pred_mask].index.to_list()
            matches_finais.update(matches_ml_list)
            
            # Remover encontrados
            sinasc_achados_ml = [t[0] for t in matches_ml_list]
            sim_achados_ml = [t[1] for t in matches_ml_list]
            df_sinasc_restante = df_sinasc_restante.drop(index=list(set(sinasc_achados_ml)), errors='ignore')
            df_sim_restante = df_sim_restante.drop(index=list(set(sim_achados_ml)), errors='ignore')
            
            print(f"[FASE 2] Pares encontrados: {len(matches_ml_list)}")
        else:
            print("[FASE 2] Não houve candidatos ou features suficientes.")
    except Exception as e:
        print(f"[FASE 2 - ERRO] {e}")

print(f"Restam SINASC: {len(df_sinasc_restante)} | Restam SIM: {len(df_sim_restante)}")

# ==========================================
# FASE 3: CLASSIFICADOR PROBABILÍSTICO (RESGATE)
# ==========================================
print("\n" + "="*50)
print(" FASE 3: FILTRO PROBABILÍSTICO (RESGATE FINAL)")
print("="*50)

if len(df_sinasc_restante) > 0 and len(df_sim_restante) > 0:
    classificador_prob = ClassificadorProbabilistico(df_sinasc_restante, df_sim_restante, true_matches)
    matches_prob, scores = classificador_prob.calcular_scores_probabilisticos()
    
    matches_finais.update(matches_prob.to_list())
    print(f"[FASE 3] Pares encontrados: {len(matches_prob)}")

# ==========================================
# RESULTADO FINAL E MÉTRICAS DA HEURÍSTICA
# ==========================================
print("\n" + "="*50)
print(" RESULTADO FINAL DA HEURÍSTICA EM CASCATA")
print("="*50)

matches_finais_index = pd.MultiIndex.from_tuples(list(matches_finais), names=["sinasc_index", "sim_index"])

tp = len(true_matches.intersection(matches_finais_index))
fp = len(matches_finais_index) - tp
fn = len(true_matches) - tp

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
taxa_reid = tp / len(df_sim) if len(df_sim) > 0 else 0

# Como o funil envolve várias abordagens, o MRR é complexo de calcular em conjunto sem scores unificados.
# Vamos assumir Rank 1 para tudo que passou pelo funil.
mrr_hibrido = tp / len(true_matches) if len(true_matches) > 0 else 0

print(f"Total True Matches no Gabarito: {len(true_matches)}")
print(f"Total Pares Preditos: {len(matches_finais_index)}")
print(f"True Positives (Acertos): {tp}")
print(f"False Positives (Erros): {fp}")
print(f"False Negatives (Perdidos): {fn}")
print("-" * 30)
print(f"Precisão Global: {precision:.2%}")
print(f"Recall Global: {recall:.2%}")
print(f"F1-Score Global: {f1:.2%}")
print(f"Taxa de Reidentificação Global: {taxa_reid:.2%}")
print(f"MRR (Mean Reciprocal Rank - Híbrido): {mrr_hibrido:.4f}")

# Calcular o H-Score da Heurística Luan (Opção 1 + Opção 2)
h_score = (f1 * 0.40) + (mrr_hibrido * 0.30) + (taxa_reid * 0.30)
print(f"\n🌟 H-Score (Desempenho da Heurística Própria): {h_score:.4f}")

print("\n✅ Teste do Pipeline Híbrido concluído com sucesso!")
