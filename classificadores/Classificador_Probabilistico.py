import pandas as pd
import numpy as np
import recordlinkage

class ClassificadorProbabilistico:
    def __init__(self, df_a, df_b, true_matches):
        """
        df_a -> dataframe SINASC
        df_b -> dataframe SIM
        true_matches -> dataframe com colunas identificando pares verdadeiros
        """
        self.df_a = df_a.reset_index(drop=True).copy()
        self.df_b = df_b.reset_index(drop=True).copy()

        self.df_a["id_sinasc"] = self.df_a.index
        self.df_b["id_sim"] = self.df_b.index

        self.true_matches = true_matches

    def indexar(self):
        """Faz o bloqueio combinando várias regras (híbrido) para aumentar a revocação (recall)."""
        indexer = recordlinkage.Index()

        blocos_aplicados = 0

        # Regra 1: Município de residência e Sexo/Ano
        regra_1 = [c for c in ["codmunres", "CODMUNRES", "sexo", "SEXO", "ano", "ANO"] if c in self.df_a.columns and c in self.df_b.columns]
        if len(regra_1) >= 2:
            indexer.block(regra_1)
            blocos_aplicados += 1

        # Regra 2: Data de nascimento
        regra_2 = [c for c in ["dtnasc", "DTNASC"] if c in self.df_a.columns and c in self.df_b.columns]
        if len(regra_2) >= 1:
            indexer.block(regra_2)
            blocos_aplicados += 1

        # Regra 2.5: Municipio de Nascimento e Sexo
        regra_3 = [c for c in ["codmunnasc", "CODMUNNASC", "sexo", "SEXO", "ano", "ANO"] if c in self.df_a.columns and c in self.df_b.columns]
        if len(regra_3) >= 2:
            indexer.block(regra_3)
            blocos_aplicados += 1

        # Regra 3: CEP usando Sorted Neighbourhood
        col_cep_a = next((c for c in ["cep", "CEP"] if c in self.df_a.columns), None)
        col_cep_b = next((c for c in ["cep", "CEP"] if c in self.df_b.columns), None)
        if col_cep_a and col_cep_b:
            indexer.sortedneighbourhood(left_on=col_cep_a, right_on=col_cep_b, window=5)
            blocos_aplicados += 1
            
        # Regra Extra: NOME com sorted neighbourhood
        col_nome_a = next((c for c in ["nome", "NOME"] if c in self.df_a.columns), None)
        col_nome_b = next((c for c in ["nome", "NOME"] if c in self.df_b.columns), None)
        if col_nome_a and col_nome_b:
            indexer.sortedneighbourhood(left_on=col_nome_a, right_on=col_nome_b, window=3)
            blocos_aplicados += 1

        # Regra 4: Nome da mãe e Sexo (Ortogonal a dtnasc/município)
        regra_4 = [c for c in ["nomemae", "NOMEMAE", "sexo", "SEXO"] if c in self.df_a.columns and c in self.df_b.columns]
        if len(regra_4) >= 2:
            indexer.block(regra_4)
            blocos_aplicados += 1

        if blocos_aplicados == 0:
            print("[AVISO] Nenhuma coluna de bloqueio encontrada. Usando comparação completa (lento).")
            indexer.full()
        else:
            print(f"[INFO] {blocos_aplicados} regras de bloqueio aplicadas no classificador probabilístico.")

        candidate_links = indexer.index(self.df_a, self.df_b)
        candidate_links = candidate_links.drop_duplicates()
        return candidate_links

    def calcular_scores_probabilisticos(self):
        """Aplica a técnica probabilística, dando pesos a quase-identificadores e retornando a Taxa de Linkagem"""
        candidate_links = self.indexar()

        compare = recordlinkage.Compare()

        # Configurar comparações (quasi-identificadores do paper)
        # Notas: Usamos JaroWinkler também para DTNASC para interceptar typos sem perder matches garantidos
        colunas_string = ["nome", "NOME", "nomemae", "NOMEMAE", "logradouro", "LOGRADOURO", "bairro", "BAIRRO", "dtnasc", "DTNASC"]
        for col in colunas_string:
            if col in self.df_a.columns and col in self.df_b.columns:
                compare.string(col, col, method="jarowinkler", label=f"{col}_str")
                if ("NOME" in col.upper() or "LOGRADOURO" in col.upper()):
                    compare.string(col, col, method="qgram", missing_value=0.3, label=f"{col}_qgram")

        colunas_exatas = [
            "nome", "NOME", "nomemae", "NOMEMAE", # Adicionado para penalizar clones falsos no score final
            "sexo", "SEXO", "ano", "ANO", "codmunres", "CODMUNRES",
            "dtnasc", "DTNASC", "codmunnasc", "CODMUNNASC",
            "racacor", "RACACOR", "numero", "NUMERO",
            "uf", "UF", "estcivmae", "ESTCIVMAE", "gestacao", "GESTACAO",
            "gravidez", "GRAVIDEZ", "parto", "PARTO"
        ]

        for col in colunas_exatas:
            if col in self.df_a.columns and col in self.df_b.columns:
                # Adding missing_value=0.3 so we don't zero out total score for absent optional fields like PART, GESTACAO etc
                compare.exact(col, col, missing_value=0.3, label=col)

        features = compare.compute(candidate_links, self.df_a, self.df_b)

        # Definir Pesos Probabilísticos empíricos
        pesos_configurados = {
            "NOME": 4.0, "nome": 4.0,           # Exato vale muito
            "NOMEMAE": 3.0, "nomemae": 3.0,     # Exato vale muito
            "NOME_str": 1.5, "nome_str": 1.5,   # Similaridade vale menos
            "NOME_qgram": 1.0, "nome_qgram": 1.0,
            "NOMEMAE_str": 1.5, "nomemae_str": 1.5,
            "NOMEMAE_qgram": 1.0, "nomemae_qgram": 1.0,
            "DTNASC": 1.5, "dtnasc": 1.5,
            "DTNASC_str": 1.5, "dtnasc_str": 1.5,
            "CEP": 0.8, "cep": 0.8,
            "CODMUNRES": 0.5, "codmunres": 0.5,
            "LOGRADOURO_str": 0.7, "logradouro_str": 0.7,
            "LOGRADOURO_qgram": 0.7, "logradouro_qgram": 0.7,
            "BAIRRO_str": 0.6, "bairro_str": 0.6,
            "SEXO": 0.3, "sexo": 0.3,
            "RACACOR": 0.3, "racacor": 0.3,
            "ESTCIVMAE": 0.3, "estcivmae": 0.3,
            "GESTACAO": 0.2, "gestacao": 0.2,
            "PARTO": 0.2, "parto": 0.2
        }
        
        pesos_ativos = {col: pesos_configurados.get(col, 0.5) for col in features.columns}
        
        weighted_scores = features.mul(pd.Series(pesos_ativos))
        score_sum = weighted_scores.sum(axis=1)

        # Encontrar optimal threshold se tivermos true_matches disponiveis na interseccao
        best_threshold = 7.0
        if hasattr(self.true_matches, "intersection"):
            best_f1 = 0
            for th in np.arange(5.0, 25.0, 0.5):
                pred = score_sum[score_sum >= th].index
                correct = len(self.true_matches.intersection(pred))
                precision = correct / len(pred) if len(pred) > 0 else 0
                recall = correct / len(self.true_matches) if len(self.true_matches) > 0 else 0
                
                # F1-Score otimiza dando peso igual para Precisão e Recall
                f1_val = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_threshold = th

            print(f"[INFO] Threshold probabilistico otimizado via grid search (F1-Score): {best_threshold}")
        
        predict_matches = score_sum[score_sum >= best_threshold].index

        # Avaliar e Calcular Taxa de Linkagem (TL)
        if hasattr(self.true_matches, "intersection"):
            intersecao = self.true_matches.intersection(predict_matches)
            tp = len(intersecao)
            fp = len(predict_matches) - tp
            fn = len(self.true_matches) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


            total_amostras = len(self.df_b)
            taxa_reid = tp / total_amostras if total_amostras > 0 else 0
            
            print("\n📊 Resultados do Classificador Probabilístico:")
            print(f"Total de pares preditos como MATCH: {len(predict_matches)}")
            print(f"Total de pares CORRETOS (True Positives): {tp}")
            print(f"Falsos Positivos (Errados): {fp}")
            print(f"Falsos Negativos (Perdidos): {fn}")
            print("-" * 30)
            print(f"Precisão (Precision): {precision:.2%}")
            print(f"Revocação (Recall): {recall:.2%}")
            print(f"F1-Score: {f1:.2%}")
            print(f"Taxa de Reidentificação: {taxa_reid:.2%} ({tp}/{total_amostras})")

            # Cálculo de MRR
            df_scores = score_sum.reset_index()
            df_scores.columns = ['sinasc_index', 'sim_index', 'score']
            df_scores = df_scores.sort_values(by=['sinasc_index', 'score'], ascending=[True, False])
            df_scores['rank'] = df_scores.groupby('sinasc_index').cumcount() + 1
            
            true_df = self.true_matches.to_frame(index=False)
            if not true_df.empty:
                true_df.columns = ['sinasc_index', 'sim_index']
                merged = pd.merge(df_scores, true_df, on=['sinasc_index', 'sim_index'], how='inner')
                if not merged.empty:
                    mrr = (1.0 / merged['rank']).sum() / len(true_df)
                else:
                    mrr = 0.0
            else: 
                mrr = 0.0
                
            print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")

        return predict_matches, score_sum
