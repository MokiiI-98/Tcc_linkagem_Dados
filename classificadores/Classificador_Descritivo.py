import pandas as pd
import recordlinkage

class ClassificadorDescritivo:
    def __init__(self, df_a, df_b, true_matches):
        """
        Abordagem Descritiva/Determinística: Baseada em regras fixas rígidas (Sem Machine Learning e sem Pesos Probabilísticos)
        """
        self.df_a = df_a.reset_index(drop=True).copy()
        self.df_b = df_b.reset_index(drop=True).copy()
        self.df_a["id_sinasc"] = self.df_a.index
        self.df_b["id_sim"] = self.df_b.index
        
        self.true_matches = true_matches
        
    def linkar(self):
        """Aplica regras determinísticas (descritivas) """
        indexer = recordlinkage.Index()
        pares_encontrados = set()

        # Regra 1: Passos muito rígidos (Nome + Município de Residência + Sexo + Data de Nascimento Exata)
        regra_1 = [c for c in ["nome", "NOME", "codmunres", "CODMUNRES", "sexo", "SEXO", "dtnasc", "DTNASC"] if c in self.df_a.columns and c in self.df_b.columns]
        if any("nome" in c.lower() for c in regra_1) and len(regra_1) >= 4:
            indexer.block(regra_1)
            links_r1 = set(indexer.index(self.df_a, self.df_b))
            pares_encontrados.update(links_r1)
            
        # Regra 2: (Nome da Mãe + Município de Nascimento + Sexo + CEP)
        indexer_2 = recordlinkage.Index()
        regra_2 = [c for c in ["nomemae", "NOMEMAE", "codmunnasc", "CODMUNNASC", "sexo", "SEXO", "cep", "CEP"] if c in self.df_a.columns and c in self.df_b.columns]
        if any("nomemae" in c.lower() for c in regra_2) and len(regra_2) >= 3:
            indexer_2.block(regra_2)
            links_r2 = set(indexer_2.index(self.df_a, self.df_b))
            pares_encontrados.update(links_r2)

        # Regra 3: Nome exato da Mae + Data Nascimento exata
        indexer_3 = recordlinkage.Index()
        regra_3 = [c for c in ["nomemae", "NOMEMAE", "dtnasc", "DTNASC"] if c in self.df_a.columns and c in self.df_b.columns]
        if len(regra_3) >= 2:
            indexer_3.block(regra_3)
            links_r3 = set(indexer_3.index(self.df_a, self.df_b))
            pares_encontrados.update(links_r3)

        # Regra 4: Nome exato da Mae + Sexo (ignora datas de nascimento ou município com erro)
        indexer_4 = recordlinkage.Index()
        regra_4 = [c for c in ["nomemae", "NOMEMAE", "sexo", "SEXO"] if c in self.df_a.columns and c in self.df_b.columns]
        if len(regra_4) >= 2:
            indexer_4.block(regra_4)
            links_r4 = set(indexer_4.index(self.df_a, self.df_b))
            pares_encontrados.update(links_r4)
            
        # Regra 5: Nome Exato + Sexo (ignora o resto) - Regra mais relaxada para garantir recall 
        indexer_5 = recordlinkage.Index()
        regra_5 = [c for c in ["nome", "NOME", "sexo", "SEXO"] if c in self.df_a.columns and c in self.df_b.columns]
        if len(regra_5) >= 2:
            indexer_5.block(regra_5)
            links_r5 = set(indexer_5.index(self.df_a, self.df_b))
            pares_encontrados.update(links_r5)

        predict_matches = pd.MultiIndex.from_tuples(list(pares_encontrados), names=["sinasc_index", "sim_index"])

        if hasattr(self.true_matches, "intersection"):
            intersecao = self.true_matches.intersection(predict_matches)
            tp = len(intersecao)
            fp = len(predict_matches) - tp
            fn = len(self.true_matches) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Taxa de Reidentificacao = TP / Total de amostras
            total_amostras = len(self.df_b)
            taxa_reid = tp / total_amostras if total_amostras > 0 else 0
            
            print("\n📊 Resultados do Classificador DESCRITIVO (Determinístico):")
            print(f"Total de pares preditos como MATCH: {len(predict_matches)}")
            print(f"Total de pares CORRETOS (True Positives): {tp}")
            print(f"Falsos Positivos (Errados): {fp}")
            print(f"Falsos Negativos (Perdidos): {fn}")
            print("-" * 30)
            print(f"Precisão (Precision): {precision:.2%}")
            print(f"Revocação (Recall): {recall:.2%}")
            print(f"F1-Score: {f1:.2%}")
            print(f"Taxa de Reidentificação: {taxa_reid:.2%} ({tp}/{total_amostras})")
            
            # Para descritivo, o MRR é a própria proporção de pares corretos encontrados, já que não há ranking (tudo é rank 1).
            mrr = tp / len(self.true_matches) if len(self.true_matches) > 0 else 0
            print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")

        return predict_matches

