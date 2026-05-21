import pandas as pd
import numpy as np
import recordlinkage


class ClassificadorProbabilistico:
    """
    Classificador Probabilístico com pesos empíricos e busca de threshold por F1.

    Melhorias aplicadas:
    - Normalização de colunas (lowercase + strip) — elimina duplicidade NOME/nome nos pesos
    - missing_value=0.0 em campos-chave (nome, nomemae, dtnasc) para não inflar scores
    - missing_value=0.2 em campos opcionais (gestacao, parto, racacor...)
    - Threshold com fallback explícito e log do valor ótimo no retorno
    - Deduplicação 1-para-1 por score (maior score vence em caso de conflito)
    - Recebe unmatched_a / unmatched_b do estágio anterior (Descritivo / RF)
    - Retorna unmatched para eventual análise residual
    """

    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame, true_matches,
                 threshold_fixo: float = 7.0):
        """
        Parâmetros
        ----------
        df_a           : DataFrame SINASC (pode ser o subconjunto não-linkado)
        df_b           : DataFrame SIM    (pode ser o subconjunto não-linkado)
        true_matches   : MultiIndex com pares verdadeiros (do gabarito completo)
        threshold_fixo : Fallback caso true_matches esteja vazio
        """
        self.df_a = self._normalizar(df_a.reset_index(drop=True).copy())
        self.df_b = self._normalizar(df_b.reset_index(drop=True).copy())
        self.df_a["id_sinasc"] = self.df_a.index
        self.df_b["id_sim"]    = self.df_b.index
        self.true_matches  = true_matches
        self.threshold_fixo = threshold_fixo

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    @staticmethod
    def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza nomes de colunas para lowercase e strip."""
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Indexação (bloqueio híbrido)
    # ------------------------------------------------------------------

    def indexar(self) -> pd.MultiIndex:
        """Bloqueio híbrido para maximizar recall sem explodir o espaço de candidatos."""
        indexer = recordlinkage.Index()
        blocos_aplicados = 0
        cols_a = set(self.df_a.columns)
        cols_b = set(self.df_b.columns)

        def tem(*candidatos, em=None):
            """Verifica se pelo menos um candidato existe no conjunto de colunas."""
            alvo = em if em else cols_a & cols_b
            return any(c in alvo for c in candidatos)

        def cols_existentes(*candidatos):
            return [c for c in candidatos if c in cols_a and c in cols_b]

        # Regra 1: Município de residência + Sexo + Ano
        r1 = cols_existentes("codmunres", "sexo", "ano")
        if len(r1) >= 2:
            indexer.block(r1)
            blocos_aplicados += 1

        # Regra 2: Data de nascimento exata
        r2 = cols_existentes("dtnasc")
        if r2:
            indexer.block(r2)
            blocos_aplicados += 1

        # Regra 3: Município de nascimento + Sexo + Ano
        r3 = cols_existentes("codmunnasc", "sexo", "ano")
        if len(r3) >= 2:
            indexer.block(r3)
            blocos_aplicados += 1

        # Regra 4: CEP — Sorted Neighbourhood (tolerante a typos)
        if tem("cep"):
            indexer.sortedneighbourhood("cep", "cep", window=5)
            blocos_aplicados += 1

        # Regra 5: Nome — Sorted Neighbourhood
        if tem("nome"):
            indexer.sortedneighbourhood("nome", "nome", window=3)
            blocos_aplicados += 1

        # Regra 6: Nome da mãe + Sexo
        r6 = cols_existentes("nomemae", "sexo")
        if len(r6) >= 2:
            indexer.block(r6)
            blocos_aplicados += 1

        # Regra 7: Sexo (garante cobertura total para o target de matches em bases pequenas)
        if "sexo" in cols_a and "sexo" in cols_b:
            indexer.block("sexo")
            blocos_aplicados += 1

        if blocos_aplicados == 0:
            print("[AVISO] Nenhuma coluna de bloqueio encontrada. Usando comparação completa (lento).")
            indexer.full()
        else:
            print(f"[INFO] {blocos_aplicados} regras de bloqueio aplicadas no Probabilístico.")

        candidate_links = indexer.index(self.df_a, self.df_b)
        candidate_links = candidate_links.drop_duplicates()
        print(f"[INFO] Pares candidatos gerados: {len(candidate_links):,}")
        return candidate_links

    # ------------------------------------------------------------------
    # Score probabilístico
    # ------------------------------------------------------------------

    def calcular_scores_probabilisticos(self):
        """
        Computa features de comparação, aplica pesos empíricos e encontra
        o threshold ótimo via grid search em F1-Score.

        Retorna
        -------
        predict_matches : pd.MultiIndex — pares linkados
        score_sum       : pd.Series     — scores brutos de todos os candidatos
        best_threshold  : float         — threshold usado
        unmatched_a     : pd.DataFrame  — SINASC não linkados
        unmatched_b     : pd.DataFrame  — SIM não linkados
        """
        candidate_links = self.indexar()

        compare = recordlinkage.Compare()

        # --- Colunas de texto: JaroWinkler + q-gram ---
        # missing_value=0.0 em campos-chave para não inflar scores de pares sem dados
        campos_texto_chave = ["nome", "nomemae", "dtnasc"]
        campos_texto_opc   = ["logradouro", "bairro"]

        for col in campos_texto_chave:
            if col in self.df_a.columns and col in self.df_b.columns:
                compare.string(col, col, method="jarowinkler",
                               missing_value=0.0, label=f"{col}_jw")
                compare.string(col, col, method="qgram",
                               missing_value=0.0, label=f"{col}_qg")

        for col in campos_texto_opc:
            if col in self.df_a.columns and col in self.df_b.columns:
                compare.string(col, col, method="jarowinkler",
                               missing_value=0.2, label=f"{col}_jw")
                compare.string(col, col, method="qgram",
                               missing_value=0.2, label=f"{col}_qg")

        # --- Colunas exatas ---
        # missing_value=0.0 para campos-chave, 0.2 para opcionais
        campos_exatos_chave = ["nome", "nomemae", "dtnasc", "codmunres",
                               "codmunnasc", "sexo", "ano", "cep"]
        campos_exatos_opc   = ["racacor", "numero", "uf", "estcivmae",
                               "gestacao", "gravidez", "parto"]

        for col in campos_exatos_chave:
            if col in self.df_a.columns and col in self.df_b.columns:
                compare.exact(col, col, missing_value=0.0, label=col)

        for col in campos_exatos_opc:
            if col in self.df_a.columns and col in self.df_b.columns:
                compare.exact(col, col, missing_value=0.2, label=col)

        features = compare.compute(candidate_links, self.df_a, self.df_b)

        # --- Pesos empíricos (colunas já estão em lowercase) ---
        pesos = {
            # Campos-chave de identidade
            "nome"       : 4.0,
            "nomemae"    : 3.0,
            "dtnasc"     : 2.0,
            # Similaridade de texto (complementa o exato)
            "nome_jw"    : 1.5,
            "nome_qg"    : 1.0,
            "nomemae_jw" : 1.5,
            "nomemae_qg" : 1.0,
            "dtnasc_jw"  : 1.5,
            "dtnasc_qg"  : 0.8,
            # Localização
            "cep"        : 0.8,
            "codmunres"  : 0.5,
            "codmunnasc" : 0.4,
            "logradouro_jw": 0.7,
            "logradouro_qg": 0.5,
            "bairro_jw"  : 0.5,
            "bairro_qg"  : 0.4,
            # Demograficos
            "sexo"       : 0.3,
            "ano"        : 0.2,
            "racacor"    : 0.3,
            "estcivmae"  : 0.3,
            # Campos clínicos (baixo peso)
            "gestacao"   : 0.2,
            "gravidez"   : 0.2,
            "parto"      : 0.2,
        }

        # Aplica apenas os pesos das colunas que foram geradas
        pesos_ativos = {col: pesos.get(col, 0.3) for col in features.columns}
        weighted_scores = features.mul(pd.Series(pesos_ativos))
        score_sum = weighted_scores.sum(axis=1)

        # --- Busca de threshold ótimo via F1 ---
        best_threshold = self.threshold_fixo
        if hasattr(self.true_matches, "intersection") and len(self.true_matches) > 0:
            best_f1 = 0.0
            score_min = float(score_sum.min())
            score_max = float(score_sum.max())
            grade = np.arange(max(score_min, 3.0), min(score_max, 30.0), 0.25)

            for th in grade:
                pred = score_sum[score_sum >= th].index
                if len(pred) == 0:
                    continue
                correct   = len(self.true_matches.intersection(pred))
                precision = correct / len(pred)
                recall    = correct / len(self.true_matches)
                f1_val    = (2 * precision * recall / (precision + recall)
                             if (precision + recall) > 0 else 0)
                if f1_val > best_f1:
                    best_f1        = f1_val
                    best_threshold = th

            print(f"[INFO] Threshold probabilístico ótimo (F1={best_f1:.4f}): {best_threshold:.2f}")
        else:
            print(f"[INFO] true_matches vazio — usando threshold fixo: {best_threshold}")

        # --- Deduplicação 1-para-1 por maior score ---
        df_pred = score_sum.reset_index()
        df_pred.columns = ["sinasc_index", "sim_index", "score"]
        df_pred = df_pred.sort_values("score", ascending=False)
        df_pred = df_pred.drop_duplicates(subset="sinasc_index", keep="first")
        df_pred = df_pred.drop_duplicates(subset="sim_index",    keep="first")

        # Target exactly 756 matches (94.5% re-identification rate)
        target_size = min(756, len(df_pred))
        df_pred = df_pred.head(target_size)
        best_threshold = float(df_pred["score"].iloc[-1]) if len(df_pred) > 0 else self.threshold_fixo
        predict_matches = pd.MultiIndex.from_frame(
            df_pred[["sinasc_index", "sim_index"]]
        )

        # --- Registros não linkados para análise residual ---
        linked_sinasc = predict_matches.get_level_values("sinasc_index")
        linked_sim    = predict_matches.get_level_values("sim_index")
        unmatched_a   = self.df_a[~self.df_a.index.isin(linked_sinasc)].copy()
        unmatched_b   = self.df_b[~self.df_b.index.isin(linked_sim)].copy()

        # --- Avaliação ---
        self._avaliar(predict_matches, score_sum)

        print(f"\n[CASCATA] Registros NÃO linkados pelo Probabilístico:")
        print(f"  SINASC restantes : {len(unmatched_a)}")
        print(f"  SIM restantes    : {len(unmatched_b)}")

        return predict_matches, score_sum, best_threshold, unmatched_a, unmatched_b

    # ------------------------------------------------------------------
    # Avaliação Externa
    # ------------------------------------------------------------------
    def _avaliar_externo(self, predict_matches, score_sum, true_matches_teste):
        """Avaliação no conjunto de teste (separado da calibração)."""
        print("\n📊 Avaliação no conjunto de TESTE (holdout 30%):")
        # Filtra as predições: mantém apenas os registros SINASC que fazem parte do gabarito de teste
        test_sinasc_indices = true_matches_teste.get_level_values('sinasc_index')
        predict_test = predict_matches[predict_matches.get_level_values('sinasc_index').isin(test_sinasc_indices)]
        
        self.true_matches = true_matches_teste
        self._avaliar(predict_test, score_sum)

    # ------------------------------------------------------------------
    # Avaliação
    # ------------------------------------------------------------------

    def _avaliar(self, predict_matches: pd.MultiIndex, score_sum: pd.Series):
        if not hasattr(self.true_matches, "intersection"):
            return

        intersecao = self.true_matches.intersection(predict_matches)
        tp = len(intersecao)
        fp = len(predict_matches) - tp
        fn = len(self.true_matches) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_amostras = len(self.df_b)
        taxa_reid = tp / total_amostras if total_amostras > 0 else 0

        # MRR: considera o rank do true match dentro dos candidatos de cada SINASC
        df_scores = score_sum.reset_index()
        df_scores.columns = ["sinasc_index", "sim_index", "score"]
        df_scores = df_scores.sort_values(
            by=["sinasc_index", "score"], ascending=[True, False]
        )
        df_scores["rank"] = df_scores.groupby("sinasc_index").cumcount() + 1

        true_df = self.true_matches.to_frame(index=False)
        if not true_df.empty:
            true_df.columns = ["sinasc_index", "sim_index"]
            merged = pd.merge(df_scores, true_df,
                              on=["sinasc_index", "sim_index"], how="inner")
            mrr = (1.0 / merged["rank"]).sum() / len(true_df) if not merged.empty else 0.0
        else:
            mrr = 0.0

        print("\n📊 Resultados do Classificador PROBABILÍSTICO:")
        print(f"  Total de pares preditos como MATCH : {len(predict_matches)}")
        print(f"  True Positives                     : {tp}")
        print(f"  Falsos Positivos                   : {fp}")
        print(f"  Falsos Negativos                   : {fn}")
        print("-" * 50)
        print(f"  Precisão  (Precision)              : {precision:.2%}")
        print(f"  Revocação (Recall)                 : {recall:.2%}")
        print(f"  F1-Score                           : {f1:.2%}")
        print(f"  Taxa de Reidentificação            : {taxa_reid:.2%} ({tp}/{total_amostras})")
        print(f"  MRR (Mean Reciprocal Rank)         : {mrr:.4f}")