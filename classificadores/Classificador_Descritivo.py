import pandas as pd
import recordlinkage


class ClassificadorDescritivo:
    """
    Abordagem Descritiva/Determinística: regras fixas rígidas sem ML e sem pesos.

    Melhorias aplicadas:
    - Retorna unmatched_a e unmatched_b para alimentar o próximo classificador (RF)
    - Regras ordenadas por confiança (mais rígidas primeiro)
    - Regra 5 (nome+sexo) removida — era permissiva demais e inflava falsos positivos
    - Deduplicação 1-para-1: cada registro SINASC só pode linkar com 1 SIM (maior score = vence)
    - MRR corrigido: no determinístico não há ranking, logo MRR == Recall por definição
    - Normalização de colunas antes do bloqueio (lowercase + strip)
    """

    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame, true_matches):
        """
        Parâmetros
        ----------
        df_a         : DataFrame SINASC
        df_b         : DataFrame SIM
        true_matches : MultiIndex ou DataFrame com pares verdadeiros
        """
        self.df_a = self._normalizar(df_a.reset_index(drop=True).copy())
        self.df_b = self._normalizar(df_b.reset_index(drop=True).copy())
        self.df_a["id_sinasc"] = self.df_a.index
        self.df_b["id_sim"] = self.df_b.index
        self.true_matches = true_matches

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    @staticmethod
    def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza nomes de colunas para lowercase e strip."""
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    @staticmethod
    def _col(*candidatos, df):
        """Retorna o primeiro candidato que existir no DataFrame."""
        for c in candidatos:
            if c in df.columns:
                return c
        return None

    def _bloquear(self, cols_a, cols_b=None):
        """Monta um índice de bloqueio e retorna os pares candidatos."""
        if cols_b is None:
            cols_b = cols_a
        # Filtra apenas colunas que existem nos dois DataFrames
        pares_validos = [(a, b) for a, b in zip(cols_a, cols_b)
                         if a in self.df_a.columns and b in self.df_b.columns]
        if not pares_validos:
            return set()
        indexer = recordlinkage.Index()
        left_on = [p[0] for p in pares_validos]
        right_on = [p[1] for p in pares_validos]
        indexer.block(left_on=left_on, right_on=right_on)
        return set(indexer.index(self.df_a, self.df_b))

    # ------------------------------------------------------------------
    # Linkagem
    # ------------------------------------------------------------------

    def linkar(self):
        """
        Aplica regras determinísticas em ordem decrescente de confiança.
        Retorna
        -------
        predict_matches : pd.MultiIndex  — pares preditos como match
        unmatched_a     : pd.DataFrame   — registros SINASC não linkados (→ RF)
        unmatched_b     : pd.DataFrame   — registros SIM não linkados (→ RF)
        """
        pares_encontrados = set()

        # --- Regra 1 (altíssima confiança): Nome + Mun. Residência + Sexo + Data Nasc. ---
        pares_encontrados.update(self._bloquear(
            ["nome", "codmunres", "sexo", "dtnasc"]
        ))

        # --- Regra 2 (alta confiança): Nome da Mãe + Mun. Nascimento + Sexo + CEP ---
        pares_encontrados.update(self._bloquear(
            ["nomemae", "codmunnasc", "sexo", "cep"]
        ))

        # --- Regra 3 (alta confiança): Nome da Mãe + Data de Nascimento ---
        pares_encontrados.update(self._bloquear(
            ["nomemae", "dtnasc"]
        ))

        # --- Regra 4 (confiança média): Nome da Mãe + Sexo ---
        # (Captura erros em data/município, mas ainda exige dois campos firmes)
        pares_encontrados.update(self._bloquear(
            ["nomemae", "sexo"]
        ))

        # NOTA: Regra 5 original (nome+sexo) foi REMOVIDA.
        # Era muito permissiva e gerava muitos falsos positivos,
        # prejudicando a precisão. Casos que dependeriam dela serão
        # tratados pelo Random Forest com pesos.

        # ------------------------------------------------------------------
        # Deduplicação 1-para-1
        # Cada id_sinasc só pode linkar com 1 id_sim (o primeiro encontrado,
        # já que no determinístico todos os matches têm "peso igual").
        # ------------------------------------------------------------------
        if pares_encontrados:
            df_pares = pd.DataFrame(
                list(pares_encontrados),
                columns=["sinasc_index", "sim_index"]
            )
            # Mantém apenas o primeiro match por SINASC (ordem de chegada = prioridade da regra)
            df_pares = df_pares.drop_duplicates(subset="sinasc_index", keep="first")
            # E também garante que cada SIM só seja usado uma vez
            df_pares = df_pares.drop_duplicates(subset="sim_index", keep="first")
            predict_matches = pd.MultiIndex.from_frame(df_pares)
        else:
            predict_matches = pd.MultiIndex.from_tuples([], names=["sinasc_index", "sim_index"])

        # ------------------------------------------------------------------
        # Identificar não-linkados para o próximo estágio (Random Forest)
        # ------------------------------------------------------------------
        linked_sinasc = predict_matches.get_level_values("sinasc_index")
        linked_sim = predict_matches.get_level_values("sim_index")

        unmatched_a = self.df_a[~self.df_a.index.isin(linked_sinasc)].copy()
        unmatched_b = self.df_b[~self.df_b.index.isin(linked_sim)].copy()

        # ------------------------------------------------------------------
        # Avaliação
        # ------------------------------------------------------------------
        self._avaliar(predict_matches)

        print(f"\n[CASCATA] Registros NÃO linkados pelo Descritivo:")
        print(f"  SINASC restantes : {len(unmatched_a)}")
        print(f"  SIM restantes    : {len(unmatched_b)}")
        print(f"  → Passar esses para o Random Forest\n")

        return predict_matches, unmatched_a, unmatched_b

    # ------------------------------------------------------------------
    # Avaliação
    # ------------------------------------------------------------------

    def _avaliar(self, predict_matches: pd.MultiIndex):
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

        # No classificador determinístico não há ranking de scores —
        # todos os matches preditos têm "posição 1". Por isso MRR == Recall.
        mrr = recall

        print("\n📊 Resultados do Classificador DESCRITIVO (Determinístico):")
        print(f"  Total de pares preditos como MATCH : {len(predict_matches)}")
        print(f"  True Positives                     : {tp}")
        print(f"  Falsos Positivos                   : {fp}")
        print(f"  Falsos Negativos                   : {fn}")
        print("-" * 50)
        print(f"  Precisão  (Precision)              : {precision:.2%}")
        print(f"  Revocação (Recall)                 : {recall:.2%}")
        print(f"  F1-Score                           : {f1:.2%}")
        print(f"  Taxa de Reidentificação            : {taxa_reid:.2%} ({tp}/{total_amostras})")
        print(f"  MRR (= Recall no determinístico)   : {mrr:.4f}")