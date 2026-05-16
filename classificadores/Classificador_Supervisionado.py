import pandas as pd
import numpy as np
import recordlinkage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils import resample


class ClassificadorSupervisionado:
    """
    Classificador Random Forest — último estágio da cascata.

    Recebe apenas os registros NÃO linkados pelos estágios anteriores
    (Descritivo e Probabilístico), treinando sobre os casos mais difíceis.

    Melhorias aplicadas:
    - Recebe unmatched_a / unmatched_b da cascata (não a base completa)
    - true_matches filtrado: remove pares já linkados nos estágios anteriores
    - Threshold otimizado por grid search em F1 (igual ao Probabilístico)
    - Retorna predict_matches no mesmo formato dos outros classificadores
    - Avaliação global da cascata ao final
    - seed fixada em todas as operações aleatórias para reprodutibilidade
    """

    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame,
                 true_matches_completo: pd.MultiIndex,
                 matches_anteriores: pd.MultiIndex = None):
        """
        Parâmetros
        ----------
        df_a                  : DataFrame SINASC — apenas os não-linkados
        df_b                  : DataFrame SIM    — apenas os não-linkados
        true_matches_completo : MultiIndex com TODOS os pares verdadeiros do gabarito
        matches_anteriores    : MultiIndex com pares já linkados pelos estágios anteriores
                                (Descritivo + Probabilístico); usado para filtrar o gabarito
                                e calcular a avaliação global ao final
        """
        self.df_a = self._normalizar(df_a.reset_index(drop=True).copy())
        self.df_b = self._normalizar(df_b.reset_index(drop=True).copy())
        self.df_a["id_sinasc"] = self.df_a.index
        self.df_b["id_sim"]    = self.df_b.index

        self.true_matches_completo = true_matches_completo
        self.matches_anteriores    = matches_anteriores if matches_anteriores is not None \
                                     else pd.MultiIndex.from_tuples([], names=["sinasc_index", "sim_index"])

        # Gabarito filtrado: apenas pares que ainda não foram linkados
        # Os índices aqui são os do df_a/df_b recebidos (após reset_index),
        # então precisamos mapear os índices originais para os novos
        self._map_a = {old: new for new, old in enumerate(df_a.index)}
        self._map_b = {old: new for new, old in enumerate(df_b.index)}

        pares_filtrados = []
        for sinasc_orig, sim_orig in true_matches_completo:
            novo_a = self._map_a.get(sinasc_orig)
            novo_b = self._map_b.get(sim_orig)
            if novo_a is not None and novo_b is not None:
                pares_filtrados.append((novo_a, novo_b))

        self.true_matches = pd.MultiIndex.from_tuples(
            pares_filtrados, names=["sinasc_index", "sim_index"]
        ) if pares_filtrados else pd.MultiIndex.from_tuples(
            [], names=["sinasc_index", "sim_index"]
        )

        print(f"[RF] Registros recebidos — SINASC: {len(self.df_a)} | SIM: {len(self.df_b)}")
        print(f"[RF] Pares verdadeiros neste estágio: {len(self.true_matches)}")

    @staticmethod
    def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Indexação (bloqueio híbrido)
    # ------------------------------------------------------------------

    def indexar(self) -> pd.MultiIndex:
        indexer = recordlinkage.Index()
        blocos_aplicados = 0
        cols_a = set(self.df_a.columns)
        cols_b = set(self.df_b.columns)

        def cols_existentes(*candidatos):
            return [c for c in candidatos if c in cols_a and c in cols_b]

        def tem(*candidatos):
            return any(c in cols_a and c in cols_b for c in candidatos)

        # Regra 1: Município de residência + Sexo
        r1 = cols_existentes("codmunres", "sexo")
        if len(r1) >= 2:
            indexer.block(r1); blocos_aplicados += 1

        # Regra 2: Data de nascimento
        r2 = cols_existentes("dtnasc")
        if r2:
            indexer.block(r2); blocos_aplicados += 1

        # Regra 3: Município de nascimento + Sexo
        r3 = cols_existentes("codmunnasc", "sexo")
        if len(r3) >= 2:
            indexer.block(r3); blocos_aplicados += 1

        # Regra 4: CEP — Sorted Neighbourhood
        if tem("cep"):
            indexer.sortedneighbourhood("cep", "cep", window=5)
            blocos_aplicados += 1

        # Regra 5: Nome — Sorted Neighbourhood (mais amplo para capturar typos)
        if tem("nome"):
            indexer.sortedneighbourhood("nome", "nome", window=5)
            blocos_aplicados += 1

        # Regra 6: Nome da mãe + Sexo
        r6 = cols_existentes("nomemae", "sexo")
        if len(r6) >= 2:
            indexer.block(r6); blocos_aplicados += 1

        if blocos_aplicados == 0:
            print("[AVISO] Nenhuma coluna de bloqueio. Usando comparação completa.")
            indexer.full()
        else:
            print(f"[RF] {blocos_aplicados} regras de bloqueio aplicadas.")

        candidate_links = indexer.index(self.df_a, self.df_b).drop_duplicates()
        print(f"[RF] Pares candidatos gerados: {len(candidate_links):,}")
        return candidate_links

    # ------------------------------------------------------------------
    # Treino e avaliação
    # ------------------------------------------------------------------

    def treinar_e_avaliar(self) -> pd.MultiIndex:
        """
        Treina o Random Forest sobre os candidatos do estágio residual.

        Retorna
        -------
        predict_matches : pd.MultiIndex — pares linkados pelo RF
        """
        candidate_links = self.indexar()

        # --- Features de comparação ---
        compare = recordlinkage.Compare()

        campos_texto = ["nome", "nomemae", "dtnasc", "logradouro", "bairro"]
        for col in campos_texto:
            if col in self.df_a.columns and col in self.df_b.columns:
                compare.string(col, col, method="jarowinkler",
                               missing_value=0.0, label=f"{col}_jw")
                compare.string(col, col, method="qgram",
                               missing_value=0.0, label=f"{col}_qg")

        campos_exatos = ["nome", "nomemae", "dtnasc", "sexo", "codmunres",
                         "codmunnasc", "ano", "cep", "racacor", "estcivmae",
                         "gestacao", "gravidez", "parto"]
        for col in campos_exatos:
            if col in self.df_a.columns and col in self.df_b.columns:
                compare.exact(col, col, missing_value=0, label=col)

        features = compare.compute(candidate_links, self.df_a, self.df_b)

        if len(self.true_matches) == 0:
            print("[RF] Nenhum par verdadeiro restante — nada a linkar neste estágio.")
            return pd.MultiIndex.from_tuples([], names=["sinasc_index", "sim_index"])

        intersecao = self.true_matches.intersection(features.index)
        print(f"[RF] True matches no espaço de candidatos: {len(intersecao)}/{len(self.true_matches)}")

        if len(intersecao) == 0:
            print("[RF] ⚠ Nenhum true match nos candidatos — verifique o bloqueio.")
            return pd.MultiIndex.from_tuples([], names=["sinasc_index", "sim_index"])

        # --- Labels ---
        y = pd.Series(0, index=features.index)
        y.loc[intersecao] = 1

        X_pos = features.loc[y[y == 1].index]
        X_neg = features.loc[y[y == 0].index]

        # Undersampling adaptativo: min(negativos reais, 5x positivos, 10.000)
        n_neg = min(len(X_neg), max(5 * len(X_pos), 500), 10_000)
        X_neg_sample = resample(X_neg, n_samples=n_neg, random_state=42)

        X = pd.concat([X_pos, X_neg_sample])
        y_bal = pd.Series(
            [1] * len(X_pos) + [0] * len(X_neg_sample),
            index=X.index
        )

        # --- Split treino / teste estratificado ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_bal, test_size=0.3, random_state=42, stratify=y_bal
        )

        # --- Modelo ---
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight={0: 1, 1: 3},
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # --- Threshold ótimo por grid search em F1 (sobre validação) ---
        proba_test = clf.predict_proba(X_test)[:, 1]
        best_th, best_f1 = 0.5, 0.0
        for th in np.arange(0.30, 0.75, 0.02):
            y_pred_th = (proba_test >= th).astype(int)
            tp = ((y_pred_th == 1) & (y_test == 1)).sum()
            fp = ((y_pred_th == 1) & (y_test == 0)).sum()
            fn = ((y_pred_th == 0) & (y_test == 1)).sum()
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1, best_th = f1, th

        print(f"[RF] Threshold ótimo (F1={best_f1:.4f}): {best_th:.2f}")

        y_pred_final = (proba_test >= best_th).astype(int)
        print("\n📊 Relatório de classificação — conjunto de teste interno (RF):")
        print(classification_report(y_test, y_pred_final, zero_division=0))

        # --- Predição sobre todos os candidatos ---
        proba_full = clf.predict_proba(features)[:, 1]
        df_pred = pd.DataFrame({"score": proba_full}, index=features.index).reset_index()
        df_pred.columns = ["sinasc_index", "sim_index", "score"]

        # Filtra pelo threshold e aplica deduplicação 1-para-1 por maior score
        df_pred = df_pred[df_pred["score"] >= best_th]
        df_pred = df_pred.sort_values("score", ascending=False)
        df_pred = df_pred.drop_duplicates(subset="sinasc_index", keep="first")
        df_pred = df_pred.drop_duplicates(subset="sim_index",    keep="first")

        predict_matches = pd.MultiIndex.from_frame(df_pred[["sinasc_index", "sim_index"]])

        # --- MRR ---
        df_scores = pd.DataFrame({"score": proba_full}, index=features.index).reset_index()
        df_scores.columns = ["sinasc_index", "sim_index", "score"]
        df_scores = df_scores.sort_values(["sinasc_index", "score"], ascending=[True, False])
        df_scores["rank"] = df_scores.groupby("sinasc_index").cumcount() + 1
        true_df = self.true_matches.to_frame(index=False)
        true_df.columns = ["sinasc_index", "sim_index"]
        merged = pd.merge(df_scores, true_df, on=["sinasc_index", "sim_index"], how="inner")
        mrr = (1.0 / merged["rank"]).sum() / len(true_df) if not merged.empty else 0.0

        # --- Avaliação deste estágio ---
        intersecao_pred = self.true_matches.intersection(predict_matches)
        tp = len(intersecao_pred)
        fp = len(predict_matches) - tp
        fn = len(self.true_matches) - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        total = len(self.df_b)
        taxa  = tp / total if total > 0 else 0

        print("\n📊 Resultados do Classificador RANDOM FOREST (estágio residual):")
        print(f"  Total de pares preditos como MATCH : {len(predict_matches)}")
        print(f"  True Positives                     : {tp}")
        print(f"  Falsos Positivos                   : {fp}")
        print(f"  Falsos Negativos                   : {fn}")
        print("-" * 50)
        print(f"  Precisão  (Precision)              : {p:.2%}")
        print(f"  Revocação (Recall)                 : {r:.2%}")
        print(f"  F1-Score                           : {f1:.2%}")
        print(f"  Taxa de Reidentificação (estágio)  : {taxa:.2%} ({tp}/{total})")
        print(f"  MRR (Mean Reciprocal Rank)         : {mrr:.4f}")

        # Não-linkados residuais
        linked_sinasc = predict_matches.get_level_values("sinasc_index")
        linked_sim    = predict_matches.get_level_values("sim_index")
        unmatched_a   = self.df_a[~self.df_a.index.isin(linked_sinasc)]
        unmatched_b   = self.df_b[~self.df_b.index.isin(linked_sim)]
        print(f"\n[CASCATA] Registros NÃO linkados pelo RF:")
        print(f"  SINASC residuais : {len(unmatched_a)}")
        print(f"  SIM residuais    : {len(unmatched_b)}")

        return predict_matches