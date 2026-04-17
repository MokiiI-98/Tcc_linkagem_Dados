import pandas as pd
import recordlinkage
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class ClassificadorSupervisionado:
    def __init__(self, df_a, df_b):
        self.df_a = df_a.copy()
        self.df_b = df_b.copy()

        # Pré-processamento inicial
        numeric_cols = ["PESO", "IDADEMAE", "QTDFILVIVO", "QTDFILMORT"]
        for col in numeric_cols:
            if col in self.df_a.columns:
                self.df_a[col] = pd.to_numeric(self.df_a[col], errors="coerce")
            if col in self.df_b.columns:
                self.df_b[col] = pd.to_numeric(self.df_b[col], errors="coerce")

        # Converter datas
        if "DTNASC" in self.df_a.columns:
            self.df_a["DTNASC"] = pd.to_datetime(self.df_a["DTNASC"], errors="coerce")
        if "DTNASC" in self.df_b.columns:
            self.df_b["DTNASC"] = pd.to_datetime(self.df_b["DTNASC"], errors="coerce")

    def indexar(self, mode="auto"):
        indexer = recordlinkage.Index()

        if mode == "auto":
            try:
                if "SEXO" in self.df_a.columns and "SEXO" in self.df_b.columns:
                    if "DTNASC" in self.df_a.columns and "DTNASC" in self.df_b.columns:
                        if pd.api.types.is_datetime64_any_dtype(self.df_a["DTNASC"]):
                            self.df_a["ANO_NASC"] = self.df_a["DTNASC"].dt.year
                        if pd.api.types.is_datetime64_any_dtype(self.df_b["DTNASC"]):
                            self.df_b["ANO_NASC"] = self.df_b["DTNASC"].dt.year

                        if "ANO_NASC" in self.df_a.columns and "ANO_NASC" in self.df_b.columns:
                            indexer.block(left_on=["SEXO", "ANO_NASC"], right_on=["SEXO", "ANO_NASC"])
                        else:
                            indexer.block("SEXO")
                    else:
                        indexer.block("SEXO")
                else:
                    indexer.full()

                pares = indexer.index(self.df_a, self.df_b)
                print(f"[INFO] Total de pares gerados (SEXO + ANO_NASC): {len(pares)}")
                return pares

            except Exception as e:
                print(f"[WARN] Erro no blocking: {e}")
                print("[INFO] Usando FullIndex como fallback")
                indexer = recordlinkage.Index()
                indexer.full()
                pares = indexer.index(self.df_a, self.df_b)
                print(f"[INFO] Total de pares gerados (full): {len(pares)}")
                return pares

        elif mode == "full":
            indexer.full()
            pares = indexer.index(self.df_a, self.df_b)
            print(f"[INFO] Total de pares gerados (full): {len(pares)}")
            return pares

        else:
            raise ValueError("Mode deve ser 'auto' ou 'full'")

    def comparar(self, pares):
        compare = recordlinkage.Compare()

        if "SEXO" in self.df_a.columns:
            compare.exact("SEXO", "SEXO", label="sexo")

        if "PESO" in self.df_a.columns:
            compare.numeric("PESO", "PESO", method="gauss", offset=200, scale=100, label="peso")

        if "RACACOR" in self.df_a.columns:
            self.df_a["RACACOR"] = self.df_a["RACACOR"].fillna("").astype(str)
            self.df_b["RACACOR"] = self.df_b["RACACOR"].fillna("").astype(str)
            compare.string("RACACOR", "RACACOR", method="jarowinkler", threshold=0.85, label="raca_cor")

        if "IDADEMAE" in self.df_a.columns:
            compare.exact("IDADEMAE", "IDADEMAE", label="idade_mae")

        print(f"[INFO] Comparadores configurados")
        return compare.compute(pares, self.df_a, self.df_b)

    def treinar_e_avaliar(self, pares, true_matches_idx, ratio=10):
        features = self.comparar(pares)
        y_true = features.index.isin(true_matches_idx).astype(int)

        # Separar matches e não-matches
        matches = features[y_true == 1]
        non_matches = features[y_true == 0]

        print(f"[INFO] Matches reais: {len(matches)}, Não-matches: {len(non_matches)}")

        if len(matches) == 0:
            print("[ERRO] Nenhum match real encontrado. Não é possível treinar.")
            return

        # Amostrar não-matches proporcionalmente
        n_non_matches = min(len(non_matches), len(matches) * ratio)
        non_matches_sampled = non_matches.sample(n=n_non_matches, random_state=42)

        # Dataset balanceado
        X_bal = pd.concat([matches, non_matches_sampled])
        y_bal = [1]*len(matches) + [0]*len(non_matches_sampled)

        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
        )

        # 🔹 Classificador ajustado para scikit-learn
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print("\n[RELATÓRIO DE CLASSIFICAÇÃO - TESTE]")
        print(classification_report(y_test, y_pred, digits=2, zero_division=0))
