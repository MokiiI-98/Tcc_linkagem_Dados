# Classificador_Supervisionado.py
import pandas as pd
import recordlinkage
from recordlinkage import LogisticRegressionClassifier
from sklearn.metrics import classification_report

class ClassificadorSupervisionado:
    def __init__(self, df_a, df_b):
        self.df_a = self.padronizar_dados(df_a.copy())
        self.df_b = self.padronizar_dados(df_b.copy())

    def padronizar_dados(self, df):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.lower()
        return df

    def indexar(self, blocking=None, strategy="auto", max_pairs=None):
        """
        Gera pares com estratégia configurável.
        - blocking: list[str] | str | None  → campos para bloco exato (ex.: ["data_nascimento","sexo"])
        - strategy: "auto" | "full"         → auto (lógica antiga) ou full index
        - max_pairs: int | None             → amostra aleatória de pares para acelerar testes
        """
        idx = recordlinkage.Index()

        if blocking:
            # Força blocking manual
            if isinstance(blocking, str):
                blocking = [blocking]
            idx.block(left_on=blocking, right_on=blocking)
            print(f"[INFO] Blocking manual: {blocking}")
        elif strategy == "full":
            idx.full()
            print("[INFO] FullIndex (sem blocking)")
        else:
            # estratégia automática (sua lógica original)
            if {'data_nascimento', 'sexo'}.issubset(self.df_a.columns) and \
               {'data_nascimento', 'sexo'}.issubset(self.df_b.columns):
                idx.block(left_on=['data_nascimento', 'sexo'],
                          right_on=['data_nascimento', 'sexo'])
                print("[INFO] Blocking automático: data_nascimento + sexo")
            elif 'data_nascimento' in self.df_a.columns and 'data_nascimento' in self.df_b.columns:
                idx.block(left_on='data_nascimento', right_on='data_nascimento')
                print("[INFO] Blocking automático: só data_nascimento")
            else:
                idx.full()
                print("[INFO] FullIndex (fallback)")

        pairs = idx.index(self.df_a, self.df_b)

        if len(pairs) == 0 and strategy != "full":
            # fallback extra
            print("[WARN] 0 pares com a estratégia escolhida. Tentando FullIndex…")
            idx = recordlinkage.Index(); idx.full()
            pairs = idx.index(self.df_a, self.df_b)
            print(f"[INFO] FullIndex gerou {len(pairs)} pares.")

        # Amostragem opcional para acelerar testes
        if max_pairs is not None and len(pairs) > max_pairs:
            pairs = pairs.to_series().sample(n=max_pairs, random_state=42).index
            print(f"[INFO] Amostrados {len(pairs)} pares de {max_pairs} (para teste).")

        print(f"[INFO] Total de pares gerados: {len(pairs)}")
        return pairs

    def comparar(self, pares, usar_cols=("data_nascimento","sexo","raca")):
        """Vetor de comparações; usa apenas as colunas existentes."""
        cmp = recordlinkage.Compare()
        for col in usar_cols:
            if col in self.df_a.columns and col in self.df_b.columns:
                cmp.exact(col, col, label=col)
        return cmp.compute(pares, self.df_a, self.df_b)

    def treinar_e_avaliar(self, pares, y_true):
        features = self.comparar(pares)
        clf = LogisticRegressionClassifier()
        clf.fit(features, y_true)

        y_pred = clf.predict(features)
        y_pred_series = pd.Series(0, index=features.index)
        y_pred_series.loc[y_pred] = 1

        y_true_series = pd.Series(0, index=features.index)
        y_true_series.loc[y_true] = 1

        print("\n[RELATÓRIO DE CLASSIFICAÇÃO]")
        print(classification_report(y_true_series, y_pred_series))
        return y_pred
