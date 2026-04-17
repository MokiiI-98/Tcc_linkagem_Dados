import pandas as pd
import recordlinkage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample


class ClassificadorSupervisionado:
    def __init__(self, df_a, df_b, true_matches):
        """
        df_a -> dataframe SINASC
        df_b -> dataframe SIM
        true_matches -> dataframe com colunas ["id_sinasc", "id_sim"]
        """

        # Garante que os índices sejam consistentes e reiniciados
        self.df_a = df_a.reset_index(drop=True).copy()
        self.df_b = df_b.reset_index(drop=True).copy()

        self.df_a["id_sinasc"] = self.df_a.index
        self.df_b["id_sim"] = self.df_b.index

        # Converte os true_matches em MultiIndex
        self.true_matches = pd.MultiIndex.from_frame(true_matches)

    def indexar(self):
        """Faz o bloqueio (reduz combinações) usando múltiplos blocos para melhorar o recall."""
        indexer = recordlinkage.Index()

        blocos_aplicados = 0

        # Regra 1: Município de residência e Sexo/Ano
        regra_1 = [c for c in ["codmunres", "CODMUNRES", "sexo", "SEXO", "ano", "ANO"] if c in self.df_a.columns and c in self.df_b.columns]
        if regra_1:
            indexer.block(regra_1)
            blocos_aplicados += 1

        # Regra 2: Data de nascimento
        regra_2 = [c for c in ["dtnasc", "DTNASC"] if c in self.df_a.columns and c in self.df_b.columns]
        if regra_2:
            indexer.block(regra_2)
            blocos_aplicados += 1

        # Regra 2.5: Municipio de Nascimento e Sexo
        regra_3 = [c for c in ["codmunnasc", "CODMUNNASC", "sexo", "SEXO", "ano", "ANO"] if c in self.df_a.columns and c in self.df_b.columns]
        if regra_3:
            indexer.block(regra_3)
            blocos_aplicados += 1

        # Regra 3: CEP usando Sorted Neighbourhood para tolerar pequenos desvios morfológicos/typos
        col_cep_a = next((c for c in ["cep", "CEP"] if c in self.df_a.columns), None)
        col_cep_b = next((c for c in ["cep", "CEP"] if c in self.df_b.columns), None)
        if col_cep_a and col_cep_b:
            indexer.sortedneighbourhood(left_on=col_cep_a, right_on=col_cep_b, window=5)
            blocos_aplicados += 1

        # Regra 4: Nome da mãe e Sexo (Ortogonal a dtnasc/município)
        regra_4 = [c for c in ["nomemae", "NOMEMAE", "sexo", "SEXO"] if c in self.df_a.columns and c in self.df_b.columns]
        if regra_4:
            indexer.block(regra_4)
            blocos_aplicados += 1

        if blocos_aplicados == 0:
            print("[AVISO] Nenhuma coluna de bloqueio encontrada. Usando comparação completa (lento).")
            # Se não há regras, usa block em colunas vazias gerando produto cartesiano
            indexer.full()
        else:
            print(f"[INFO] {blocos_aplicados} regras de bloqueio aplicadas.")

        candidate_links = indexer.index(self.df_a, self.df_b)
        
        # Remove duplicatas oriundas de múltiplos blocos (mesmo par retornado em regras diferentes)
        candidate_links = candidate_links.drop_duplicates()
        return candidate_links

    def treinar_e_avaliar(self):
        """Treina um classificador supervisionado agressivo para True Positives"""
        # Gera os pares candidatos
        candidate_links = self.indexar()

        # Extrai features de comparação
        compare = recordlinkage.Compare()
        
        # Variáveis string (textuais) e tipográficas
        colunas_string = ["nome", "NOME", "nomemae", "NOMEMAE", "logradouro", "LOGRADOURO", "bairro", "BAIRRO", "dtnasc", "DTNASC"]
        for col in colunas_string:
            if col in self.df_a.columns and col in self.df_b.columns:
                # Usar Jaro-Winkler ajuda muito a encontrar matches onde a data de nascimento tem 1 dígito errado (ex: 01012024 vs 01102024)
                compare.string(col, col, method="jarowinkler", label=f"{col}_str")
                
                # Para nomes, usa qgram também para capturar palavras invertidas (tokens)
                if ("NOME" in col.upper() or "LOGRADOURO" in col.upper()):
                    compare.string(col, col, method="qgram", label=f"{col}_qgram")

        # Variáveis exatas (categóricas, datas, códigos)
        colunas_exatas = [
            "sexo", "SEXO", "ano", "ANO", "codmunres", "CODMUNRES",
            "dtnasc", "DTNASC", "codmunnasc", "CODMUNNASC",
            "racacor", "RACACOR", "numero", "NUMERO",
            "uf", "UF", "estcivmae", "ESTCIVMAE", "gestacao", "GESTACAO",
            "gravidez", "GRAVIDEZ", "parto", "PARTO"
        ]
        
        for col in colunas_exatas:
            if col in self.df_a.columns and col in self.df_b.columns:
                compare.exact(col, col, missing_value=0, label=col)

        features = compare.compute(candidate_links, self.df_a, self.df_b)
        print("DEBUG features.index:", features.index[:5])

        # Identifica os matches que existem na interseção
        intersecao = self.true_matches.intersection(features.index)

        if len(intersecao) == 0:
            print("⚠ Nenhum true match encontrado na interseção. Verifique os IDs!")
            return

        print(f"✅ True matches disponíveis no treino: {len(intersecao)}")

        # Cria labels: 1 = match, 0 = não-match
        y = pd.Series(0, index=features.index)
        y.loc[intersecao] = 1

        # Divide positivos e negativos
        X_pos = features.loc[y[y == 1].index]
        X_neg = features.loc[y[y == 0].index]

        # Undersampling dos negativos (limita para 5x os positivos)
        X_neg_sample = resample(
            X_neg, n_samples=min(len(X_neg), 5 * len(X_pos)), random_state=42
        )

        X = pd.concat([X_pos, X_neg_sample])
        y = pd.Series([1] * len(X_pos) + [0] * len(X_neg_sample), index=X.index)

        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Treina o modelo com peso intermediário para classe 1 (Otimizando Recall sem destruir Precision)
        clf = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight={0: 1, 1: 3}, n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # Avaliação com limite de probabilidade ajustado para aumentar True Positives
        probabilidades = clf.predict_proba(X_test)[:, 1]
        y_pred = (probabilidades >= 0.40).astype(int)
        
        print("\n📊 Relatório de classificação (Otimizado para MAIOR Precisão + Recall):")
        print(classification_report(y_test, y_pred))

        return clf
