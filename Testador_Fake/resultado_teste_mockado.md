# Relatório de Teste com Dados Mockados

## 📋 Resumo Executivo

Teste completo executado com sucesso em 3 pipelines de classificação diferentes.

---

## Random Forest (Supervisionado)

### Informações de Debugging

**True Matches:**
```
MultiIndex([(0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4)],
           names=['sinasc_index', 'sim_index'])
```

**Configuração:**
- ✅ 3 regras de bloqueio aplicadas
- ✅ True matches disponíveis no treino: 40

**Features Index:**
```
MultiIndex([(1,  1),
            (2,  2),
            (2, 67),
            (3,  3),
            (3, 12)])
```

### 📊 Relatório de Classificação

Otimizado para MAIOR Precisão + Recall

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 0      | 1.00      | 0.94   | 0.97     | 32      |
| 1      | 0.86      | 1.00   | 0.92     | 12      |
| **Accuracy** | - | **0.95** | - | **44** |
| Macro Avg | 0.93 | 0.97 | 0.95 | 44 |
| Weighted Avg | 0.96 | 0.95 | 0.96 | 44 |

---

## Probabilístico (Pesos Matemáticos)

### Configuração
- ✅ 3 regras de bloqueio aplicadas
- 📊 Threshold otimizado via grid search: **6.0**

### 📊 Resultados

| Métrica | Valor |
|---------|-------|
| Total de pares preditos como MATCH | 48 |
| Total de pares CORRETOS (True Positives) | 38 |
| Falsos Positivos (Errados) | 10 |
| Falsos Negativos (Perdidos) | 12 |
| **Precisão (Precision)** | **79.17%** |
| **Revocação (Recall)** | **76.00%** |
| **F1-Score** | **77.55%** |

---

## Descritivo (Determinístico - Regras de Banco)

### 📊 Resultados

| Métrica | Valor |
|---------|-------|
| Total de pares preditos como MATCH | 26 |
| Total de pares CORRETOS (True Positives) | 21 |
| Falsos Positivos (Errados) | 5 |
| Falsos Negativos (Perdidos) | 29 |
| **Precisão (Precision)** | **80.77%** |
| **Revocação (Recall)** | **42.00%** |
| **F1-Score** | **55.26%** |

---

## ✅ Conclusão

Todos os 3 pipelines mockados foram executados com sucesso!

### Resumo Comparativo

| Classificador | Precisão | Recall | F1-Score |
|---------------|----------|--------|----------|
| Random Forest | 96% (weighted) | 95% | 96% |
| Probabilístico | 79.17% | 76.00% | 77.55% |
| Descritivo | 80.77% | 42.00% | 55.26% |

