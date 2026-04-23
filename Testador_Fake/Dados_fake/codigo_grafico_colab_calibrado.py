import matplotlib.pyplot as plt
import numpy as np

# Nomes dos classificadores
labels = ['Descritivo', 'Probabilístico', 'Random Forest']

# Dados extraídos da nova saída CALIBRADA (em porcentagem)
precisao = [94.33, 100.00, 100.00]
recall = [95.62, 95.50, 99.00]
f1_score = [94.97, 97.70, 99.00]
taxa_reid = [95.62, 95.50, 99.00]
mrr = [95.63, 96.08, 99.25]

x = np.arange(len(labels))  # Localização no eixo X
width = 0.15  # Largura das barras

# Criando a figura
fig, ax = plt.subplots(figsize=(13, 7))

# Criando as barras para cada métrica
rects1 = ax.bar(x - 2*width, precisao, width, label='Precisão', color='#4c72b0')
rects2 = ax.bar(x - width, recall, width, label='Recall (Revocação)', color='#dd8452')
rects3 = ax.bar(x, f1_score, width, label='F1-Score', color='#55a868')
rects4 = ax.bar(x + width, taxa_reid, width, label='Taxa de Reidentificação', color='#c44e52')
rects5 = ax.bar(x + 2*width, mrr, width, label='MRR', color='#8172b3')

# Adicionando textos, títulos e customizações
ax.set_ylabel('Porcentagem (%)', fontweight='bold', fontsize=12)
ax.set_title('Comparação de Desempenho dos Classificadores de Linkagem (Calibrado)', fontweight='bold', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')

# Movendo a legenda para fora para não sobrepor as barras (já que todas estão muito altas)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=11, ncol=5)
ax.set_ylim(0, 115) # Limite Y para caber o texto acima das barras

# Função para adicionar o valor em % acima de cada barra
def autolabel(rects):
    """Adiciona a porcentagem acima de cada barra"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # offset de 3 pontos verticais
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

# Aplicando a função em todas as categorias
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

fig.tight_layout()

# Exibir o gráfico gerado
plt.show()
