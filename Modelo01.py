# Importando as bibliotecas usadas
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# MODELO DE REGRESSÃO POLINOMIAL
# Lê um arquivo Excel chamado 'tabela_excel3.xlsx', que contém os dados de vendas.
data = pd.read_excel('tabela_excel3.xlsx')

# Converte a coluna 'data' em um formato de data reconhecido pelo pandas.
data['data'] = pd.to_datetime(data['data'])

# Calcula o número de dias desde o primeiro registro de data até cada ponto no tempo, armazenando o resultado em uma nova coluna 'days_since_start'.
data['days_since_start'] = (data['data'] - data['data'].min()).dt.days

# X: variável independente, que representa os dias desde o início.
# y: variável dependente, que são as vendas registradas.
X = data['days_since_start'].values.reshape(-1, 1)
y = data['vendas'].values

# Transforma a variável independente (X) em uma nova matriz com termos polinomiais de grau 2 (quadrático).
poly_features = PolynomialFeatures(degree=2)

# Contém as variáveis independentes transformadas em uma forma polinomial.
X_poly = poly_features.fit_transform(X)

# Cria o modelo de regressão linear.
model = LinearRegression()

# Ajusta o modelo aos dados transformados, aprendendo a relação entre os dias e as vendas.
model.fit(X_poly, y)

# Usa o modelo treinado para prever as vendas (y_pred) com base nos dias (X_poly).
y_pred = model.predict(X_poly)

# Armazena as previsões em uma nova coluna da tabela de dados.
data['previsao_polynomial'] = y_pred

# Agrupa os dados por mês, depois encontra a previsão máxima de vendas para cada mês, após isso seleciona os 3 meses com as maiores previsões de vendas e converte esses meses para uma lista.
top_3_meses = data.groupby(data['data'].dt.month)['previsao_polynomial'].max().nlargest(3).index.tolist()
print("Os três meses com maior demanda prevista são:", top_3_meses)

# Plota dois gráficos: um com os dados reais de vendas e outro com as previsões do modelo polinomial, depois adiciona uma legenda para diferenciar os dados reais e as previsões e exibe o gráfico.
plt.plot(data['data'], data['vendas'], label='Dados reais')
plt.plot(data['data'], data['previsao_polynomial'], label='Previsão (Polynomial)', color='red')
plt.legend()
plt.show()
