# Importando as bibliotecas usadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Modelo com Árvore de Descisão, Random Forest, Gradient Boosting

# Os dados de vendas são lidos de um arquivo Excel (tabela_excel3.xlsx).
data = pd.read_excel('tabela_excel3.xlsx')

# A coluna 'data' é convertida para o formato de data.
data['data'] = pd.to_datetime(data['data'])

# A coluna 'days_since_start' é criada para armazenar o número de dias desde o início dos dados.
data['days_since_start'] = (data['data'] - data['data'].min()).dt.days

# X representa os dias desde o início dos dados (independentes)
# y representa as vendas (dependentes).
X = data['days_since_start'].values.reshape(-1, 1)
y = data['vendas'].values

# DecisionTreeRegressor é um modelo simples de árvore de decisão.
tree_model = DecisionTreeRegressor()
tree_model.fit(X, y)

# RandomForestRegressor é um conjunto de várias árvores de decisão, o que melhora a robustez do modelo.
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# GradientBoostingRegressor é uma técnica que constrói modelos sequenciais onde cada modelo tenta corrigir o erro do anterior.
gb_model = GradientBoostingRegressor()
gb_model.fit(X, y)

# Aqui são feitas previsões para os dados de vendas com base nos três modelos.
y_pred_tree = tree_model.predict(X)
y_pred_rf = rf_model.predict(X)
y_pred_gb = gb_model.predict(X)

# Aqui o código encontra os três meses com maiores vendas previstas para cada modelo. Isso é feito agrupando as vendas por mês e identificando os três maiores valores.
top_3_meses_tree = data.groupby(data['data'].dt.month)['vendas'].max().nlargest(3).index.tolist()
top_3_meses_rf = data.groupby(data['data'].dt.month)['vendas'].max().nlargest(3).index.tolist()
top_3_meses_gb = data.groupby(data['data'].dt.month)['vendas'].max().nlargest(3).index.tolist()

# Os três meses com maior demanda para cada um dos modelos são impressos
print("Os três meses com maior demanda prevista para a Árvore de Decisão são:", top_3_meses_tree)
print("Os três meses com maior demanda prevista para o Random Forest são:", top_3_meses_rf)
print("Os três meses com maior demanda prevista para o Gradient Boosting são:", top_3_meses_gb)

# Um gráfico é criado mostrando as vendas reais (data['vendas']) e as previsões de cada modelo.
# Cada modelo é representado por uma cor diferente, permitindo comparar visualmente a precisão das previsões.
plt.figure(figsize=(10, 6))
plt.plot(data['data'], data['vendas'], label='Dados reais', color='black')
plt.plot(data['data'], y_pred_tree, label='Previsão (Árvore de Decisão)', color='blue')
plt.plot(data['data'], y_pred_rf, label='Previsão (Random Forest)', color='green')
plt.plot(data['data'], y_pred_gb, label='Previsão (Gradient Boosting)', color='red')
plt.legend()
plt.show()

