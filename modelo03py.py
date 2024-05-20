import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

## Modelo com Árvore de Descisão, Random Forest, Gradient Boosting

data = pd.read_excel('tabela_excel3.xlsx')

data['data'] = pd.to_datetime(data['data'])
data['days_since_start'] = (data['data'] - data['data'].min()).dt.days
X = data['days_since_start'].values.reshape(-1, 1)
y = data['vendas'].values

tree_model = DecisionTreeRegressor()
tree_model.fit(X, y)

rf_model = RandomForestRegressor()
rf_model.fit(X, y)

gb_model = GradientBoostingRegressor()
gb_model.fit(X, y)

y_pred_tree = tree_model.predict(X)
y_pred_rf = rf_model.predict(X)
y_pred_gb = gb_model.predict(X)

top_3_meses_tree = data.groupby(data['data'].dt.month)['vendas'].max().nlargest(3).index.tolist()
top_3_meses_rf = data.groupby(data['data'].dt.month)['vendas'].max().nlargest(3).index.tolist()
top_3_meses_gb = data.groupby(data['data'].dt.month)['vendas'].max().nlargest(3).index.tolist()

print("Os três meses com maior demanda prevista para a Árvore de Decisão são:", top_3_meses_tree)
print("Os três meses com maior demanda prevista para o Random Forest são:", top_3_meses_rf)
print("Os três meses com maior demanda prevista para o Gradient Boosting são:", top_3_meses_gb)

plt.figure(figsize=(10, 6))
plt.plot(data['data'], data['vendas'], label='Dados reais', color='black')
plt.plot(data['data'], y_pred_tree, label='Previsão (Árvore de Decisão)', color='blue')
plt.plot(data['data'], y_pred_rf, label='Previsão (Random Forest)', color='green')
plt.plot(data['data'], y_pred_gb, label='Previsão (Gradient Boosting)', color='red')
plt.legend()
plt.show()

