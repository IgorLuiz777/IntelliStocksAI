import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

## MODELO DE REGRESSÃO POLINOMIAL

data = pd.read_excel('tabela_excel3.xlsx')

data['data'] = pd.to_datetime(data['data'])
data['days_since_start'] = (data['data'] - data['data'].min()).dt.dayso
X = data['days_since_start'].values.reshape(-1, 1)
y = data['vendas'].values

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

data['previsao_polynomial'] = y_pred

top_3_meses = data.groupby(data['data'].dt.month)['previsao_polynomial'].max().nlargest(3).index.tolist()

print("Os três meses com maior demanda prevista são:", top_3_meses)

plt.plot(data['data'], data['vendas'], label='Dados reais')
plt.plot(data['data'], data['previsao_polynomial'], label='Previsão (Polynomial)', color='red')
plt.legend()
plt.show()
