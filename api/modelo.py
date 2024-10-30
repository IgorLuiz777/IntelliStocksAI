import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def prever_vendas(data):
    # Converte a coluna 'data' em um formato de data reconhecido pelo pandas.
    data['data'] = pd.to_datetime(data['data'])

    # Calcula o número de dias desde o primeiro registro de data até cada ponto no tempo.
    data['days_since_start'] = (data['data'] - data['data'].min()).dt.days

    # X: variável independente (dias desde o início).
    X = data['days_since_start'].values.reshape(-1, 1)
    y = data['vendas'].values

    # Transforma a variável independente em uma matriz com termos polinomiais de grau 2.
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)

    # Cria e ajusta o modelo de regressão linear.
    model = LinearRegression()
    model.fit(X_poly, y)

    # Faz previsões.
    y_pred = model.predict(X_poly)

    # Adiciona previsões ao DataFrame.
    data['previsao_polynomial'] = y_pred

    return data[['data', 'vendas', 'previsao_polynomial']]
