# Importando as bibliotecas usadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

## Modelo ARIMA (AutoRegressive Integrated Moving Average)

# Lê os dados de vendas de um arquivo Excel (tabela_excel3.xlsx).
data = pd.read_excel('tabela_excel3.xlsx')

# A coluna de datas é convertida para o formato datetime do pandas
data['data'] = pd.to_datetime(data['data'])

# A coluna 'data' é definida como o índice do DataFrame para facilitar a análise de séries temporais.
# Ordenação dos dados pelo índice: Garante que os dados estejam ordenados cronologicamente.
data.set_index('data', inplace=True)
data.sort_index(inplace=True)

# O modelo ARIMA é configurado com três parâmetros: 1, 1 e 1
model = ARIMA(data['vendas'], order=(1,1,1))

# O modelo é ajustado aos dados de vendas (data['vendas']) usando o método fit().
result = model.fit()

# O método predict() realiza a previsão de vendas. 
# A previsão começa da primeira data disponível no conjunto de dados (start=data.index[0]) e vai até a última (end=data.index[-1]). 
# O parâmetro typ='levels' indica que a previsão deve ser feita nos níveis reais da série (em vez de nas diferenças).
forecast = result.predict(start=data.index[0], end=data.index[-1], typ='levels')

# A função nlargest(3) seleciona os três maiores valores previstos de vendas.
top_3_meses = forecast.nlargest(3).index.month.tolist()
print("Os três meses com maior demanda prevista são:", top_3_meses)

# Os dados reais de vendas são plotados ao lado das previsões geradas pelo modelo ARIMA.
plt.plot(data.index, data['vendas'], label='Dados reais')
plt.plot(forecast.index, forecast, label='Previsão (ARIMA)', color='red')
plt.legend()
plt.show()
