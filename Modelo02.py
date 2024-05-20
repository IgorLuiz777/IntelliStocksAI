import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

## Modelo ARIMA (AutoRegressive Integrated Moving Average)

data = pd.read_excel('tabela_excel3.xlsx')

data['data'] = pd.to_datetime(data['data'])
data.set_index('data', inplace=True)
data.sort_index(inplace=True)

model = ARIMA(data['vendas'], order=(1,1,1))
result = model.fit()

forecast = result.predict(start=data.index[0], end=data.index[-1], typ='levels')

top_3_meses = forecast.nlargest(3).index.month.tolist()

print("Os três meses com maior demanda prevista são:", top_3_meses)

plt.plot(data.index, data['vendas'], label='Dados reais')
plt.plot(forecast.index, forecast, label='Previsão (ARIMA)', color='red')
plt.legend()
plt.show()
