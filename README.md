## IntelliStocksAI - 3 modelos
Este repositório contém três projetos distintos para prever vendas usando diferentes técnicas de modelagem. Cada projeto utiliza um método específico de aprendizado de máquina para analisar dados históricos e gerar previsões. Abaixo, você encontrará uma breve descrição de cada projeto.

## Dependências (Bibliotecas)
- pandas
- numpy
- matplotlib
- scikit-learn
- matplotlib
- statsmodels
- matplotlib
- scikit-learn

## 1. Modelos de Regressão (Árvore de Decisão, Random Forest e Gradient Boosting)
Neste projeto, são aplicados três modelos de regressão para prever vendas com base em dados históricos:

- Árvore de Decisão: Um modelo simples que faz previsões baseadas em decisões binárias.
- Random Forest: Um ensemble de árvores de decisão que melhora a robustez e a precisão das previsões.
- Gradient Boosting: Um método que cria modelos sequenciais para corrigir erros dos modelos anteriores.

Visão geral desse modelo
O código lê os dados de vendas, ajusta os modelos, realiza previsões e compara os resultados para identificar os meses com maior demanda prevista.


## 2. Modelo ARIMA (AutoRegressive Integrated Moving Average)
Este projeto utiliza o modelo ARIMA para previsões de séries temporais:

ARIMA: Um modelo estatístico que combina auto-regressão, diferenciação e média móvel para prever valores futuros com base em dados passados.
O código carrega os dados de vendas, ajusta o modelo ARIMA, realiza previsões e identifica os meses com maior demanda. As previsões são comparadas com os dados reais em um gráfico.

## 3. Regressão Polinomial
Este projeto aplica a regressão polinomial para prever vendas:

Regressão Polinomial: Um tipo de regressão linear que utiliza termos polinomiais (neste caso, de grau 2) para capturar relacionamentos não lineares entre variáveis.
O código lê os dados de vendas, ajusta o modelo de regressão polinomial, faz previsões e determina os meses com maior demanda prevista. As previsões são visualizadas junto com os dados reais em um gráfico.

