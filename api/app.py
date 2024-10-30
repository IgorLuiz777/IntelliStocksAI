from flask import Flask, request, jsonify
import pandas as pd
from modelo import prever_vendas

app = Flask(__name__)

@app.route('/prever', methods=['POST'])
def prever():
    # Verifica se um arquivo foi enviado.
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files['file']

    # Verifica se o arquivo é um CSV.
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Formato de arquivo inválido. Apenas CSV é aceito."}), 400

    # Lê o arquivo CSV em um DataFrame.
    data = pd.read_csv(file)

    # Faz a previsão.
    previsoes = prever_vendas(data)

    # Agrupa as previsões por mês e encontra a máxima previsão de vendas.
    top_3_meses = previsoes.groupby(previsoes['data'].dt.month)['previsao_polynomial'].max().nlargest(3)

    # Converte os meses e previsões para listas.
    meses = top_3_meses.index.tolist()
    previsoes_meses = top_3_meses.values.tolist()

    # Retorna os meses e suas previsões como JSON.
    return jsonify(dict(zip(meses, previsoes_meses)))

if __name__ == '__main__':
    app.run(debug=True)
