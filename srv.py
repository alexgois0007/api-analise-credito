import os
import logging
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify

# -----------------------------------------------------------------------
# Configuração básica
# -----------------------------------------------------------------------
app = Flask(__name__, static_url_path='/static')

print(app.static_folder)

logging.basicConfig(level=logging.INFO)
logger = app.logger

# Caminho absoluto baseado na localização deste arquivo (evita erro de
# "arquivo não encontrado" quando o script é executado de outra pasta)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Modelo carregado com sucesso de %s", MODEL_PATH)
except FileNotFoundError:
    logger.error("Modelo não encontrado em %s", MODEL_PATH)
    raise


def montar_features(form_data):
    """
    Extrai, valida e converte os campos do formulário/JSON,
    retornando um np.array pronto para o model.predict().

    O template.html já envia os campos categóricos como valores
    numéricos (radio/select com value="1"/"0"), então aqui só
    precisamos validar presença e converter o tipo — não há
    mapeamento de texto para número a fazer.

    Lança ValueError com mensagem clara se algo estiver errado.
    """
    campos_obrigatorios = [
        'gridRadiosSexo', 'dependentes', 'gridRadiosCasado',
        'gridRadiosTrabalhoProprio', 'rendimento', 'educacao',
        'valoremprestimo'
    ]
    faltando = [c for c in campos_obrigatorios if c not in form_data or form_data[c] == '']
    if faltando:
        raise ValueError(f"Campo(s) obrigatório(s) faltando: {', '.join(faltando)}")

    try:
        sexo = int(form_data['gridRadiosSexo'])
        casado = int(form_data['gridRadiosCasado'])
        trabalho_conta_propria = int(form_data['gridRadiosTrabalhoProprio'])
        educacao = int(form_data['educacao'])
        dependentes = int(form_data['dependentes'])
        rendimento = float(form_data['rendimento'])
        valoremprestimo = float(form_data['valoremprestimo'])
    except (TypeError, ValueError):
        raise ValueError("Um ou mais campos possuem valor inválido ou não numérico")

    dados = {
        'sexo': sexo,
        'casado': casado,
        'dependentes': dependentes,
        'educacao': educacao,
        'trabalho_conta_propria': trabalho_conta_propria,
        'rendimento': rendimento,
        'valoremprestimo': valoremprestimo,
    }

    # ATENÇÃO: a ordem das colunas abaixo precisa ser IDÊNTICA à ordem
    # usada no treino do modelo (X_train). Se o dataset de treino tinha
    # outra ordem de colunas, ajuste aqui.
    features = np.array([[
        sexo, casado, dependentes, educacao,
        trabalho_conta_propria, rendimento, valoremprestimo
    ]])

    return features, dados


# -----------------------------------------------------------------------
# Rotas
# -----------------------------------------------------------------------
@app.route('/')
def display_gui():
    return render_template('template.html')


@app.route('/verificar', methods=['POST'])
def verificar():
    """Rota usada pelo formulário HTML (template.html)."""
    try:
        features, dados = montar_features(request.form)
    except ValueError as e:
        logger.warning("Dados inválidos recebidos no formulário: %s", e)
        return render_template('template.html', erro=str(e)), 400

    logger.info(":::::: Dados de Teste ::::::")
    for chave, valor in dados.items():
        logger.info("%s: %s", chave, valor)

    try:
        classe = model.predict(features)[0]
    except Exception:
        logger.exception("Erro ao executar predição")
        return render_template('template.html', erro="Erro interno ao processar a predição"), 500

    logger.info("Classe Predita: %s", classe)

    return render_template('template.html', classe=str(classe))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Rota JSON, útil para testes via curl/Postman e para deixar claro
    que a lógica de predição é uma API de verdade, independente do HTML.

    Exemplo de uso:
    curl -X POST http://localhost:5500/api/predict \
      -H "Content-Type: application/json" \
      -d '{
            "gridRadiosSexo": 1,
            "gridRadiosCasado": 1,
            "dependentes": 2,
            "educacao": 1,
            "gridRadiosTrabalhoProprio": 0,
            "rendimento": 5000,
            "valoremprestimo": 15000
          }'
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"erro": "Corpo da requisição precisa ser um JSON válido"}), 400

    try:
        features, dados = montar_features(payload)
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400

    try:
        classe = model.predict(features)[0]
    except Exception:
        logger.exception("Erro ao executar predição via API")
        return jsonify({"erro": "Erro interno ao processar a predição"}), 500

    return jsonify({
        "classe_predita": str(classe),
        "dados_utilizados": dados
    }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"erro": "Rota não encontrada"}), 404


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5500))
    app.run(host='localhost', port=port)