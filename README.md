# 💳 Análise de Risco de Crédito com Machine Learning

API web desenvolvida em **Flask** que utiliza um modelo de **Random Forest** para prever se um cliente terá o crédito (empréstimo) aprovado ou negado, com base em dados socioeconômicos informados em um formulário.

> ⚠️ Projeto desenvolvido para fins de portfólio, utilizando dados fictícios/públicos de benchmark. Não deve ser utilizado para decisões reais de concessão de crédito.

---

## 🔗 Demo

<!-- Substitua pelo link do deploy quando estiver disponível -->
🚧 Em breve — deploy ainda não publicado. Enquanto isso, veja como rodar localmente na seção abaixo.

---

## 📋 Sobre o Projeto

A aplicação recebe informações do solicitante do crédito — sexo, estado civil, número de dependentes, grau de instrução, se trabalha por conta própria, rendimento mensal e valor do empréstimo solicitado — e retorna uma predição de aprovação ou negação, gerada por um modelo de classificação treinado previamente.

O projeto foi construído com dois pontos de entrada:

- **Interface web** (`/`): formulário HTML estilizado com Bootstrap, pensado para uso humano.
- **API REST em JSON** (`/api/predict`): endpoint pensado para integração com outras aplicações, testável via `curl`/Postman.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.11**
- **Flask** — servidor web e roteamento
- **scikit-learn** — treinamento do modelo (Random Forest)
- **joblib** — serialização/carregamento do modelo treinado
- **NumPy** — manipulação dos dados de entrada
- **Bootstrap 4** — estilização do formulário

---

## 📂 Estrutura do Projeto

```
Portfolio/
├── srv.py                  # Servidor Flask (rotas, validação, predição)
├── model.pkl                # Modelo treinado (Random Forest)
├── requirements.txt          # Dependências do projeto
├── Procfile                  # Configuração para deploy (Render/Railway/Heroku)
├── templates/
│   └── template.html         # Formulário web
└── static/                   # Arquivos estáticos (caso necessário)
```

---

## ▶️ Como Rodar Localmente

1. Clone o repositório:

```bash
git clone https://github.com/alexgois0007/portfolio.git
cd portfolio
```

2. Crie um ambiente virtual (recomendado):

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute o servidor:

```bash
python srv.py
```

5. Acesse no navegador:

```
http://localhost:5500
```

---

## 🔌 Testando a API via JSON

```bash
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
```

**Resposta esperada:**

```json
{
  "classe_predita": "1",
  "dados_utilizados": {
    "sexo": 1,
    "casado": 1,
    "dependentes": 2,
    "educacao": 1,
    "trabalho_conta_propria": 0,
    "rendimento": 5000.0,
    "valoremprestimo": 15000.0
  }
}
```

---

## 📊 Sobre os Dados e o Modelo

O modelo foi treinado com um dataset de benchmark de risco de crédito, contendo variáveis como sexo, estado civil, número de dependentes, grau de instrução, tipo de ocupação, rendimento e valor do empréstimo solicitado. O algoritmo utilizado foi **Random Forest Classifier**, escolhido por sua boa performance em problemas de classificação binária com variáveis mistas (numéricas e categóricas).

<!-- Se tiver métricas de avaliação do modelo (acurácia, precisão, recall, matriz de confusão),
     vale muito a pena adicionar aqui — é um dos pontos que mais pesa na avaliação técnica -->

---

## 🚀 Próximos Passos

- [ ] Publicar deploy em produção (Render/Railway)
- [ ] Adicionar métricas de avaliação do modelo no README
- [ ] Adicionar testes automatizados
- [ ] Persistir histórico de predições em banco de dados

---

## 👤 Autor

**Alex Gois**
[GitHub](https://github.com/alexgois0007)