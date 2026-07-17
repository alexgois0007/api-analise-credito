### Bibliotecas necessárias ----
import os 
import re
import pandas as pd
import numpy as np

### Carregar dataset ----

df = pd.read_csv('loan.csv')
df.head()

## Verificando o balanceamento dos Labels (Y - Aprovado, N - Negado), 
# basicamente contamos a quantidade de registros da variável Target: a coluna Loan_Status.
# Observamos que tem um desbalançeamento das classes
print(df['Loan_Status'].value_counts())


          
# Vamos dar uma equilibrada, reduzindo o valor de Aprovados (Y), vamos pegar apenas 200 
# (existem diversas outras técnicas para tratar classes desbalançeadas,
df2 = df[df.Loan_Status=='Y'].sample(200)
 
# Abaixo, anexa as 200 amostras da classe Y com os registros da classe N, em um dataframe que chamamos de data
df = pd.concat([df2, df[df.Loan_Status == 'N'].sample(192)], ignore_index=True)
print(df['Loan_Status'].value_counts())

## Checando valores nulos ----
print(df.isnull().sum())

# Podemos observar que temos algumas colunas contendo valores nulos, como por exemplo: 
# Gender, Married, Dependents, Credit_History e etc.

# Existem algumas técnicas para preenchimento de valores nulos, no nosso projeto, iremos utilizar dois tipos:

# Valor Majoritário (assume o valor majoritário da variável e adiciona nos registros nulos;
# Valor Médio (calcula a média dos valores da variável e adiciona nos registros nulos).
# Preenchendo Missing Values:

# Dependents: Assumindo o valor majoritário.
# Self_Employed: Assumindo o valor majoritário.
# Loan_Amount_Term: Preenchendo com o valor médio.
# Credit_History: Assumindo o valor majoritário.
# Married: Assumindo o valor majoritário.
# Gender: Assumindo o valor majoritário.

df['Gender'] = df['Gender'].fillna('Male')
df['Married'] = df['Married'].fillna('No')
df['Dependents'] = df['Dependents'].fillna('0')
df['Self_Employed'] = df['Self_Employed'].fillna('No')
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Credit_History'] = df['Credit_History'].fillna(1.0)
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())

### Verificando valores ausentes novamente -----
print(df.isnull().sum())

#### Transformando dados categóricos -----
# Várias colunas do dataframe (data) são categóricas, precisamos transforma-las em discretas 
# ("discretiza-las"), são elas: Gender, Married, Education, Self_Employed, Dependents e Loan_Status.
gender_values = {'Female' : 0, 'Male' : 1} 
married_values = {'No' : 0, 'Yes' : 1}
education_values = {'Graduate' : 0, 'Not Graduate' : 1}
employed_values = {'No' : 0, 'Yes' : 1}
dependent_values = {'3+': 3, '0': 0, '2': 2, '1': 1}
loan_values = {'Y':1,'N':0}
df.replace({'Gender': gender_values,
                'Married': married_values, 
                 'Education': education_values,
                 'Self_Employed': employed_values, 
                 'Dependents': dependent_values,
                 'Loan_Status': loan_values
                }, inplace=True)

### Seleção de variáveis ----
df.drop(['Loan_ID','CoapplicantIncome','Loan_Amount_Term','Credit_History','Property_Area'],axis=1,inplace=True)

### Verificando o dataset 
print(df.head())

#### Criação do modelo de Machine Learning ----
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Instanciando o classificador Random Forest
clf_rf = RandomForestClassifier(n_estimators=100,min_samples_split=2)

# Dividindo o conjunto de dados (Valores - X (variáveis independentes) e Rótulos - y (target ou variávies dependentes))
X = df.drop('Loan_Status',axis=1)
y = df['Loan_Status']

# Agora, vamos dividir o conjunto em Treinamento e Teste, utilizando o train_test_split importado acima
#  (tamanho do teste: 30% do conjunto)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y,test_size=0.30,random_state=42)
# Treinando o Modelo - Na etapa abaixo o algoritmo aprenderá os padrões de dados de treinamento que mapeiam as 
# variáveis para o destino e gerará um modelo que captura esses relacionamentos. O modelo de Machine Learning poderá, então, 
# ser usado para obter previsões dos novos dados cuja resposta de destino você não conhece.

print(clf_rf.fit(X_treino,y_treino))

### Métricas de validação ----
from sklearn import metrics
# Iremos utilizar o crosstab para visualizar as classificações.
# Onde veremos a proporção de acertos, comparando o que foi Predito com o Real.

print(pd.crosstab(y_teste, clf_rf.predict(X_teste), rownames=['Real'], colnames=['Predito'], margins=True), '')

# Vamos agora gerar um Relatório de Classificação, ele nos mostra com mais detalhes algumas métricas importantes, tais como: 
# precision, recall, f1-score...
print (metrics.classification_report(y_teste,clf_rf.predict(X_teste)))

### Colocando o modelo em produção -----
# Aqui iremos colocar pra funcionar o nosso modelo, iremos criar uma página da web bem simples para realizar os devidos testes.
# Antes, vamos persistir nosso modelo para o disco! Utilizaremos a biblioteca joblib (que importaremos a seguir)
# Para que estou fazendo isso? Para mandar o modelo persistido para meu servidor web.
import joblib
joblib.dump(clf_rf, "C:/Users/Alex Gois/Documents/Portfolio/model.pkl")
model = joblib.load('model.pkl')

### Teste de Classificação 
# Vamos criar um teste, como se fosse a entrada de dados da nossa página da Web (imagem no final do arquivo)
# Na página devemos informar algumas informações dos possíveis clientes, como:

# Sexo (Masculino(1) e Feminino(0))
# Número de Dependentes
# Casado (Sim(1) ou Não(0))
# Grau de Instrução (Graduado(1) ou Não Graduado(0))
# Trabalha por conta própria (Sim(1) ou Não(2))
# Rendimento (rendimentos do cliente)
# Valor do Emprestimo (valor que o cliente quer emprestado)
# A ordem que vai para o modelo é: Gender(Sexo), Married (Casado), Dependents (Número de Dependentes), Education (Grau de Instrução), Self_Employed (Trabalha por conta própria), ApplicantIncome (Rendimentos) e LoanAmount (Valor do Emprestimo)

# Realizando um testes com as seguintes informações:
# Gender(Sexo) = 1
# Married (Casado) = 1
# Dependents (Número de Dependentes) = 3
# Education (Grau de Instrução) = 0
# Self_Employed (Trabalha por conta própria) = 0
# ApplicantIncome (Rendimentos) = 9504
# LoanAmount (Valor do Emprestimo) = 275


teste = np.array([[1,1,3,0,2,9504,275.0]])
# Rodando o Modelo pra prever o teste que fizemos:
model.predict(teste)     

# O Modelo nos retornou as seguintes probabilidades:
print(model.predict_proba(teste))

# Ou seja, o modelo informou com base nos dados de treinamento, que o cliente (do teste) tem a probabilidade de 21% de pagar o empréstimo, 
# então ele não irá liberar o crédito solicitado para esse cliente.


