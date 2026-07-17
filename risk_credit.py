### Projeto de Análise de Risco de Crédito ####

### Ignorar mensagens ---- 
import warnings
warnings.filterwarnings('ignore')

# Instalar bibliotecas necessárias ----
import pandas as pd
import numpy as np

# Bibliotecas de visualização
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')  
plt.show()

import seaborn as sns

## Bibliotecas de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

sns.set(style='whitegrid')

print('Bibliotecas importadas e configuração concluída.')


#### Carregar dataset ----
file_path = 'C:/Users/Alex Gois/Documents/Portfolio/Credit Risk Benchmark Dataset.csv'
df = pd.read_csv(file_path, encoding='ascii')

### Visualização dos dados ----
print('Dataset shape:', df.shape)
print('Primeiras linhas:')
print(df.head())

### Exibir todos os tipos de dados 
print('\nData types:')
print(df.dtypes)


### Limpeza e Pré-processamento de Dados
# Verificação de valores ausentes e consistência dos dados
print('Valores ausentes em cada coluna:')
print(df.isnull().sum())

# Se forem encontrados valores ausentes pode-se considerar imputar ou remover.
# Aqui, simplesmente removeremos as linhas com valores ausentes para simplificar.
df_clean = df.dropna()
print('\nForma após remover valores ausentes:', df_clean.shape)

### Estatísticas Básicas ----
print('\nEstatísticas Descritivas:')
print(df_clean.describe())

#### Análise Exploratória de Dados ----
## Histogramas de todas as colunas numéricas
df_clean.hist(bins=20, figsize=(15, 10), color='teal', edgecolor='black')
plt.suptitle('Histogramas de Recursos Numéricos', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

## Gráfico de Pares para inspecionar relacionamentos entre recursos
sns.pairplot(df_clean, diag_kind='kde', corner=True, plot_kws={'alpha':0.5}, 
vars=df_clean.columns.drop('dlq_2yrs'))
plt.suptitle('Gráfico de Pares de Recursos (excluindo alvo)', y=1.02)
plt.show()

## Mapa de Calor de Correlação
# Para correlação, focamos apenas em características numéricas
numeric_df = df_clean.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(10,8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de Calor de Correlação de Características Numéricas')
    plt.tight_layout()
    plt.show()
else:
    print('Não há colunas numéricas suficientes para um mapa de calor de correlação.')

#### Modelagem Preditiva ----
# Tentaremos prever a variável-alvo 'dlq_2yrs' usando as características restantes.
X = df_clean.drop('dlq_2yrs', axis=1)
y = df_clean['dlq_2yrs']

# Dividir os dados em conjuntos de treinamento e teste (divisão 70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Inicializar e ajustar um modelo de regressão logística
# Observação: A regressão logística foi escolhida porque o alvo parece ser binário (ou categórico) por natureza.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Faça previsões no conjunto de teste
y_pred = model.predict(X_test)

## Avaliação do Modelo
acc = accuracy_score(y_test, y_pred)
print('Acurácia do Modelo de Regressão Logística:', acc)

# Plotar a Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# Plotar a curva ROC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Curva ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Característica Operacional do Receptor')
plt.legend(loc='inferior direito')
plt.tight_layout()
plt.show()

# Importância da Permutação
print('\nImportância da Permutação:')
result = permutation_importance(modelo, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importances= result.importance_mean
indices = np.argsort(importances)

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importance[indices], cor='azul-celeste')
plt.yticks(range(len(indices)),[X_test.columns[i] for i in indices])
plt.xlabel('Importância Média')
plt.title('Importância de Permutação de Características')
plt.tight_layout()
plt.show()