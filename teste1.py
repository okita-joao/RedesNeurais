

# Importações
import pandas as pd
import numpy as np
import math
import random

from DataProcessing import DataSegmentation, MinMaxScaler
from Modelos import ModeloLogistico

# Importando o dataset base
datasetOriginal = pd.read_csv("dataset-breast-cancer.csv")

datasetBase = datasetOriginal[['radius_mean', 'texture_mean', 'perimeter_mean', 'concavity_worst', 'area_worst', 'smoothness_worst', 'diagnosis']]

m_base: int = datasetBase.shape[0]
n_base: int = datasetBase.shape[1]

for i in range(m_base):
    diagnostico = datasetBase.iloc[i, n_base-1]

    if(diagnostico == "M"):
        datasetBase.iloc[i, n_base-1] = "1"
    else:
        datasetBase.iloc[i, n_base-1] = "0"

# Alterando o dtype da coluna de diagnóstico para integer
datasetBase['diagnosis'].astype(int)

# Segmentando o conjunto de dados original para datasets de treino e teste
(X_Treino, Y_Treino, X_Teste, Y_Teste) = DataSegmentation(datasetBase.drop(columns=["diagnosis"]), datasetBase[["diagnosis"]], 0.8, 42)

# Normalizando o conjunto de dados
normalizador = MinMaxScaler()
(X_Treino_N, X_Teste_N) = normalizador.Scale(X_Treino, X_Teste)

# Definindo o número de casos de exemplo e o número de features
(m, n) = X_Treino_N.shape


# Criando os nós da rede neural:
# Camada Oculta [1]  
m11 = ModeloLogistico()
m21 = ModeloLogistico()

# Camada de Saída [2]
m12 = ModeloLogistico()


# Definindo os parâmetros dos nós da camada oculta de forma arbitrária:
# Camada Oculta [1]
m11.coeficientes = np.random.rand(X_Treino.shape[1])
m11.interceptacao = random.random()

m21.coeficientes = np.random.rand(X_Treino.shape[1])
m21.interceptacao = random.random()

W1 = np.array([m11.coeficientes, m21.coeficientes]).reshape(2, n)
B1 = np.array([m11.interceptacao, m21.interceptacao]).reshape(1, 2) 

# Camada de Saída [2]
m12.coeficientes = np.random.rand(1, 2)
m12.interceptacao = random.random()

W2 = np.array([m12.coeficientes]).reshape(1, 2)
B2 = m12.interceptacao

# Calculando a matriz Z[1]:
Z1 = np.dot(X_Treino_N, W1.T) + B1

# Calculando a matriz A[1]:
A1 = 1/(1+np.exp(-Z1))

# Calculando a matriz Z[2]:
Z2 = np.dot(A1, W2.T) + B2

# Calculando a matriz A[2]:
A2 = 1/(1+np.exp(-Z2))

# A2 é a matriz de predições finais para cada caso do dataset de treino:
Y_ = A2

# Printando as predições finais da rede:
print(Y_)

# Printando o valor da função de custo da rede:
Y_Treino = Y_Treino.to_numpy().astype(int).reshape(m, 1)

J = (-1/m)*np.sum(Y_Treino*np.log(Y_) + (1-Y_Treino)*np.log(1 - Y_))
print(J)



