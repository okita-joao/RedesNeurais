# Cabeçalho:
"""
João Pedro Navarro Okita
RA: 176.530
"""

# Descrição do arquivo:
"""
Arquivo python com a implementação de métodos de tratamento de dados
"""

# Importação de bibliotecas utilizadas
import numpy as np
import pandas as pd
import math
import random


# MÉTODO PARA SEGMENTAÇÃO DE DATASETS
def DataSegmentation(X_base: pd.DataFrame, Y_base: pd.DataFrame, percentTreino: float, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Setando a semente de aleatoriedade
    rng = np.random.default_rng(seed=random_seed)
    
    # Descobrindo as dimensões do dataset base
    (m, n) = X_base.shape

    # Definindo as dimensões de cada conjunto de dados distinto
    m_treino = math.floor(m*percentTreino)
    n_treino = n
    
    m_teste = m - m_treino
    n_teste = n
    
    # Criando os dataframes
    X_Treino = X_base.head(0) 
    X_Teste = X_base.head(0)
    Y_Treino = Y_base.head(0)
    Y_Teste = Y_base.head(0)
    
    # Distruindo aleatóriamente entre os dois conjuntos de dados
    casos_de_treino = rng.choice(m-1, size=m_treino, replace=False)
    casos_de_treino.sort()
    
    j: int = 0
    for i in range(m):
        x = X_base.loc[i]
        y = Y_base.loc[i]
        
        if j < m_treino and casos_de_treino[j] == i:
            X_Treino.loc[X_Treino.shape[0]] = x
            Y_Treino.loc[Y_Treino.shape[0]] = y
            j += 1
        else:
            X_Teste.loc[X_Teste.shape[0]] = x
            Y_Teste.loc[Y_Teste.shape[0]] = y

    return (X_Treino, Y_Treino, X_Teste, Y_Teste)


# CLASSE DO ESCALONADOR MinMax
class MinMaxScaler:
    def __init__(self):
        self.maximos = []
        self.minimos = []

    def Scale(self, X_Treino: pd.DataFrame, X_Teste: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        (m, n) = X_Treino.shape

        (m_teste, n_teste) = X_Teste.shape
        
        for i in range(n):
            self.maximos.append(X_Treino[X_Treino.columns[i]].max())
            self.minimos.append(X_Treino[X_Treino.columns[i]].min())
        
        # Criando as matrizes normalizadas
        X_Treino_N = X_Treino.head(0)
        X_Teste_N = X_Teste.head(0)
        
        # Normalizando o conjunto de treino
        for i in range(m):
            linha = []
            for j in range(n):
                linha.append((X_Treino.iloc[i, j] - self.minimos[j])/(self.maximos[j] - self.minimos[j]))
            X_Treino_N.loc[i] = linha
        
        # Normalizando o conjunto de teste
        for i in range(m_teste):
            linha = []
            for j in range(n_teste):
                linha.append((X_Teste.iloc[i, j] - self.minimos[j])/(self.maximos[j] - self.minimos[j]))
            X_Teste_N.loc[i] = linha

        return (X_Treino_N, X_Teste_N)