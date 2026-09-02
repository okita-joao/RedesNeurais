# Cabeçalho:
"""
João Pedro Navarro Okita
RA: 176.530
"""

# Descrição do arquivo:
"""
Arquivo python com a implementação da classe de modelos logísticos
"""

# Importação de bibliotecas utilizadas
import numpy as np
import pandas as pd
import math
import random


# CLASSE PARA MODELOS LOGISTICOS
class ModeloLogistico:
    # Método Construtor da Classe:
    def __init__(self):
        self.coeficientes: np.array = None
        self.interceptacao: float = 0
        self.limite: float = 0.5

    # Método para calcular o valor da função de ativação do modelo para "z":
    def funcaoAtivacao(self, z: float) -> float:
        return 1/(1+math.exp(-z))

    # Método para predizer a classe de um caso de exemplo "x":
    def predict(self, x: np.array) -> int:
        if (self.coeficientes is None):
            print("Modelo ainda não treinado!")
            return -1

        try:
            z = float(np.squeeze(np.dot(self.coeficientes, x) + self.interceptacao))
            a = self.funcaoAtivacao(z)
            if a >= self.limite:
                return 1
            else:
                return 0

        except:
            print("Caso de entrada inválido!")
            return -1

    # Método para treinar o modelo logístico a partir de um dataset especifico:
    def train(self, w0: np.array, b0: float, X_Treino: pd.DataFrame, Y_Treino: pd.DataFrame, epocas: int, a: float):
        X = X_Treino.to_numpy().astype(float)
        Y = Y_Treino.to_numpy().astype(int)

        # Definindo o números de exemplos (m_) e o número de features (n_)
        m_ = X.shape[0]
        n_ = X.shape[1]

        # Copiando o chute inicial passado como parâmetro
        w = w0.copy().reshape((1, n_))
        b = b0

        # Iteração das épocas e treinamento do modelo
        for i in range(epocas):
            
            # Forward Propagation
            z = np.dot(X, w.T) + b
            A = 1/(1+np.exp((-1)*z))
        
            J = np.sum(-Y*np.log(A) - (1-Y)*np.log(1-A))/m_

            # Print do valor da função de custo na época específica
            print(f"J(w, b) = {J}")

            assert(A.shape == Y.shape)
            
            # Backward Propagation
            dz = (A - Y).reshape((1, m_))
            
            dw = (np.dot(dz, X))/m_
            dw = dw.reshape(w.shape)
            
            db = np.sum(dz)/m_
            db = float(np.squeeze(db))

            # Atualização dos parâmetros
            w = w - a*dw
            b = b - a*db
        
        self.coeficientes = w.copy()
        self.interceptacao = b