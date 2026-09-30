import numpy as np

# Criando um gerador de números aleatórios
rng = np.random.default_rng()

class Funcao_Ativacao:
    def __init__(self):
        self.funcao = "sigmoide"

    def calcula_ativacao(self, Z: np.ndarray) -> np.ndarray:
        if self.funcao == "sigmoide":
            return (1/(1 + np.exp(-Z)))

        else:
            return Z
    
    # Supõe que A = g(Z)
    def derivada_ativacao(self, A: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if self.funcao == "sigmoide":
            return np.power(A, 2)*(A - 1)

        else:
            return Z

    def set_funcao_ativacao(self, funcao: str):
        self.funcao = funcao.lower()

class Funcao_Perca:
    def __init__(self):
        self.funcao = "CEG"

    def calcula_perca(self, p: np.ndarray, y: np.ndarray) -> float:
        if self.funcao == "CEG":
            return -np.sum(y*np.log(p))
        
        elif self.funcao == "MSE":
            return np.sum(np.power((y - p), 2))/2
        
        else:
            self.set_funcao_perca("MSE")
            return self.calcula_perca(p, y)
        
    def set_funcao_perca(self, funcao: str):
        self.funcao = funcao

class Camada:
    def __init__(self, entrada: int, saida: int):
        self.num_entradas: int = entrada
        self.num_neuronios: int = saida
        self.parametros = rng.uniform(low=0.0, high=1.0, size=(saida, entrada))
        self.bias = rng.uniform(low=0.0, high=1.0, size=(1, saida))
        self.funcao_ativacao = Funcao_Ativacao()
        self.Z = False
        self.A = False

    def set_ativacao(self, funcao: str):
        self.funcao_ativacao.set_funcao_ativacao(funcao)

    def calcula_saida(self, X: np.ndarray) -> np.ndarray:
        self.Z = np.dot(X, self.parametros.T) + self.bias
        self.A = self.funcao_ativacao.calcula_ativacao(self.Z)
        return self.A

    def calcula_derivada_ativacao(self) -> np.ndarray:
        return self.funcao_ativacao.derivada_ativacao(self.A, self.Z)

class RedeMLP:
    def __init__(self, arquitetura: list[int]):
        self.camadas = []

        for i in range(len(arquitetura)-1):
            self.camadas.append(Camada(arquitetura[i], arquitetura[i+1]))

        self.funcao_perca = Funcao_Perca()

    def predict(self, X: np.ndarray) -> np.ndarray:
        A = X.copy()
        for camada in self.camadas:
            A = camada.calcula_saida(A)

        return A

    def foward(self, X: np.ndarray) -> np.ndarray:
        A = X.copy()
        for camada in self.camadas:
            A = camada.calcula_saida(A)

        return A
    
    def fit(self, X_treino: np.ndarray, y_treino: np.ndarray, a: float, epocas: int):
        m: int = X_treino.shape[0]
        n: int = X_treino.shape[1]
        s: int = y_treino.shape[0]
        
        for i in range(epocas):
            # Executa a foward propagation e calcula a previsão da rede para todos os casos de exemplo
            A = self.foward(X_treino)

            if(A.shape != y_treino.shape):
                print("Erro no calculo da propagação, A.shape != y_treino.shape")
                return 

            # Calcula e printa o valor da função de custo J da rede
            J = self.funcao_perca.calcula_perca(A, y_treino)/m
            print(J)

            # Calcula o erro de forma iterativa de todas as camadas da rede, começando pela camada de saída

            # Calculo da variação de J em função dos parâmetros da camada de saída
            num_camadas: int = len(self.camadas)
            camada_saida: Camada = self.camadas[num_camadas - 1]
            derivada_camada_saida = camada_saida.calcula_derivada_ativacao().reshape(m, s) # g'(Z[saída])

            erro_k: np.ndarray = (((A - y_treino).reshape(m, s))*derivada_camada_saida)/m

            dJ = np.dot(erro_k.T, A)

            # Atualizando os parâmetros da camada de saída
            camada_saida.parametros = camada_saida.parametros - a*dJ

            # Camadas restantes (até a camada de início)
            for i in range(num_camadas - 2, 0, -1):
                camada: Camada = self.camadas[i]
                M = False
                
                if i >= 1:
                    M = self.camadas[i-1].A
                else:
                    M = X_treino

                derivada_camada = camada.calcula_derivada_ativacao() # g'(Z[camada[i]])

                erro_j: np.ndarray = ((np.dot(erro_k, camada.parametros))*derivada_camada)/m

                dJ = np.dot(erro_j.T, M)

                # Aualizando os parâmetros da camada[i]
                camada.parametros = camada.parametros - a*dJ

                erro_k = erro_j.copy()

    def set_funcao_perca(self, funcao: str):
        self.funcao_perca.set_funcao_perca(funcao)

        







