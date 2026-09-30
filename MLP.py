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
            return A*(A - 1)

        else:
            return np.ones_like(Z)

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
    
    def fit(self, X: np.ndarray, y: np.ndarray, a: float, epocas: int):
        m = X.shape[0]
        for epoca in range(epocas):
            A = self.forward(X)
            print(self.funcao_perca.calcula_perca(A, y) / m)

            # erro da camada de saída: (A - y) * g'(Z)
            delta = (A - y) * self.camadas[-1].calcula_derivada_ativacao()

            for l in range(len(self.camadas) - 1, -1, -1):
                camada = self.camadas[l]
                entrada = self.camadas[l-1].A if l > 0 else X

                dW = delta.T @ entrada / m
                db = delta.sum(axis=0, keepdims=True) / m

                # propaga o erro ANTES de atualizar os pesos
                if l > 0:
                    delta_ant = (delta @ camada.parametros) * self.camadas[l-1].calcula_derivada_ativacao()

                camada.parametros -= a * dW
                camada.bias -= a * db

                if l > 0:
                    delta = delta_ant

    def set_funcao_perca(self, funcao: str):
        self.funcao_perca.set_funcao_perca(funcao)

        







