import numpy as np

# Criando um gerador de números aleatórios
rng = np.random.default_rng()

class Funcao_Ativacao:
    def __init__(self):
        self.funcao = "sigmoide"

    def calcula_ativacao(self, Z: np.ndarray) -> np.ndarray:
        if self.funcao == "sigmoide":
            return (1/(1 + np.exp(-Z)))

        elif self.funcao == "relu":
            return np.maximum(Z, 0)

        else:
            return Z

    def set_funcao_ativacao(self, funcao: str):
        self.funcao = funcao.lower()

class Camada:
    def __init__(self, entrada: int, saida: int):
        self.num_entradas: int = entrada
        self.num_neuronios: int = saida
        self.parametros = rng.uniform(low=0.0, high=1.0, size=(saida, entrada))
        self.bias = rng.uniform(low=0.0, high=1.0, size=(1, saida))
        self.funcao_ativacao = Funcao_Ativacao()

    def set_ativacao(self, funcao: str):
        self.funcao_ativacao.set_funcao_ativacao(funcao)

    def calcula_saida(self, X: np.ndarray) -> np.ndarray:
        Z = np.dot(X, self.parametros.T) + self.bias
        A = self.funcao_ativacao.calcula_ativacao(Z)
        return A

class RedeMLP:
    def __init__(self, arquitetura: list[int]):
        self.camadas = []

        for i in range(len(arquitetura)-1):
            self.camdas.append(Camada(arquitetura[i], arquitetura[i+1]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        A = X.copy()
        for camada in self.camadas:
            A = camada.calcula_saida(A)

        return A






