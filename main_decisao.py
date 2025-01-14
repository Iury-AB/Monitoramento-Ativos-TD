import numpy as np
from problem import probdef_new, Struct

def ler_solucoes_do_csv(nome_arquivo):
    """
    Lê um arquivo CSV contendo soluções com f1_value, f2_value e matriz solution.
    Filtra soluções duplicadas, mantendo apenas uma de cada.
    Retorna uma lista de objetos que armazenam essas informações.

    Parâmetros:
        nome_arquivo (str): Nome do arquivo CSV contendo as soluções.

    Retorno:
        list: Lista de objetos contendo f1_value, f2_value e a matriz solution.
    """
    solucoes = []
    vistos = set()  # Para armazenar soluções únicas com hash

    with open(nome_arquivo, mode='r') as arquivo:
        next(arquivo)  # Ignora o cabeçalho
        for linha in arquivo:
            dados = linha.strip().split(',')  # Divide os campos separados por vírgula
            f1_value = float(dados[2])  # Terceira coluna: f1_value
            f2_value = float(dados[3])  # Quarta coluna: f2_value
            matriz_flat = list(map(float, dados[4:]))  # Resto: matriz linearizada
            
            # Gera um hash único para cada solução baseada em f1_value, f2_value e a matriz
            hash_solucao = (f1_value, f2_value, tuple(matriz_flat))
            if hash_solucao in vistos:
                continue  # Solução já foi vista, ignora
            
            vistos.add(hash_solucao)  # Marca como vista
            matriz = np.array(matriz_flat).reshape(14, 125)  # Reconstrói a matriz
            solucoes.append(Struct(f1_value=f1_value, f2_value=f2_value, solution=matriz))
    
    return solucoes

probdata = probdef_new()

#vetor de alternativas a serem usadas no metodo de decisao
alternativas = ler_solucoes_do_csv("solucoes.csv")
