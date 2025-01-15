import numpy as np
from problem import probdef_new, Struct
import pandas as pd
from geopy.distance import geodesic
from heurisitcs import fobj_1, fobj_2
import json

def ler_solucoes_do_csv(nome_arquivo):
    """
    Lê um arquivo CSV contendo soluções com f1_value, f2_value e matriz solution no formato JSON.
    Filtra soluções duplicadas com base em f1_value e f2_value.
    Retorna uma lista de objetos que armazenam essas informações.

    Parâmetros:
        nome_arquivo (str): Nome do arquivo CSV contendo as soluções.

    Retorno:
        list: Lista de objetos contendo f1_value, f2_value e a matriz solution.
    """
    # Lê o CSV em um DataFrame
    df = pd.read_csv(nome_arquivo, header=0)  # Supondo que o CSV tem cabeçalho

    # Remove duplicatas com base em 'f1_value' e 'f2_value'
    df = df.drop_duplicates(subset=['f1_value', 'f2_value'], keep='first')

    solucoes = []
    for _, row in df.iterrows():
        f1_value = float(row['f1_value'])
        f2_value = float(row['f2_value'])

        # Converte a matriz JSON de volta para um array numpy
        matriz_json = row['Matriz Solution (14x125)']
        matriz = np.array(json.loads(matriz_json))  # Reconstrói a matriz a partir do JSON

        # Cria o objeto solução
        solucao = Struct()
        solucao.f1_value = f1_value
        solucao.f2_value = f2_value
        solucao.solution = matriz
        solucoes.append(solucao)

    return solucoes

def construir_matriz_bases(caminho_csv):
    """
    Lê um arquivo CSV com separador ';' e números decimais com ',' e constrói uma matriz 
    com as latitudes e longitudes únicas das bases como floats.

    Parâmetros:
        caminho_csv (str): Caminho para o arquivo CSV sem cabeçalho.

    Retorno:
        numpy.ndarray: Matriz 14x2 com as latitudes e longitudes únicas das bases em formato float.
    """
    import pandas as pd
    import numpy as np

    # Lê o CSV sem cabeçalho, com separador ';' e converte ',' em '.' para leitura correta como float
    df = pd.read_csv(
        caminho_csv,
        sep=';',
        header=None,
        names=['lat_base', 'lon_base', 'lat_ativo', 'lon_ativo', 'distancia'],
        converters={
            'lat_base': lambda x: float(x.replace(',', '.')),
            'lon_base': lambda x: float(x.replace(',', '.')),
        }
    )

    # Filtra latitudes e longitudes únicas das bases
    bases_unicas = df[['lat_base', 'lon_base']].drop_duplicates().reset_index(drop=True)

    # Converte para matriz numpy de float
    matriz_bases = bases_unicas.to_numpy(dtype=float)

    # Verifica se há exatamente 14 bases
    if len(bases_unicas) != 14:
        raise ValueError(f"Esperado 14 bases, mas foram encontradas {len(bases_unicas)}. Verifique os dados no CSV.")

    return matriz_bases

def calcular_distancia_bases(coord_base1, coord_base2):
    """
    Calcula a distância entre duas bases utilizando as coordenadas de latitude e longitude.

    Parâmetros:
        coord_base1 (tuple): Coordenadas da base 1 no formato (latitude, longitude).
        coord_base2 (tuple): Coordenadas da base 2 no formato (latitude, longitude).

    Retorno:
        float: Distância em quilômetros entre as duas bases.
    """
    # Converte as coordenadas em tuplas e calcula a distância geodésica
    distancia = geodesic(coord_base1, coord_base2).kilometers
    return distancia

def robustez_indisponibilidade(solution, matriz_bases, probdata):
    """
    Avalia a qualidade de uma solução considerando a perturbação causada pela indisponibilidade de bases ocupadas.
    
    Parâmetros:
        solution (np.ndarray): Matriz 14x125 indicando a alocação das equipes (0, 1, 2 ou 3).
        fobj_1 (func): Função que calcula o valor de f1 para uma configuração de solução.
        fobj_2 (func): Função que calcula o valor de f2 para uma configuração de solução.
        csv_path (str): Caminho para o arquivo CSV com as latitudes e longitudes das bases e ativos.
        
    Retorna:
        float: Índice de qualidade calculado a partir da norma euclidiana das perturbações normalizadas.
    """
    # Constantes de normalização
    f1_min, f1_max = 1010.5, 6279.2
    f2_min, f2_max = 412.6, 2119.8

    # Identificar as bases ocupadas
    bases_ocupadas = set(np.where(solution.solution.sum(axis=1) > 0)[0])

    # Valores iniciais de f1 e f2
    f1_original = fobj_1(solution, probdata).fitness
    f2_original = fobj_2(solution, probdata).fitness

    perturbacoes_f1 = []
    perturbacoes_f2 = []

    # Iterar sobre cada base ocupada
    for base_indisponivel in bases_ocupadas:
        # Encontrar a base mais próxima
        distancias = [
            calcular_distancia_bases(
                matriz_bases[base_indisponivel],
                matriz_bases[base_proxima]
            )
            if base_proxima != base_indisponivel else float('inf')
            for base_proxima in range(len(matriz_bases))
        ]
        base_mais_proxima = np.argmin(distancias)

        # Realocar as equipes e ativos para a base mais próxima
        solution_modificada = solution
        solution_modificada.solution[base_mais_proxima] += solution_modificada.solution[base_indisponivel]
        solution_modificada.solution[base_indisponivel] = 0

        # Calcular os novos valores de f1 e f2
        f1_modificado = fobj_1(solution_modificada, probdata).fitness
        f2_modificado = fobj_2(solution_modificada, probdata).fitness

        # Calcular as perturbações normalizadas
        perturbacao_f1 = abs(f1_modificado - f1_original) / (f1_max - f1_min)
        perturbacao_f2 = abs(f2_modificado - f2_original) / (f2_max - f2_min)

        perturbacoes_f1.append(perturbacao_f1)
        perturbacoes_f2.append(perturbacao_f2)

    # Calcular as médias das perturbações normalizadas
    media_perturbacao_f1 = np.mean(perturbacoes_f1)
    media_perturbacao_f2 = np.mean(perturbacoes_f2)

    # Calcular a norma euclidiana
    robustez = np.sqrt(media_perturbacao_f1 ** 2 + media_perturbacao_f2 ** 2)
    solution.robustez = robustez

    return solution

probdata = probdef_new()

#vetor de alternativas a serem usadas no metodo de decisao
alternativas = ler_solucoes_do_csv("solucoes.csv")
coord_bases = construir_matriz_bases("probdata.csv")

for alternativa in alternativas:
    robustez_indisponibilidade(alternativa, coord_bases, probdata)
    print(alternativa.robustez)

