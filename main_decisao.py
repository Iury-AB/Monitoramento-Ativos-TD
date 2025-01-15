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

def robustez_indisponibilidade(x, matriz_bases, probdata):
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
    bases_ocupadas = set(np.where(x.solution.sum(axis=1) > 0)[0])

    # Valores iniciais de f1 e f2
    f1_original = fobj_1(x, probdata).fitness
    f2_original = fobj_2(x, probdata).fitness

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
        solution_modificada = x
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
    x.robustez = robustez

    return x

def balanco_carga(x, probdata):
    carga_eq = np.zeros(3)
    xyh = x.solution
    for j in range(0, probdata.n):
        for i in range(0, probdata.m):
            if (xyh[i][j] == 1):
                carga_eq[0] += 1
                break
            elif (xyh[i][j] == 2):
                carga_eq[1] += 1
                break
            elif (xyh[i][j] == 3):
                carga_eq[2] += 1
                break
    sd = np.std(carga_eq)
    x.balanco_sd = sd
    return x

def ahp_classic(solucoes, matriz_comparacao_atributos):
    """
    Implementa o método AHP clássico para escolher entre soluções considerando quatro atributos.

    Parâmetros:
        solucoes (list): Lista de objetos, onde cada objeto tem os atributos `f1_value`, `f2_value`, `robustez`, `balanco_sd`.
        matriz_comparacao_atributos (numpy.ndarray): Matriz 4x4 de comparação par a par entre os atributos.

    Retorno:
        Object: Solução escolhida após a execução do AHP.
    """

    def normalizar_matriz(matriz):
        """Normaliza a matriz e retorna os pesos calculados."""
        col_sum = np.sum(matriz, axis=0)
        matriz_norm = matriz / col_sum
        pesos = np.mean(matriz_norm, axis=1)
        return pesos

    # Cria as matrizes de comparação par a par para cada atributo
    n = len(solucoes)
    matrizes_atributos = { 
        'f1_value': np.ones((n, n)), 
        'f2_value': np.ones((n, n)), 
        'robustez': np.ones((n, n)), 
        'balanco_sd': np.ones((n, n)) 
    }
    
    # Dicionário com os valores mínimos e máximos de cada atributo
    limites = {
        'f1_value': {'min': 1010.5, 'max': 1185.0},
        'f2_value': {'min': 412.6, 'max': 427.5},
        'robustez': {'min': 0, 'max': 0.03},
        'balanco_sd': {'min': 0, 'max': 17.82},
    }
    
    # Preenche as matrizes de comparação par a par com base nas diferenças normalizadas
    for i in range(n):
        for j in range(i + 1, n):  # Calcula apenas metade superior
            for atributo in ['f1_value', 'f2_value', 'robustez', 'balanco_sd']:
                val_i = getattr(solucoes[i], atributo)
                val_j = getattr(solucoes[j], atributo)
                
                # Obtém os limites para o atributo atual
                min_atributo = limites[atributo]['min']
                max_atributo = limites[atributo]['max']
                
                # Normaliza a diferença com base nos limites do atributo
                diff_normalizado = abs(val_i - val_j) / (max_atributo - min_atributo)
                
                # Normaliza a escala para 1 a 9
                if diff_normalizado != 0:
                    comparacao = min(max(round(1 + 8 * diff_normalizado, 2), 1), 9)
                else:
                    comparacao = 1  # Se os valores forem iguais
                
                # Preenche a matriz de comparações
                if val_i < val_j:
                    matrizes_atributos[atributo][i, j] = min(comparacao, 9)
                else:
                    matrizes_atributos[atributo][i, j] = min(round(1 / comparacao, 2), 9)
                
                matrizes_atributos[atributo][j, i] = min(round(1 / matrizes_atributos[atributo][i, j], 2), 9)

    # Calcula os pesos das soluções em cada atributo
    pesos_solucoes_por_atributo = {}
    for atributo, matriz in matrizes_atributos.items():
        pesos_solucoes_por_atributo[atributo] = normalizar_matriz(matriz)
    
    # Calcula os pesos dos atributos (prioridades) usando a matriz de comparação 4x4 fornecida
    pesos_atributos = normalizar_matriz(matriz_comparacao_atributos)

    # Calcula o peso final de cada solução
    pesos_finais_solucoes = np.zeros(n)
    for i in range(n):
        for j, atributo in enumerate(['f1_value', 'f2_value', 'robustez', 'balanco_sd']):
            pesos_finais_solucoes[i] += pesos_solucoes_por_atributo[atributo][i] * pesos_atributos[j]
    
    # Seleciona a solução com maior peso final
    indice_melhor_solucao = np.argmax(pesos_finais_solucoes)
    return (indice_melhor_solucao, solucoes[indice_melhor_solucao])

probdata = probdef_new()

#vetor de alternativas a serem usadas no metodo de decisao
alternativas = ler_solucoes_do_csv("solucoes.csv")
coord_bases = construir_matriz_bases("probdata.csv")

for alternativa in alternativas:
    robustez_indisponibilidade(alternativa, coord_bases, probdata)
    balanco_carga(alternativa, probdata)

atributos_par_a_par = [[1, 3, 5, 1/2], [1/3, 1, 4, 3], [1/5, 1/4, 1, 7], [2, 1/3, 1/7, 1]]

print(ahp_classic(alternativas, atributos_par_a_par)[0])