import pandas as pd
import math
import numpy as np


class Struct:
    pass


def carregar_matriz_distancias(arquivo_csv):
    print(arquivo_csv)
    # Ler o arquivo CSV sem cabeçalho
    dados = pd.read_csv(arquivo_csv, sep=";", decimal=",", header=None)

    # Extrair coordenadas únicas de bases e ativos
    coordenadas_bases = dados[[0, 1]].drop_duplicates().reset_index(drop=True)
    coordenadas_ativos = dados[[2, 3]].drop_duplicates().reset_index(drop=True)

    # Mapear coordenadas para índices
    base_index = {tuple(coord): idx for idx, coord in coordenadas_bases.iterrows()}
    ativo_index = {tuple(coord): idx for idx, coord in coordenadas_ativos.iterrows()}

    # Inicializar matriz de distâncias (m bases x n ativos)
    matriz_distancias = np.zeros((len(coordenadas_bases), len(coordenadas_ativos)))

    # Preencher a matriz com as distâncias
    for _, linha in dados.iterrows():
        base_coord = (linha[0], linha[1])
        ativo_coord = (linha[2], linha[3])
        distancia = linha[4]

        # Pegar os índices da base e do ativo e inserir a distância na matriz
        i = base_index[base_coord]
        j = ativo_index[ativo_coord]
        matriz_distancias[i, j] = distancia

        df = pd.DataFrame(data=matriz_distancias)

    return df


def carregar_probabilidades(arquivo_xlsx):
    dados = pd.read_excel(arquivo_xlsx, header=None)
    p = dados.iloc[:, 2].values
    return p


def probdef(s=np.int8(3), eta=0.2, csv="probdata.csv"):
    distancias = carregar_matriz_distancias(csv)
    m, n = distancias.shape  # m bases e n ativos
    probdata = Struct()
    probdata.eta = eta
    probdata.n = n
    probdata.m = m
    probdata.s = s
    probdata.d = distancias
    probdata.csv = csv
    return probdata

#TODO: usar essa função no lugar de probdef

def probdef_new(s=np.int8(3), eta=0.2, csv_dist="probdata.csv", xlsx_prob="probfalhaativos.xlsx"):
    distancias = carregar_matriz_distancias(csv_dist)
    p = carregar_probabilidades(xlsx_prob)
    m,n = distancias.shape
    probdata = Struct()
    probdata.eta = eta
    probdata.n = n
    probdata.m = m
    probdata.s = s
    probdata.d = distancias
    probdata.p = p
    probdata.csv = csv_dist
    return probdata