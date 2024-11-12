import numpy as np
import pandas as pd

def carregar_matriz_distancias(arquivo_csv):
    # Ler o arquivo CSV sem cabeçalho
    dados = pd.read_csv(arquivo_csv, sep=";", decimal=",", header=None)

    # Extrair coordenadas únicas de bases e ativos
    coordenadas_bases = dados[[0, 1]].drop_duplicates().reset_index(drop=True)
    coordenadas_ativos = dados[[2, 3]].drop_duplicates().reset_index(drop=True)

    # Mapear coordenadas para índices
    base_index = {tuple(coord): idx for idx, coord in coordenadas_bases.iterrows()}
    ativo_index = {tuple(coord): idx for idx, coord in coordenadas_ativos.iterrows()}

    # Inicializar matriz de distâncias (14 bases x 125 ativos)
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

    return matriz_distancias

def objetivo_f1(variaveis, distancias):
    dist_soma = 0
    for i in range(0, m):
        for j in range(0, n):
            if(variaveis[i][j] != 0):
                dist_soma +=  distancias[i][j]

    return dist_soma

#[base][ativo] = distancia entre base e ativo
distancias = carregar_matriz_distancias("probdata.csv")

m,n = distancias.shape #m bases e n ativos
s = 3 #equipes

#matriz com as variaveis do problema
#[base][ativo] = equipe responsavel pelo ativo
xyh = np.zeros((m, n), dtype=int) 


