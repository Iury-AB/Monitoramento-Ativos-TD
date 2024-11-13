'''
Importa os módulos usados
'''
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
from random import sample
import math

np.set_printoptions(threshold=np.inf) # diretiva para imprimir todos os elementos de uma matriz

'''
Define um tipo de dado similar ao Pascal "record" or C "struct"
'''
class Struct:
    pass

'''
Ler o arquivo csv contendo as posições geográficas das bases e ativos
'''
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

def probdef(arquivo_csv):
    probdata = Struct()
    probdata.d = carregar_matriz_distancias(arquivo_csv)
    probdata.m,probdata.n = probdata.d.shape
    probdata.s = 3
    probdata.eta = 0.2
    return probdata

'''
Implementa a função objetivo do problema
'''
def fobj_1(x, probdata):
    dist_soma = 0
    '''
    Modelou-se as variáveis como uma matriz base x ativos de equipes atribuidas
    Temos m bases e n ativos e s equipes
    k = 0,1,2,3 indica qual das 3 equipes está alocada, se k=0, nenhuma equipe está alocada
           i1 i2 ... in
    xyh = [k  k  ... k ] b1
          [k  k  ... k ] b2
                 ...
          [k  k  ... k ] bm

    f1 = soma(i:1->n) soma(j:1->m) [ x_ij * d_ij ]
    '''
    xyh=x.solution
    for i in range(0, probdata.m):
        for j in range(0, probdata.n):
            if(xyh[i][j] != 0):
                dist_soma +=  probdata.d[i][j]

    x.fitness = dist_soma

    return x

'''
Implementa uma solução inicial para o problema
'''
def sol_inicial(probdata,apply_constructive_heuristic=False):
    
    '''
    Matriz solução: xyh = [  a1 a2 ... ai ... an
                          b1
                          b2
                          ...
                          bj
                          ...
                          bm                    ]
    ''' 
    if not apply_constructive_heuristic:
        # Constrói solução inicial aleatoriamente
        x = Struct()
        xyh = np.zeros((probdata.m,probdata.n), dtype=int) # cria uma matriz de elementos de mesma forma do arquivo csv atribuindo valores zero
        resp=math.ceil(probdata.eta*probdata.n/probdata.s) # Calcula a R6
        equipes_sorteadas = sample(range(1, probdata.s + 1), probdata.s) # sorteia aleatoriamente s equipes
        bases_sorteadas = sample(range(0,probdata.m),probdata.s) # sorteia aleatoriamente s bases

        for i,equipe in enumerate(equipes_sorteadas):
            if (i == 0):
                xyh[bases_sorteadas[i],i:resp] = equipes_sorteadas[i] # Atribui os resp elementos da base de índice 0 à equipe i
            elif (i == len(equipes_sorteadas) - 1):
                xyh[bases_sorteadas[i],(i)*resp:probdata.n] = equipes_sorteadas[i] # Atribui à última equipe os últimoa ativos à base i
            else:
                xyh[bases_sorteadas[i],i*resp:(i+1)*resp] = equipes_sorteadas[i] # Atribui os resp elementos seguintes da base 1 à equipe i
            
        x.solution = xyh
    
    else:
        ## Constrói solução inicial usando uma heurística construtiva
        x = Struct()
        x.solution = np.zeros((probdata.n,probdata.m), dtype=int)
        job = np.argsort(probdata.d[:,4].corr(axis=1))    # ativos ordenadas de acordo com a correlaçao das distânciais
        for ativo in job[::-1]:        
            base = np.argmin(probdata.d[:,ativo]) # atribui as tarefas em ordem decrescente de variância ao agente de menor custo
            x.solution.insert(base,ativo)
        
    return x


'''
Implementa a função neighborhoodChange
'''
def neighborhoodChange(x, y, k):
    
    if y.fitness < x.fitness:
        x = copy.deepcopy(y)
        k = 1
    else:
        k += 1
        
    return x, k

'''
Implementa a função shake
'''
def shake(x, k, probdata):
        
    y = copy.deepcopy(x)
    r = np.random.permutation(probdata.n)
    
    if k == 1:             # exchange two random positions
        y.solution[r[0]] = x.solution[r[1]]
        y.solution[r[1]] = x.solution[r[0]]        
    elif k == 2:           # exchange three random positions
        y.solution[r[0]] = x.solution[r[1]]
        y.solution[r[1]] = x.solution[r[2]]
        y.solution[r[2]] = x.solution[r[0]]
    elif k == 3:           # shift positions     
        z = y.solution.pop(r[0])
        y.solution.insert(r[1],z)
    
    return y

probdata= probdef("probdata.csv")
x = sol_inicial(probdata)
print(x.solution)