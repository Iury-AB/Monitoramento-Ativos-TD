'''
Importa os módulos usados
'''
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd

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

def fobj_1(xyh, d):
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
    
    for i in range(0, m):
        for j in range(0, n):
            if(xyh[i][j] != 0):
                dist_soma +=  d[i][j]

    return dist_soma

'''
Define os dados de uma instância arbitrária do problema
'''
def probdef(n=50):

    # n: número de ativos
    # m: número de bases
    # s: número de equipes
        
   
    distancias = carregar_matriz_distancias("probdata.csv")
    print(distancias[0][0])

    n = 125 #ativos
    m = 4 #bases
    s = 3 #equipes
    eta = 0.2
    
    xij = np.zeros(n, dtype=int)
    yjk = np.zeros(m, dtype=int)
    hik = np.zeros(n, dtype=int)

   
    probdata = Struct()
    probdata.eta = eta
    probdata.n = n
    probdata.m = m
    probdata.s = s
    probdata.d = distancias
        
    return probdata

'''
Implementa uma solução inicial para o problema
'''
def sol_inicial(probdata,apply_constructive_heuristic):
    
    '''
    Matriz solução: e = [  b1 b2 ... bj ... bm
                          a1
                          a2
                          ...
                          ai
                          ...
                          an                    ]
    ''' 
    if apply_constructive_heuristic == False:        
        # Constrói solução inicial aleatoriamente
        x = Struct()
        e = np.zeros((n,m), dtype=int) # cria uma matriz de elementos de mesma forma do arquivo csv atribuindo valores zero
        resp=np.ceil(probdata.eta*probdata.n/probdata.s) # Calcula a R6
        e[0:resp,0]=1 # Atribui os resp elementos da base 0 à equipe 1
        e[resp:2*resp,1]=2 # Atribui os resp elementos seguintes da base 1 à equipe 2
        e[2*resp:n,2]=3 # Atribui os elementos restantes da base 2 à equipe 3
                
        x.solution = e
    
    else:
        ## Constrói solução inicial usando uma heurística construtiva
        x = Struct()
        x.solution = np.zeros((n,m), dtype=int)
        job = np.argsort(probdata.d[:,4].corr(axis=1))    # ativos ordenadas de acordo com a correlaçao das distânciais
        for ativo in job[::-1]:        
            base = np.argmin(probdata.d[:,ativo]) # atribui as tarefas em ordem decrescente de variância ao agente de menor custo
            x.solution.insert(base,ativo)
        
    return x




#[base][ativo] = distancia entre base e ativo
d = carregar_matriz_distancias("probdata.csv")

m,n = d.shape #m bases e n ativos
s = 3 #equipes

#matriz com as variaveis de decisão do problema
#[base][ativo] = equipe responsavel pelo ativo
xyh = np.zeros((m, n), dtype=int) 


