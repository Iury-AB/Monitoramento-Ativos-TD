'''
Importa os módulos usados
'''
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
from random import sample
import math
from itertools import combinations
from plot import plot_melhor_solucao
import time


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
    n_equipes_base = 0
    for j in range(0, probdata.n):
        for i in range(0, probdata.m):
            if(xyh[i][j] != 0):
                dist_soma +=  probdata.d[i][j]
                n_equipes_base += 1

    x.fitness = dist_soma

    return x

def fobj_2 (x, probdata):
    carga_eq = np.zeros(3)
    xyh = x.solution
    for j in range(0,probdata.n):
        for i in range(0,probdata.m):
            if(xyh[i][j] == 1):
                carga_eq[0] += 1
                break

            elif(xyh[i][j] == 2):
                carga_eq[1] += 1
                break

            elif(xyh[i][j] == 3):
                carga_eq[2] += 1
                break

    carga_max = max(carga_eq)
    carga_min = min(carga_eq)
    x.fitness = carga_max - carga_min
    return x

'''
Define os dados de uma instância arbitrária do problema
'''
def probdef(s=3,eta=0.2,csv="probdata.csv"):

    # n: número de ativos
    # m: número de bases
    # s: número de equipes
    # eta: percentual de responsabilidade das esquipes
    # csv: caminho do arquivo contendo as coordenadas e distâncias.
        
   
    distancias = carregar_matriz_distancias(csv)
    
    m,n = distancias.shape #m bases e n ativos
   
    probdata = Struct()
    probdata.eta = eta
    probdata.n = n
    probdata.m = m
    probdata.s = s
    probdata.d = distancias
    probdata.csv = csv
        
    return probdata

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
    if apply_constructive_heuristic == False:    
        # Constrói solução inicial aleatoriamente
        x = Struct()
        xyh = np.zeros((probdata.m,probdata.n), dtype=int) # cria uma matriz de elementos de mesma forma do arquivo csv atribuindo valores zero
        min = math.ceil(probdata.eta*probdata.n/probdata.s) # Calcula a R6
        media = math.floor(probdata.n/probdata.s)
        sorteado = sample(range(min, media), 1)
        resp = sorteado[0]
        x.resp=resp
        equipes_sorteadas = sample(range(1, probdata.s + 1), probdata.s) # sorteia aleatoriamente s equipes
        bases_sorteadas = sample(range(0,probdata.m),probdata.s) # sorteia aleatoriamente s bases
        x.bases_ocupadas = set(bases_sorteadas)

        for i,equipe in enumerate(equipes_sorteadas):
            if (i == 0):
                xyh[bases_sorteadas[i],i:resp] = equipes_sorteadas[i] # Atribui os resp elementos da base de índice 0 à equipe i
            elif (i == len(equipes_sorteadas) - 1):
                xyh[bases_sorteadas[i],(i)*resp:probdata.n] = equipes_sorteadas[i] # Atribui à última equipe os últimoa ativos à base i
            else:
                xyh[bases_sorteadas[i],i*resp:(i+1)*resp] = equipes_sorteadas[i] # Atribui os resp elementos seguintes da base i das bases sorteadas à equipe i
            
        x.solution = xyh
    
    else:
        ## Constrói solução inicial usando uma heurística construtiva
        x = Struct()
        xyh = np.zeros((probdata.m,probdata.n), dtype=int)
        media = math.floor(probdata.n/probdata.s)
        resp = media
        x.resp = resp
        equipes_sorteadas = sample(range(1, probdata.s + 1), probdata.s) # sorteia aleatoriamente s equipes
        #bases_sorteadas = sample(range(0,probdata.m),probdata.s) # sorteia aleatoriamente s bases
        ativos = np.argsort(probdata.d.var(axis=0))    # ativos ordenadas de acordo com a correlaçao das distânciais
        bases = np.argsort(probdata.d.var(axis=1))    # ativos ordenadas de acordo com a correlaçao das distânciais
        bases_sorteadas = bases[0:probdata.s]
        x.bases_ocupadas = set(bases_sorteadas)

        j = 0
        i = 0

        for ativo in ativos[::-1]:               

            xyh[bases_sorteadas[i],ativo] = equipes_sorteadas[i] # Atribui os resp elementos seguintes da base i das bases sorteadas à equipe i
            
            j = j + 1

            if (i == 0 and j > resp) or (j >  (probdata.s -1)*resp and i < len(equipes_sorteadas)-1):
                i = i + 1 

        x.solution = xyh
        
        
    return x


'''
Implementa a função neighborhoodChange
'''
def neighborhoodChange(x, y, k):
    
    if y.fitness < x.fitness:
        x = copy.deepcopy(y)
        k = 1
        print("vizinhanca {}".format(k))
    else:
        k += 1
        print("vizinhanca {}".format(k))
        
    return x, k

def troque_coluna(x, y, probdata):
    n= sample(range(0, probdata.n), 2) # sorteia 2 ativos para permutar
    y.solution[:,n[0]] = x.solution[:,n[1]]
    y.solution[:,n[1]] = x.solution[:,n[0]] 
    #print(x.solution)
    #print(y.solution)
    #print('mudança')

    return y

def troque_linha (x, y, probdata):
    m= sample(range(0, probdata.m), 2) # realocação de equipes entre duas bases
    while y.bases_ocupadas.isdisjoint(m) or y.bases_ocupadas.issuperset(m):
        m= sample(range(0, probdata.m), 2)

    y.solution[m[0],:] = x.solution[m[1],:]
    y.solution[m[1],:] = x.solution[m[0],:]
    #print(x.solution)
    #print(y.solution)
    #print('mudança')
        
    intersecao = y.bases_ocupadas.intersection(m)
        

    if len(intersecao) == 1 :
        diff = y.bases_ocupadas.difference(m)
        m.remove(list(intersecao)[0])            
        uniao = diff.union(set(m))
        y.bases_ocupadas.clear()
        y.bases_ocupadas = uniao
    return y

'''
Implementa a função shake
'''
combinacao_ativo = list(combinations(range(125), 2))
combinacao_base = list(combinations(range(14), 2))
combinacao_ativo_base = []
for ativo in combinacao_ativo:
    for base in combinacao_base:
        combinacao_ativo_base.append((ativo, base))

def shake(x, k, probdata):
        
    y = copy.deepcopy(x)  
        
    if k == 1:             # Pode ou não alterar aleatoriamente atvios entre equipes
        y = troque_coluna(x,y, probdata) 

    elif k == 2:           # altera aleatoriamente até duas equipes de bases
        y = troque_linha(x, y, probdata)        
        
    elif k == 3: # altera aleatoriamente uma equipe de ativo e base
        z = troque_linha(x, y, probdata)
        y = copy.deepcopy(z)
        y = troque_coluna(z, y, probdata)
    elif k == 4: # poe 2 equipes na mesma base
        pass
    elif k == 5: # remove 2 equipe da mesma base
        pass
    elif k == 6:
        i_n0 = 0
        i_n1 = 0
        achou = False
        while not achou:
            n = sample(range(0,probdata.n),2)
            for i,base in enumerate(x.bases_ocupadas):
                if x.solution[base,n[0]]!=0 :
                    i_n0 = base
                  

                if x.solution[base,n[1]]!=0:
                    i_n1 = base

                if x.solution[i_n0,n[0]] != x.solution[i_n1,n[1]] and x.solution[i_n0,n[0]] !=0 and x.solution[i_n1,n[1]] !=0 :
                    n0_carga = np.where(x.solution[i_n1]!=0)[0]
                    if len(n0_carga) > x.resp - 1:
                        achou=True
                        y.solution[i_n0,n[1]] = x.solution[i_n0,n[0]]
                        y.solution[i_n1,n[1]] = 0


    return y

def vizinhanca(x, k, i):
        
    y = copy.deepcopy(x)
    
    if k == 1:             # Pode ou não alterar aleatoriamente atvios entre equipes
        dupla = combinacao_ativo[i]
        y.solution[:,dupla[0]] = x.solution[:,dupla[1]]
        y.solution[:,dupla[1]] = x.solution[:,dupla[0]] 

    elif k == 2:           # altera aleatoriamente até duas equipes de bases
        dupla = combinacao_base[i]
        y.solution[dupla[0],:] = x.solution[dupla[1],:]
        y.solution[dupla[1],:] = x.solution[dupla[0],:]
        
    elif k == 3: # altera aleatoriamente uma equipe de ativo e base
        return y
        dupla = combinacao_ativo_base[i]
        # Troca linhas diretamente
        y.solution[dupla[1][0], :], y.solution[dupla[1][1], :] = (
            y.solution[dupla[1][1], :].copy(),
            y.solution[dupla[1][0], :].copy(),
        )

        # Troca colunas diretamente
        y.solution[:, dupla[0][0]], y.solution[:, dupla[0][1]] = (
            y.solution[:, dupla[0][1]].copy(),
            y.solution[:, dupla[0][0]].copy(),
        )
    elif k == 4: # poe 2 equipes na mesma base
        pass
    elif k == 5: # remove 2 equipe da mesma base
        pass
    elif k == 6:
        i_n0 = 0
        i_n1 = 0
        achou = False
        while not achou:
            dupla = combinacao_ativo[i]
            for i,base in enumerate(x.bases_ocupadas):
                if x.solution[base,dupla[0]]!=0 :
                    i_n0 = base
                  
                if x.solution[base,dupla[1]]!=0:
                    i_n1 = base

                if x.solution[i_n0,dupla[0]] != x.solution[i_n1,dupla[1]] and x.solution[i_n0,dupla[0]] !=0 and x.solution[i_n1,dupla[1]] !=0 :
                    n0_carga = np.where(x.solution[i_n1]!=0)[0]
                    if len(n0_carga) > x.resp - 1:
                        achou=True
                        y.solution[i_n0,dupla[1]] = x.solution[i_n0,dupla[0]]
                        y.solution[i_n1,dupla[1]] = 0

    return y

def firstImprovement(x, obj, k, probdata):
    tam_k = 0
    if k == 1:
        tam_k = len(combinacao_ativo)
    elif k == 2:
        tam_k = len(combinacao_base)
    elif k == 3:
        return x
    elif k == 6:
        tam_k = len(combinacao_ativo)

    while (True):
        y=x
        i = 0
        while ((x.fitness >= y.fitness) and i != tam_k):
            xi = vizinhanca(x, k, i)
            xi = obj(xi, probdata)
            x = xi if (xi.fitness < x.fitness) else x
            i += 1 
            #print("{}\n".format(i))
            #print("{}\n".format(x.fitness))

        if(x.fitness >= y.fitness):
            print("{}".format(x.fitness))
            break
    return x

'''
Implementa uma metaheurística RVNS
'''

# Contador do número de soluções candidatas avaliadas
num_sol_avaliadas = 0
num_sol2_avaliadas = 0

# Máximo número de soluções candidatas avaliadas
max_num_sol_avaliadas = 10000

# Número de estruturas de vizinhanças definidas
kmax = 3

# Faz a leitura dos dados da instância do problema
probdata = probdef()

# Gera solução inicial
x = sol_inicial(probdata, True)
x2 = sol_inicial(probdata, True)

# Avalia solução inicial
x = fobj_1(x,probdata)
x2 = fobj_2(x2, probdata)
num_sol_avaliadas += 1
num_sol2_avaliadas += 1

# Armazena dados para plot
historico = Struct()
historico.sol = []
historico.fit = []
historico.sol.append(x.solution)
historico.fit.append(x.fitness)

historico2 = Struct()
historico2.sol = []
historico2.fit = []
historico2.sol.append(x2.solution)
historico2.fit.append(x2.fitness)

tempo_inicio = time.time()
tempo_timite = 30

# Ciclo iterativo do método
while True:
    
    k = 1
    while k <= kmax:
        
        # Gera uma solução candidata na k-ésima vizinhança de x
        y = shake(x,k,probdata)
        y = fobj_1(y,probdata)
        z = firstImprovement(y, fobj_1, k, probdata)
        z = fobj_1(z, probdata)
        
        # Atualiza solução corrente e estrutura de vizinhança (se necessário)
        x,k = neighborhoodChange(x,z,k)
        
        # Armazena dados para plot
        historico.sol.append(x.solution)
        historico.fit.append(x.fitness)
    
    if(time.time() - tempo_inicio > tempo_timite):
        break

tempo_inicio = time.time()

while True:
    
    k = 1
    while k <= kmax:
        
        # Gera uma solução candidata na k-ésima vizinhança de x        
        y2 = shake(x2,k,probdata)
        y2 = fobj_2(y2,probdata)
        z2 = firstImprovement(y2, fobj_2, k, probdata)
        z2 = fobj_2(z2, probdata)
        num_sol2_avaliadas += 1
        
        # Atualiza solução corrente e estrutura de vizinhança (se necessário)
        x2,k = neighborhoodChange(x2,z2,k)
        
        # Armazena dados para plot
        historico2.sol.append(x2.solution)
        historico2.fit.append(x2.fitness)

    if(time.time() - tempo_inicio > tempo_timite):
        break

print('\n--- SOLUÇÃO INICIAL CONSTRUÍDA ---\n')
print('Sequência de tarefas atribuídas aos agentes:\n')
print('x = {}\n'.format(historico.sol[0]))
print('fitness(x) = {:.1f}\n'.format(historico.fit[0]))

print('\n--- MELHOR SOLUÇÃO ENCONTRADA ---\n')
print('Sequência de tarefas atribuídas aos agentes:\n')
print('x = {}\n'.format(x.solution))
print('fitness(x) = {:.1f}\n'.format(x.fitness))
print('Valor de f2 para a solucao:\n')
print('f2={}.'.format(fobj_2(x,probdata).fitness))

plt.figure()
s = len(historico.fit)
plt.plot(np.linspace(0,s-1,s),historico.fit,'k-')
plt.title('Evolução da qualidade da solução')
plt.xlabel('Número de avaliações')
plt.ylabel('fitness(x)')

plt.figure()
s2 = len(historico2.fit)
plt.plot(np.linspace(0,s2-1,s2),historico2.fit,'k-')
plt.title('Evolução da qualidade da solução 2')
plt.xlabel('Número de avaliações')
plt.ylabel('fitness(x)')
plot_melhor_solucao(probdata,x.solution)
plt.show()

