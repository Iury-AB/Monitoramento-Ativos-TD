import numpy as np
import matplotlib.pyplot as plt
import time

from heurisitcs import sol_inicial, fobj_1, fobj_2_old, fobj_2, shake, firstImprovement, neighborhoodChange
from problem import probdef, probdef_new, Struct
from plot import plot_melhor_solucao

np.set_printoptions(threshold=np.inf) # diretiva para imprimir todos os elementos de uma matriz

# Parâmetros gerais
probdata = probdef_new()
kmax = 4
tempo_timite = 15
n_execucoes = 5

historicos1 = []
historicos2 = []
melhores_solucoes1 = []
melhores_fitness1 = []
melhores_solucoes2 = []
melhores_fitness2 = []

for execucao in range(n_execucoes):
    # Gera solução inicial
    x = sol_inicial(probdata)
    x2 = sol_inicial(probdata)

    # Avalia solução inicial
    x = fobj_1(x,probdata)
    x2 = fobj_2(x2, probdata)

    # Armazena dados para plot
    historico = Struct()
    historico.sol = []
    historico.fit = []

    historico2 = Struct()
    historico2.sol = []
    historico2.fit = []

    historico.sol.append(x.solution)
    historico.fit.append(x.fitness)

    historico2.sol.append(x2.solution)
    historico2.fit.append(x2.fitness)

    tempo_inicio = time.time()
    print(f'\n--- AVALIAÇÃO DA SOLUÇÃO INICIAL ---')
    print(f'fitness 1(x) = {x.fitness:.1f}\n')
    print(f'fitness 2(x) = {x2.fitness:.1f}\n')

    # Otimização para f1
    while True:
        k = 1
        while k <= kmax:
            y = shake(x,k,probdata)
            y = fobj_1(y,probdata)
            z = firstImprovement(y, fobj_1, k, probdata)
            z = fobj_1(z, probdata)
            x,k = neighborhoodChange(x,z,k)
            historico.sol.append(x.solution)
            historico.fit.append(x.fitness)
        if(time.time() - tempo_inicio > tempo_timite):
            break

    historicos1.append(historico)
    melhores_solucoes1.append(x.solution)
    melhores_fitness1.append(x.fitness)

    tempo_inicio = time.time()

    # Otimização para f2
    while True:
        k = 1
        while k <= kmax:
            y2 = shake(x2,k,probdata)
            y2 = fobj_2(y2, probdata)
            z2 = firstImprovement(y2, fobj_2, k, probdata)
            z2 = fobj_2(z2, probdata)
            x2,k = neighborhoodChange(x2,z2,k)
            historico2.sol.append(x2.solution)
            historico2.fit.append(x2.fitness)
        if time.time() - tempo_inicio > tempo_timite:
            break

    historicos2.append(historico2)
    melhores_solucoes2.append(x2.solution)
    melhores_fitness2.append(x2.fitness)

    print(f'\n--- EXECUÇÃO {execucao + 1} ---')
    print(f'fitness 1(x) = {x.fitness:.1f}\n')
    print(f'fitness 2(x) = {x2.fitness:.1f}\n')

indice_melhor1 = np.argmin(melhores_fitness1)
melhor_solucao_global1 = melhores_solucoes1[indice_melhor1]
melhor_fitness_global1 = melhores_fitness1[indice_melhor1]

indice_melhor2 = np.argmin(melhores_fitness2)
melhor_solucao_global2 = melhores_solucoes2[indice_melhor2]
melhor_fitness_global2 = melhores_fitness2[indice_melhor2]

print('\n--- MELHOR SOLUÇÃO GLOBAL ENCONTRADA para f1 ---\n')
print(f'fitness f1(x) = {melhor_fitness_global1:.1f}\n')

print('\n--- MELHOR SOLUÇÃO GLOBAL ENCONTRADA para f2 ---\n')
print(f'fitness f2(x) = {melhor_fitness_global2:.1f}\n')

# Plot das curvas de convergência f1
plt.figure()
for execucao, historico in enumerate(historicos1):
    s = len(historico.fit)
    plt.plot(np.linspace(0,s-1,s), historico.fit, label=f'Execução {execucao + 1}')
plt.title('Evolução da qualidade da solução (f1)')
plt.xlabel('Número de avaliações')
plt.ylabel('fitness(x)')
plt.legend()

# Plot das curvas de convergência f2
plt.figure()
for execucao, historico in enumerate(historicos2):
    s = len(historico.fit)
    plt.plot(np.linspace(0,s-1,s), historico.fit, label=f'Execução {execucao + 1}')
plt.title('Evolução da qualidade da solução (f2)')
plt.xlabel('Número de avaliações')
plt.ylabel('fitness(x)')
plt.legend()

# Plota a melhor solução global
plot_melhor_solucao(probdata, melhor_solucao_global1)
plot_melhor_solucao(probdata, melhor_solucao_global2)