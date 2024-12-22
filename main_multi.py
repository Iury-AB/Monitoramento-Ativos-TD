import numpy as np
import matplotlib.pyplot as plt
import time

from heurisitcs import (
    shake,
    first_improvement_new,
    fobj_weighted_normalized,
    neighborhoodChange,
    sol_inicial,
    fobj_epsilon_restrito
)
from problem import probdef_new, Struct
from plot import plot_melhor_solucao

probdata = probdef_new()
kmax = 4
tempo_limite = 15
n_execucoes = 5

# ========== SOMA PONDERADA (Pw) ==========

def run_weighted_sum(probdata, w1, w2):

    # Gera sol inicial
    x = sol_inicial(probdata)
    # Avalia
    x = fobj_weighted_normalized(x, probdata, w1, w2)

    tempo_inicio = time.time()
    iter_count = 0  # contador de iterações externas

    while True:
        # Imprime a cada iteração do loop externo
        elapsed = time.time() - tempo_inicio
        print(f"[WeightedSum] Iter={iter_count}, Current Fitness={x.fitness:.4f}, Elapsed={elapsed:.2f}s")

        k = 1
        while k <= kmax:
            # Debug do valor de k
            print(f"  -> k={k}, best_fitness={x.fitness:.4f}")

            y = shake(x, k, probdata)
            y = fobj_weighted_normalized(y, probdata, w1, w2)


            z = first_improvement_new(y, fobj_weighted_normalized, k, probdata, w1=w1, w2=w2)

            z = fobj_weighted_normalized(z, probdata, w1, w2)

            x, k = neighborhoodChange(x, z, k)

        iter_count += 1

        if time.time() - tempo_inicio > tempo_limite:
            print(f"[WeightedSum] Tempo limite de {tempo_limite}s atingido. Encerrando loop.")
            break

    return x

# ========== EPSILON-RESTRITO (Pε) ==========

def run_epsilon_restrito(probdata, eps):
    """
    Roda o RVNS usando fobj_epsilon_restrito com restrição f2(x) <= eps.
    Retorna a melhor solução final encontrada (x).
    """
    x = sol_inicial(probdata)
    x = fobj_epsilon_restrito(x, probdata, eps)

    tempo_inicio = time.time()
    iter_count = 0

    while True:
        elapsed = time.time() - tempo_inicio
        print(f"[EpsRestrict] Iter={iter_count}, Current Fitness={x.fitness:.4f}, Elapsed={elapsed:.2f}s")

        k = 1
        while k <= kmax:
            print(f"  -> k={k}, best_fitness={x.fitness:.4f}")

            y = shake(x, k, probdata)
            y = fobj_epsilon_restrito(y, probdata, eps)

            z = first_improvement_new(y, fobj_epsilon_restrito, k, probdata, eps=eps)
            z = fobj_epsilon_restrito(z, probdata, eps)

            x, k = neighborhoodChange(x, z, k)

        iter_count += 1

        if time.time() - tempo_inicio > tempo_limite:
            print(f"[EpsRestrict] Tempo limite de {tempo_limite}s atingido. Encerrando loop.")
            break

    return x

# ========== Rotina principal para gerar as fronteiras ==========

def is_dominated(solA, solB):
    cond1 = (solB.f1_val <= solA.f1_val)
    cond2 = (solB.f2_val <= solA.f2_val)
    strict = (solB.f1_val < solA.f1_val) or (solB.f2_val < solA.f2_val)
    return cond1 and cond2 and strict

def get_nondominated_set(solutions):
    nd_set = []
    for sA in solutions:
        dominated = False
        for sB in solutions:
            if is_dominated(sA, sB):
                dominated = True
                break
        if not dominated:
            nd_set.append(sA)
    return nd_set

if __name__ == "__main__":
    # Abordagem 1: Soma Ponderada
    pesos = [(0.0,1.0), (0.25,0.75), (0.5,0.5), (0.75,0.25), (1.0,0.0)]

    solutions_pw = []
    for (w1,w2) in pesos:
        for execucao in range(n_execucoes):
            print(f"\n=== Rodando Soma Ponderada com w1={w1}, w2={w2}, Execução={execucao+1} ===")
            sol = run_weighted_sum(probdata, w1, w2)
            solutions_pw.append(sol)

    nd_pw = get_nondominated_set(solutions_pw)

    # Abordagem 2: ε-restrito
    eps_values = [700, 1000, 1300, 1600, 2000]
    solutions_eps = []
    for eps in eps_values:
        for execucao in range(n_execucoes):
            print(f"\n=== Rodando Eps-Restrito com eps={eps}, Execução={execucao+1} ===")
            sol = run_epsilon_restrito(probdata, eps)
            solutions_eps.append(sol)

    nd_eps = get_nondominated_set(solutions_eps)

    if len(nd_pw) > 20:
        nd_pw = nd_pw[:20]
    if len(nd_eps) > 20:
        nd_eps = nd_eps[:20]

    plt.figure()
    f1_pw = [s.f1_val for s in nd_pw]
    f2_pw = [s.f2_val for s in nd_pw]
    plt.scatter(f1_pw, f2_pw, c='red', label='Soma Ponderada')

    f1_eps = [s.f1_val for s in nd_eps]
    f2_eps = [s.f2_val for s in nd_eps]
    plt.scatter(f1_eps, f2_eps, c='blue', label='Eps-Restrito')

    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.title('Fronteiras Não-Dominadas (Pw e Pε)')
    plt.legend()
    plt.grid(True)
    plt.show()

    best_pw = min(nd_pw, key=lambda s: s.fitness)
    best_eps = min(nd_eps, key=lambda s: s.fitness)
    plot_melhor_solucao(probdata, best_pw.solution)
    plot_melhor_solucao(probdata, best_eps.solution)
