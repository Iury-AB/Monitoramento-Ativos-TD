import numpy as np
import matplotlib.pyplot as plt
import time
import csv

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
    x.w1 = w1
    x.w2 = w2
    # Avalia
    x = fobj_weighted_normalized(x, probdata, w1, w2)

    tempo_inicio = time.time()
    iter_count = 0  # contador de iterações externas

    while True:
        # Imprime a cada iteração do loop externo
        elapsed = time.time() - tempo_inicio
        #print(f"[WeightedSum] Iter={iter_count}, Current Fitness={x.fitness:.4f}, Elapsed={elapsed:.2f}s")

        k = 1
        while k <= kmax:
            # Debug do valor de k
            #print(f"  -> k={k}, best_fitness={x.fitness:.4f}")

            y = shake(x, k, probdata)
            y = fobj_weighted_normalized(y, probdata, w1, w2)


            z = first_improvement_new(y, fobj_weighted_normalized, k, probdata, w1=w1, w2=w2)

            z = fobj_weighted_normalized(z, probdata, w1, w2)

            x, k = neighborhoodChange(x, z, k)

        iter_count += 1

        if time.time() - tempo_inicio > tempo_limite:
            #print(f"[WeightedSum] Tempo limite de {tempo_limite}s atingido. Encerrando loop.")
            break

    return x

# ========== EPSILON-RESTRITO (Pε) ==========

def run_epsilon_restrito(probdata, restrita, eps):
    """
    Roda o RVNS usando fobj_epsilon_restrito com restrição f2(x) <= eps.
    Retorna a melhor solução final encontrada (x).
    """
    x = sol_inicial(probdata)
    x = fobj_epsilon_restrito(x, restrita, probdata, eps)
    x.eps = eps

    tempo_inicio = time.time()
    iter_count = 0

    while True:
        elapsed = time.time() - tempo_inicio
        #print(f"[EpsRestrict] Iter={iter_count}, Current Fitness={x.fitness:.4f}, Elapsed={elapsed:.2f}s")

        k = 1
        while k <= kmax:
            #print(f"  -> k={k}, best_fitness={x.fitness:.4f}")

            y = shake(x, k, probdata)
            y = fobj_epsilon_restrito(y, restrita, probdata, eps)

            z = first_improvement_new(y, fobj_epsilon_restrito, k, probdata, eps=eps, restrita=restrita)
            z = fobj_epsilon_restrito(z, restrita, probdata, eps)

            x, k = neighborhoodChange(x, z, k)

        iter_count += 1

        if time.time() - tempo_inicio > tempo_limite:
            #print(f"[EpsRestrict] Tempo limite de {tempo_limite}s atingido. Encerrando loop.")
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

def salvar_solucoes_em_csv(nd_pw, nd_eps, nome_arquivo):
    """
    Salva as soluções de nd_pw e nd_eps em um arquivo CSV.
    Cada solução inclui a matriz solution (14x125), f1_value e f2_value.

    Parâmetros:
        nd_pw (list): Lista de soluções não-dominadas da abordagem Pw.
        nd_eps (list): Lista de soluções não-dominadas da abordagem Pε.
        nome_arquivo (str): Nome do arquivo CSV para salvar os dados.
    """
    with open(nome_arquivo, mode='w', newline='') as arquivo:
        writer = csv.writer(arquivo)
        
        # Cabeçalhos
        writer.writerow(['Abordagem', 'ID Solucao', 'f1_value', 'f2_value', 'Matriz Solution (14x125)'])
        
        # Escreve as soluções de nd_pw
        for idx, sol in enumerate(nd_pw, start=1):
            matriz_flat = sol.solution.flatten()  # Flatten da matriz 14x125
            linha = ['Pw', idx, sol.f1_val, sol.f2_val] + matriz_flat.tolist()
            writer.writerow(linha)
        
        # Escreve as soluções de nd_eps
        for idx, sol in enumerate(nd_eps, start=1):
            matriz_flat = sol.solution.flatten()  # Flatten da matriz 14x125
            linha = ['Pe', idx, sol.f1_val, sol.f2_val] + matriz_flat.tolist()
            writer.writerow(linha)

if __name__ == "__main__":
    # Abordagem 1: Soma Ponderada
    pesos = [
        (0.0, 1.0),
        (0.1, 0.9),
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2),
        (0.9, 0.1),
        (1.0, 0.0),
    ]

    solutions_pw = []
    for (w1,w2) in pesos:
        for execucao in range(n_execucoes):
            print(f"\n=== Rodando Soma Ponderada com w1={w1}, w2={w2}, Execução={execucao+1} ===")
            sol = run_weighted_sum(probdata, w1, w2)
            solutions_pw.append(sol)

    nd_pw = get_nondominated_set(solutions_pw)

    # Abordagem 2: ε-restrito
    eps_values = [[1000, 2000, 3000, 4000, 5000], [700, 1000, 1300, 1600, 2000]]

    solutions_eps = []
    for func in (1,2):
        for eps in eps_values[func-1]:
            for execucao in range(n_execucoes):
                print(f"\n=== Rodando Eps-Restrito com eps={eps}, Execução={execucao+1} e {func} restrita ===")
                sol = run_epsilon_restrito(probdata, func, eps)
                solutions_eps.append(sol)

    nd_eps = get_nondominated_set(solutions_eps)

    # (1) Lista de soluções dominadas: aquelas que NÃO estão em nd_*
    dominated_pw = [sol for sol in solutions_pw if sol not in nd_pw]
    dominated_eps = [sol for sol in solutions_eps if sol not in nd_eps]

    # (2) Limita as não-dominadas a no máx. 20 (caso queira manter)
    if len(nd_pw) > 20:
        nd_pw = nd_pw[:20]
    if len(nd_eps) > 20:
        nd_eps = nd_eps[:20]

    # (3) Plot de TODAS as soluções, separando por cor/forma se dominadas ou não
    plt.figure(figsize=(8, 6))

    # --- Pw: dominadas ---
    f1_pw_dom = [s.f1_val for s in dominated_pw]
    f2_pw_dom = [s.f2_val for s in dominated_pw]
    plt.scatter(f1_pw_dom, f2_pw_dom, c='pink', marker='o', alpha=0.7, label='Dominadas (Pw)')

    # --- Pw: não-dominadas ---
    f1_pw_nd = [s.f1_val for s in nd_pw]
    f2_pw_nd = [s.f2_val for s in nd_pw]
    plt.scatter(f1_pw_nd, f2_pw_nd, c='red', marker='o', label='Não-Dominadas (Pw)')

    # --- Pε: dominadas ---
    f1_eps_dom = [s.f1_val for s in dominated_eps]
    f2_eps_dom = [s.f2_val for s in dominated_eps]
    plt.scatter(f1_eps_dom, f2_eps_dom, c='lightskyblue', marker='^', alpha=0.7, label='Dominadas (Pε)')

    # --- Pε: não-dominadas ---
    f1_eps_nd = [s.f1_val for s in nd_eps]
    f2_eps_nd = [s.f2_val for s in nd_eps]
    plt.scatter(f1_eps_nd, f2_eps_nd, c='blue', marker='^', label='Não-Dominadas (Pε)')

    # Anotar somente as não-dominadas, se desejar (opcional):
    for s in nd_pw:
        if hasattr(s, 'w1') and hasattr(s, 'w2'):
            plt.annotate(
                f"(w1={s.w1}, w2={s.w2})",
                xy=(s.f1_val, s.f2_val),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='red'
            )
    for s in nd_eps:
        if hasattr(s, 'eps'):
            plt.annotate(
                f"(eps={s.eps})",
                xy=(s.f1_val, s.f2_val),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='blue'
            )

    plt.xlabel('f1(x) - Custo de Deslocamento')
    plt.ylabel('f2(x) - Risco Esperado de Falha')
    plt.title('Todas as Soluções: Dominadas vs. Não-Dominadas (Pw e Pε)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # (4) Plotar as MELHORES soluções não-dominadas (já filtradas) no final
    best_pw = min(nd_pw, key=lambda s: s.fitness)
    best_eps = min(nd_eps, key=lambda s: s.fitness)
    plot_melhor_solucao(probdata, best_pw.solution)
    plot_melhor_solucao(probdata, best_eps.solution)

    salvar_solucoes_em_csv(nd_pw, nd_eps, "solucoes.csv")

