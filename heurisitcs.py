import numpy as np
import copy
import math
from random import sample
from collections import Counter
from itertools import combinations
from problem import Struct
from utils import base_mais_proxima


def fobj_1(x, probdata):
    dist_soma = 0
    xyh = x.solution
    for j in range(0, probdata.n):
        for i in range(0, probdata.m):
            if (xyh[i][j] != 0):
                dist_soma += probdata.d.at[i, j]
    x.fitness = dist_soma
    return x


def fobj_2_old(x, probdata):
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
    carga_max = max(carga_eq)
    carga_min = min(carga_eq)
    x.fitness = carga_max - carga_min
    return x


def fobj_2(x, probdata):
    # xyh = x.solution (m x n)
    # p_i = probdata.p[i]
    # d_ij = probdata.d.at[i,j]
    dist_soma = 0
    xyh = x.solution
    for j in range(probdata.n):       # para cada ativo
        for i in range(probdata.m):   # para cada base
            if xyh[i][j] != 0:
                dist_soma += probdata.p[j] * probdata.d.at[i,j]
    x.fitness = dist_soma
    return x

def normalizada(f, x, min, max):
    f_normal = (f(x)-min)/(max-min)
    return f_normal

def sol_inicial(probdata, apply_constructive_heuristic=False):
    x = Struct()
    xyh = np.zeros((probdata.m, probdata.n), dtype=np.int8)
    if apply_constructive_heuristic == False:
        min_val = math.ceil(probdata.eta * probdata.n / probdata.s)
        media = math.floor(probdata.n / probdata.s)
        sorteado = sample(range(min_val, media), 1)
        resp = sorteado[0]
        x.resp = resp
        equipes_sorteadas = sample(range(np.int8(1), probdata.s + np.int8(1)), probdata.s)
        bases_sorteadas = sample(range(0, probdata.m), probdata.s)
        x.bases_ocupadas = set(bases_sorteadas)
        for i, equipe in enumerate(equipes_sorteadas):
            if (i == 0):
                xyh[bases_sorteadas[i], i:resp] = equipes_sorteadas[i]
            elif (i == len(equipes_sorteadas) - 1):
                xyh[bases_sorteadas[i], (i) * resp:probdata.n] = equipes_sorteadas[i]
            else:
                xyh[bases_sorteadas[i], i * resp:(i + 1) * resp] = equipes_sorteadas[i]
        x.solution = xyh
    else:
        bases_mais_proximas = {}
        for ativo in range(probdata.n):
            bases_mais_proximas.update({ativo: np.argmin(probdata.d[ativo])})

        bases = bases_mais_proximas.values()
        freq_bases_mais_proximas = Counter(bases)
        bases_ordenadas = dict(sorted(freq_bases_mais_proximas.items(), key=lambda item: item[1], reverse=True))
        bases_sorteadas = list(bases_ordenadas.keys())[:probdata.s]

        min_val = math.ceil(probdata.eta * probdata.n / probdata.s)
        x.resp = min_val
        x.bases_ocupadas = set(bases_sorteadas)

        for ativo in bases_mais_proximas.keys():
            base = bases_mais_proximas[ativo]
            if base in bases_sorteadas:
                xyh[base, ativo] = bases_sorteadas.index(base) + 1
            else:
                i = np.argmin(probdata.d[ativo])
                distancia = probdata.d.at[i, ativo]
                distancia_bases_sorteadas = {}
                for b in bases_sorteadas:
                    distancia_bases_sorteadas.update({b: probdata.d.at[b, ativo]})
                base = base_mais_proxima(distancia_bases_sorteadas, distancia)
                xyh[base, ativo] = bases_sorteadas.index(base) + 1

        x.solution = xyh
    return x


def neighborhoodChange(x, y, k):
    if y.fitness < x.fitness:
        x = copy.deepcopy(y)
        k = 1
    else:
        k += 1
    return x, k


def troque_coluna(x, y, probdata):
    n = sample(range(0, probdata.n), 2)
    y.solution[:, n[0]] = x.solution[:, n[1]]
    y.solution[:, n[1]] = x.solution[:, n[0]]
    return y


def troque_linha(x, y, probdata):
    m = sample(range(0, probdata.m), 2)
    while y.bases_ocupadas.isdisjoint(m) or y.bases_ocupadas.issuperset(m):
        m = sample(range(0, probdata.m), 2)

    y.solution[m[0], :] = x.solution[m[1], :]
    y.solution[m[1], :] = x.solution[m[0], :]

    intersecao = y.bases_ocupadas.intersection(m)
    if len(intersecao) == 1:
        diff = y.bases_ocupadas.difference(m)
        m.remove(list(intersecao)[0])
        uniao = diff.union(set(m))
        y.bases_ocupadas.clear()
        y.bases_ocupadas = uniao
    return y


combinacao_ativo = list(combinations(range(125), 2))
combinacao_base = list(combinations(range(14), 2))
combinacao_ativo_base = []
for ativo in combinacao_ativo:
    for base in combinacao_base:
        combinacao_ativo_base.append((ativo, base))


def shake(x, k, probdata):
    y = copy.deepcopy(x)
    if k == 1:
        y = troque_coluna(x, y, probdata)
    elif k == 2:
        y = troque_linha(x, y, probdata)
    elif k == 3:
        z = troque_linha(x, y, probdata)
        y = copy.deepcopy(z)
        y = troque_coluna(z, y, probdata)
    elif k == 4:
        i_n0 = 0
        i_n1 = 0
        achou = False
        while not achou:
            n = sample(range(0, probdata.n), 2)
            for i, base in enumerate(x.bases_ocupadas):
                if x.solution[base, n[0]] != 0:
                    i_n0 = base
                if x.solution[base, n[1]] != 0:
                    i_n1 = base
                if x.solution[i_n0, n[0]] != x.solution[i_n1, n[1]] and x.solution[i_n0, n[0]] != 0 and x.solution[
                    i_n1, n[1]] != 0:
                    n0_carga = np.where(x.solution[i_n1] != 0)[0]
                    if len(n0_carga) > x.resp - 1:
                        achou = True
                        y.solution[i_n0, n[1]] = x.solution[i_n0, n[0]]
                        y.solution[i_n1, n[1]] = 0
    return y


def vizinhanca(x, k, i):
    y = copy.deepcopy(x)
    if k == 1:
        dupla = combinacao_ativo[i]
        y.solution[:, dupla[0]] = x.solution[:, dupla[1]]
        y.solution[:, dupla[1]] = x.solution[:, dupla[0]]
    elif k == 2:
        dupla = combinacao_base[i]
        y.solution[dupla[0], :] = x.solution[dupla[1], :]
        y.solution[dupla[1], :] = x.solution[dupla[0], :]
    elif k == 3:
        i_n0 = 0
        i_n1 = 0
        achou = False
        while not achou:
            dupla = combinacao_ativo[i]
            for ix, base in enumerate(x.bases_ocupadas):
                if x.solution[base, dupla[0]] != 0:
                    i_n0 = base
                if x.solution[base, dupla[1]] != 0:
                    i_n1 = base
                if x.solution[i_n0, dupla[0]] != x.solution[i_n1, dupla[1]] and x.solution[i_n0, dupla[0]] != 0 and \
                        x.solution[i_n1, dupla[1]] != 0:
                    n0_carga = np.where(x.solution[i_n1] != 0)[0]
                    if len(n0_carga) > x.resp - 1:
                        achou = True
                        y.solution[i_n0, dupla[1]] = x.solution[i_n0, dupla[0]]
                        y.solution[i_n1, dupla[1]] = 0
    return y


def firstImprovement(x, obj, k, probdata):
    while True:
        y = x
        i = 0
        vizinhos = []
        n_viz = 40
        for j in range(n_viz):
            vizinhos.append(shake(x, k, probdata))
        while ((x.fitness >= y.fitness) and i != n_viz):
            xi = vizinhos[i]
            xi = obj(xi, probdata)
            x = xi if (xi.fitness < x.fitness) else x
            i += 1
        if (x.fitness >= y.fitness):
            break
    return x