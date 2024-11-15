import matplotlib.pyplot as plt
import pandas as pd


def plot_melhor_solucao(probdata, solucao):
    # Extraindo as coordenadas das bases e dos ativos
    dados = pd.read_csv(probdata.csv, sep=";", decimal=",", header=None)
    coordenadas_bases = dados[[0, 1]].drop_duplicates().reset_index(drop=True)
    coordenadas_ativos = dados[[2, 3]].drop_duplicates().reset_index(drop=True)

    # Preparando o plot
    plt.figure(figsize=(10, 8))

    # Plotando as bases de manutenção
    plt.scatter(coordenadas_bases[0], coordenadas_bases[1], c='blue', marker='s', s=100, label='Base de Manutenção')

    # Plotando os ativos
    plt.scatter(coordenadas_ativos[2], coordenadas_ativos[3], c='green', marker='o', s=50, label='Ativo')

    # Destacando as bases ocupadas e conectando ativos às equipes
    bases_ocupadas = set()
    for i in range(probdata.m):
        for j in range(probdata.n):
            equipe = solucao[i, j]
            if equipe != 0:
                base_coord = coordenadas_bases.iloc[i].values
                ativo_coord = coordenadas_ativos.iloc[j].values
                # Destacar base ocupada
                if i not in bases_ocupadas:
                    plt.scatter(base_coord[0], base_coord[1], c='red', marker='s', s=150, label='Base Ocupada')
                    bases_ocupadas.add(i)
                # Conectar ativo à base/equipe responsável
                plt.plot([base_coord[0], ativo_coord[0]], [base_coord[1], ativo_coord[1]], 'k--', linewidth=0.5)

    # Configurações finais do plot
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Melhor Solução Encontrada - Localização dos Ativos e Bases de Manutenção')
    plt.legend()
    plt.grid(True)
    plt.show()

