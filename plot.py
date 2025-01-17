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

def plot_solutions_front(solucoes, idx_ahp, survivors, titulo="Fronteira de Soluções"):
    """
    Plota as soluções no plano (f1_value, f2_value),
    destacando a melhor do AHP e as survivors do ELECTRE.

    Parâmetros
    ----------
    solucoes : list
        Lista de objetos (Struct) com atributos f1_value, f2_value etc.
    idx_ahp : int
        Índice da solução escolhida pelo AHP.
    survivors : list of int
        Índices das soluções sobreviventes do ELECTRE I.
    titulo : str
        Título do gráfico.
    """
    plt.figure(figsize=(8,6))
    # Monta arrays de f1 e f2
    f1_vals = [sol.f1_value for sol in solucoes]
    f2_vals = [sol.f2_value for sol in solucoes]

    # Faz scatter de todas as soluções
    # Começamos assumindo que nenhuma é destaque
    plt.scatter(f1_vals, f2_vals, c='gray', marker='o', label='Demais Soluções', alpha=0.6)

    # Destacar as survivors do ELECTRE (em triângulos)
    electre_f1 = [f1_vals[i] for i in survivors]
    electre_f2 = [f2_vals[i] for i in survivors]
    plt.scatter(electre_f1, electre_f2, c='blue', marker='^',
                s=80, label='Survivors ELECTRE', alpha=0.9)

    # Destacar a melhor do AHP (em formato estrela)
    ahp_f1 = f1_vals[idx_ahp]
    ahp_f2 = f2_vals[idx_ahp]
    # Se a melhor do AHP também estiver em survivors, escolhemos outra cor/forma para isso
    if idx_ahp in survivors:
        plt.scatter(ahp_f1, ahp_f2, c='red', marker='*',
                    s=200, label='Melhor AHP (também survivor)', alpha=1.0)
    else:
        plt.scatter(ahp_f1, ahp_f2, c='red', marker='*',
                    s=200, label='Melhor AHP (fora survivors)', alpha=1.0)

    plt.title(titulo)
    plt.xlabel('f1_value (distância total)')
    plt.ylabel('f2_value (risco esperado)')
    plt.legend()
    plt.grid(True)
    plt.show()