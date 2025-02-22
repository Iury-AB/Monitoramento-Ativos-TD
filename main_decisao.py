import numpy as np

from electre_plot import plot_matrix_values
from problem import probdef_new, Struct
import pandas as pd
from geopy.distance import geodesic
from heurisitcs import fobj_1, fobj_2
import json
from plot import plot_melhor_solucao, plot_solutions_front
from ahp_plot import plotar_tabela_comparacao


def ler_solucoes_do_csv(nome_arquivo):
    """
    Lê um arquivo CSV contendo soluções com f1_value, f2_value e matriz solution no formato JSON.
    Filtra soluções duplicadas com base em f1_value e f2_value.
    Retorna uma lista de objetos que armazenam essas informações.

    Parâmetros:
        nome_arquivo (str): Nome do arquivo CSV contendo as soluções.

    Retorno:
        list: Lista de objetos contendo f1_value, f2_value e a matriz solution.
    """
    # Lê o CSV em um DataFrame
    df = pd.read_csv(nome_arquivo, header=0)  # Supondo que o CSV tem cabeçalho

    # Remove duplicatas com base em 'f1_value' e 'f2_value'
    df = df.drop_duplicates(subset=['f1_value', 'f2_value'], keep='first')

    solucoes = []
    for _, row in df.iterrows():
        f1_value = float(row['f1_value'])
        f2_value = float(row['f2_value'])

        # Converte a matriz JSON de volta para um array numpy
        matriz_json = row['Matriz Solution (14x125)']
        matriz = np.array(json.loads(matriz_json))  # Reconstrói a matriz a partir do JSON

        # Cria o objeto solução
        solucao = Struct()
        solucao.f1_value = f1_value
        solucao.f2_value = f2_value
        solucao.solution = matriz
        solucoes.append(solucao)

    return solucoes


def construir_matriz_bases(caminho_csv):
    """
    Lê um arquivo CSV com separador ';' e números decimais com ',' e constrói uma matriz 
    com as latitudes e longitudes únicas das bases como floats.

    Parâmetros:
        caminho_csv (str): Caminho para o arquivo CSV sem cabeçalho.

    Retorno:
        numpy.ndarray: Matriz 14x2 com as latitudes e longitudes únicas das bases em formato float.
    """
    import pandas as pd
    import numpy as np

    # Lê o CSV sem cabeçalho, com separador ';' e converte ',' em '.' para leitura correta como float
    df = pd.read_csv(
        caminho_csv,
        sep=';',
        header=None,
        names=['lat_base', 'lon_base', 'lat_ativo', 'lon_ativo', 'distancia'],
        converters={
            'lat_base': lambda x: float(x.replace(',', '.')),
            'lon_base': lambda x: float(x.replace(',', '.')),
        }
    )

    # Filtra latitudes e longitudes únicas das bases
    bases_unicas = df[['lat_base', 'lon_base']].drop_duplicates().reset_index(drop=True)

    # Converte para matriz numpy de float
    matriz_bases = bases_unicas.to_numpy(dtype=float)

    # Verifica se há exatamente 14 bases
    if len(bases_unicas) != 14:
        raise ValueError(f"Esperado 14 bases, mas foram encontradas {len(bases_unicas)}. Verifique os dados no CSV.")

    return matriz_bases


def calcular_distancia_bases(coord_base1, coord_base2):
    """
    Calcula a distância entre duas bases utilizando as coordenadas de latitude e longitude.

    Parâmetros:
        coord_base1 (tuple): Coordenadas da base 1 no formato (latitude, longitude).
        coord_base2 (tuple): Coordenadas da base 2 no formato (latitude, longitude).

    Retorno:
        float: Distância em quilômetros entre as duas bases.
    """
    # Converte as coordenadas em tuplas e calcula a distância geodésica
    distancia = geodesic(coord_base1, coord_base2).kilometers
    return distancia


def robustez_indisponibilidade(x, matriz_bases, probdata):
    """
    Avalia a qualidade de uma solução considerando a perturbação causada pela indisponibilidade de bases ocupadas.
    
    Parâmetros:
        solution (np.ndarray): Matriz 14x125 indicando a alocação das equipes (0, 1, 2 ou 3).
        fobj_1 (func): Função que calcula o valor de f1 para uma configuração de solução.
        fobj_2 (func): Função que calcula o valor de f2 para uma configuração de solução.
        csv_path (str): Caminho para o arquivo CSV com as latitudes e longitudes das bases e ativos.
        
    Retorna:
        float: Índice de qualidade calculado a partir da norma euclidiana das perturbações normalizadas.
    """
    # Constantes de normalização
    f1_min, f1_max = 1010.5, 6279.2
    f2_min, f2_max = 412.6, 2119.8

    # Identificar as bases ocupadas
    bases_ocupadas = set(np.where(x.solution.sum(axis=1) > 0)[0])

    # Valores iniciais de f1 e f2
    f1_original = fobj_1(x, probdata).fitness
    f2_original = fobj_2(x, probdata).fitness

    perturbacoes_f1 = []
    perturbacoes_f2 = []

    # Iterar sobre cada base ocupada
    for base_indisponivel in bases_ocupadas:
        # Encontrar a base mais próxima
        distancias = [
            calcular_distancia_bases(
                matriz_bases[base_indisponivel],
                matriz_bases[base_proxima]
            )
            if base_proxima != base_indisponivel else float('inf')
            for base_proxima in range(len(matriz_bases))
        ]
        base_mais_proxima = np.argmin(distancias)

        # Realocar as equipes e ativos para a base mais próxima
        solution_modificada = x
        solution_modificada.solution[base_mais_proxima] += solution_modificada.solution[base_indisponivel]
        solution_modificada.solution[base_indisponivel] = 0

        # Calcular os novos valores de f1 e f2
        f1_modificado = fobj_1(solution_modificada, probdata).fitness
        f2_modificado = fobj_2(solution_modificada, probdata).fitness

        # Calcular as perturbações normalizadas
        perturbacao_f1 = abs(f1_modificado - f1_original) / (f1_max - f1_min)
        perturbacao_f2 = abs(f2_modificado - f2_original) / (f2_max - f2_min)

        perturbacoes_f1.append(perturbacao_f1)
        perturbacoes_f2.append(perturbacao_f2)

    # Calcular as médias das perturbações normalizadas
    media_perturbacao_f1 = np.mean(perturbacoes_f1)
    media_perturbacao_f2 = np.mean(perturbacoes_f2)

    # Calcular a norma euclidiana
    robustez = np.sqrt(media_perturbacao_f1 ** 2 + media_perturbacao_f2 ** 2)
    x.robustez = robustez

    return x


def balanco_carga(x, probdata):
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
    sd = np.std(carga_eq)
    x.balanco_sd = sd
    return x


def ahp_classic(solucoes, matriz_comparacao_atributos):
    """
    Implementa o método AHP clássico para escolher entre soluções considerando quatro atributos.

    Parâmetros:
        solucoes (list): Lista de objetos, onde cada objeto tem os atributos `f1_value`, `f2_value`, `robustez`, `balanco_sd`.
        matriz_comparacao_atributos (numpy.ndarray): Matriz 4x4 de comparação par a par entre os atributos.

    Retorno:
        Object: Solução escolhida após a execução do AHP.
    """

    def normalizar_matriz(matriz):
        """Normaliza a matriz e retorna os pesos calculados."""
        col_sum = np.sum(matriz, axis=0)
        matriz_norm = matriz / col_sum
        pesos = np.mean(matriz_norm, axis=1)
        return pesos

    def calcular_consistencia(matriz):
        """
        Calcula a razão de consistência (CR) de uma matriz de comparação.
        
        Parâmetros:
            matriz (list of lists): Matriz de comparação par a par.
        
        Retorno:
            float: Razão de consistência (CR).
        """
        # Converte a matriz para um numpy array, se necessário
        matriz = np.array(matriz)
        n = matriz.shape[0]  # Obtém o tamanho da matriz

        pesos = normalizar_matriz(matriz)

        # Calcula λ_max
        w_temp = np.dot(matriz, pesos)
        lambda_max = np.mean(w_temp / pesos)

        # Índice de consistência (CI)
        ci = (lambda_max - n) / (n - 1)

        # Índices de consistência aleatórios (RI) para matrizes de tamanho 1 a 10
        ri_table = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_table.get(n, 1.49)  # Usa 1.49 como padrão para n > 10

        # Razão de consistência (CR)
        cr = ci / ri if ri != 0 else 0
        return cr

    # Cria as matrizes de comparação par a par para cada atributo
    n = len(solucoes)
    matrizes_atributos = {
        'f1_value': np.ones((n, n)),
        'f2_value': np.ones((n, n)),
        'robustez': np.ones((n, n)),
        'balanco_sd': np.ones((n, n))
    }

    # Dicionário com os valores mínimos e máximos de cada atributo
    limites = {
        'f1_value': {'min': 1010.5, 'max': 1185.0},
        'f2_value': {'min': 412.6, 'max': 427.5},
        'robustez': {'min': 0, 'max': 0.03},
        'balanco_sd': {'min': 0, 'max': 17.82},
    }

    # Preenche as matrizes de comparação par a par com base nas diferenças normalizadas
    for i in range(n):
        for j in range(i + 1, n):  # Calcula apenas metade superior
            for atributo in ['f1_value', 'f2_value', 'robustez', 'balanco_sd']:
                val_i = getattr(solucoes[i], atributo)
                val_j = getattr(solucoes[j], atributo)

                # Obtém os limites para o atributo atual
                min_atributo = limites[atributo]['min']
                max_atributo = limites[atributo]['max']

                # Normaliza a diferença com base nos limites do atributo
                diff_normalizado = abs(val_i - val_j) / (max_atributo - min_atributo)

                # Normaliza a escala para 1 a 9
                if diff_normalizado != 0:
                    comparacao = min(max(round(1 + 8 * diff_normalizado, 2), 1), 9)
                else:
                    comparacao = 1  # Se os valores forem iguais

                # Preenche a matriz de comparações
                if val_i < val_j:
                    matrizes_atributos[atributo][i, j] = min(comparacao, 9)
                else:
                    matrizes_atributos[atributo][i, j] = min(round(1 / comparacao, 2), 9)

                matrizes_atributos[atributo][j, i] = min(round(1 / matrizes_atributos[atributo][i, j], 2), 9)

    for atributo in ['f1_value', 'f2_value', 'robustez', 'balanco_sd']:
        cons = calcular_consistencia(matrizes_atributos[atributo])
        print(f"(AHP) Consistencia dos pesos para {atributo}: {cons}")
        plotar_tabela_comparacao(matrizes_atributos[atributo], range(8))

    # Calcula os pesos das soluções em cada atributo
    pesos_solucoes_por_atributo = {}
    for atributo, matriz in matrizes_atributos.items():
        pesos_solucoes_por_atributo[atributo] = normalizar_matriz(matriz)

    # Calcula os pesos dos atributos (prioridades) usando a matriz de comparação 4x4 fornecida
    pesos_atributos = normalizar_matriz(matriz_comparacao_atributos)
    print(f"(AHP) Pesos dos atributos: {pesos_atributos}")

    # Calcula o peso final de cada solução
    pesos_finais_solucoes = np.zeros(n)
    for i in range(n):
        for j, atributo in enumerate(['f1_value', 'f2_value', 'robustez', 'balanco_sd']):
            pesos_finais_solucoes[i] += pesos_solucoes_por_atributo[atributo][i] * pesos_atributos[j]

    # Seleciona a solução com maior peso final
    indice_melhor_solucao = np.argmax(pesos_finais_solucoes)
    razao_de_consistencia = calcular_consistencia(matriz_comparacao_atributos)
    return (indice_melhor_solucao, solucoes[indice_melhor_solucao], razao_de_consistencia, pesos_atributos)


def electre_1(solucoes, pesos_atributos, c_threshold, d_threshold):
    """
    Implementação do metodo ELECTRE I para avaliar e filtrar soluções
    considerando quatro atributos:
    - f1_value
    - f2_value
    - robustez
    - balanco_sd

    Observação: neste código, assume-se que os quatro atributos (f1, f2, robustez e balanco)
    seguem a lógica "quanto menor, melhor"

    Parâmetros:
    ----------
    solucoes : list
        Lista de objetos (Struct) contendo os atributos a serem avaliados:
        - sol.f1_value
        - sol.f2_value
        - sol.robustez
        - sol.balanco_sd

    pesos_atributos : list ou np.ndarray
        Pesos atribuídos a cada critério [w_f1, w_f2, w_robustez, w_balanco].
        Devem somar 1 ou ter proporções coerentes.

    c_threshold : float
        Limiar de concordância (entre 0 e 1). Indica o nível mínimo de
        concordância necessário para que uma solução i sobreclassifique j.

    d_threshold : float
        Limiar de discordância (entre 0 e 1). Indica o nível máximo de
        discordância permitido para que i sobreclassifique j.

    Retorna:
    --------
    outranking_matrix : np.ndarray
        Matriz n x n, onde cada elemento outranking[i,j] é 1 caso a solução i
        sobreclassifique a solução j segundo o ELECTRE I, ou 0 caso contrário.

    survivors : list
        Lista de índices das soluções que não foram eliminadas pelo metodo.
        Cada índice corresponde à posição da solução na lista `solucoes`.
    """

    # 'sense' indica a forma de interpretar o critério:
    #  1  => "menor é melhor"
    # -1 => "maior é melhor"
    sense = [1, 1, 1, 1]

    # Determina a quantidade de soluções
    n = len(solucoes)

    # Cria a matriz de decisão (n x 4), cada linha representa uma solução,
    # e cada coluna representa um critério (f1, f2, robustez, balanco).
    # Multiplica o atributo pelo 'sense' para padronizar o critério.
    dec_mat = np.zeros((n, 4))
    for i, sol in enumerate(solucoes):
        dec_mat[i, 0] = sense[0] * sol.f1_value
        dec_mat[i, 1] = sense[1] * sol.f2_value
        dec_mat[i, 2] = sense[2] * sol.robustez
        dec_mat[i, 3] = sense[3] * sol.balanco_sd

    # Conversão de pesos em array NumPy para facilitar operações
    w = np.array(pesos_atributos)
    w_total = w.sum()

    # Criação das matrizes de concordância e discordância (n x n)
    # Cada posição (i, j) será preenchida com valores entre 0 e 1.
    concordance = np.zeros((n, n))
    discordance = np.zeros((n, n))

    # Cálculo do min, max e amplitude de cada critério para normalizar diferenças
    min_j = dec_mat.min(axis=0)
    max_j = dec_mat.max(axis=0)
    range_j = max_j - min_j

    # Preenche as matrizes de concordância e discordância
    for i in range(n):
        for j in range(n):
            # Se i == j, não há comparação de i com ele mesmo
            if i == j:
                continue

            # Identifica quais critérios favorecem i (c_indices) e
            # em quais i é pior do que j (d_indices)
            c_indices = []
            d_indices = []
            for c in range(4):
                # Se dec_mat[i,c] <= dec_mat[j,c], i é melhor ou igual no critério c
                if dec_mat[i, c] <= dec_mat[j, c]:
                    c_indices.append(c)
                else:
                    d_indices.append(c)

            # Calcula o grau de concordância C(i,j) normalizando pela soma dos pesos
            c_ij = sum(w[k] for k in c_indices) / w_total

            # Calcula a discórdia D(i,j):
            # Considera a maior diferença relativa (ou zero se i não é pior em nenhum critério)
            if len(d_indices) == 0:
                d_ij = 0.0
            else:
                diffs = []
                for c in d_indices:
                    diff = (dec_mat[i, c] - dec_mat[j, c]) / range_j[c]
                    diffs.append(diff)
                d_ij = max(diffs) if diffs else 0.0

            concordance[i, j] = c_ij
            discordance[i, j] = d_ij

    # Monta a matriz de sobreclassificação (outranking):
    # outranking[i,j] = 1 se i sobreclassifica j, 0 caso contrário
    outranking = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # A condição para sobreclassificar no ELECTRE I:
            #  - Concordância >= c_threshold
            #  - Discordância <= d_threshold
            if concordance[i, j] >= c_threshold and discordance[i, j] <= d_threshold:
                outranking[i, j] = 1

    # Determina as soluções que não foram eliminadas:
    # Uma solução i é eliminada se existe uma outra solução k que a sobreclassifica
    # (outranking[k, i] = 1) sem que i sobreclassifique k de volta.
    survivors = []
    for i in range(n):
        eliminated = False
        for k in range(n):
            if k != i:
                # Se outranking[k, i] == 1 e outranking[i, k] == 0,
                # significa que k vence i sem reciprocidade, eliminando i
                if outranking[k, i] == 1 and outranking[i, k] == 0:
                    eliminated = True
                    break
        if not eliminated:
            survivors.append(i)

            # --------------------------
            # PLOTANDO AS MATRIZES
            # --------------------------

            plot_matrix_values(dec_mat,
                               title="Matriz de Performance (dec_mat)",
                               x_label="Critérios",
                               y_label="Soluções",
                               fmt=".3f")
            plot_matrix_values(concordance,
                               title="Matriz de Concordância",
                               x_label="Solução j",
                               y_label="Solução i",
                               fmt=".3f")

            plot_matrix_values(discordance,
                               title="Matriz de Discordância",
                               x_label="Solução j",
                               y_label="Solução i",
                               fmt=".3f")

            plot_matrix_values(outranking,
                               title="Matriz de Sobreclassificação (0 ou 1)",
                               x_label="Solução j",
                               y_label="Solução i",
                               fmt=".0f")  # se for só 0 ou 1

    return outranking, survivors


probdata = probdef_new()

# vetor de alternativas a serem usadas no metodo de decisao
alternativas = ler_solucoes_do_csv("solucoes.csv")
coord_bases = construir_matriz_bases("probdata.csv")

for alternativa in alternativas:
    robustez_indisponibilidade(alternativa, coord_bases, probdata)
    balanco_carga(alternativa, probdata)

atributos_par_a_par = [
    [1, 5, 7, 2],  # f1_value comparado com os outros atributos
    [1 / 5, 1, 3, 1 / 3],  # f2_value comparado com os outros atributos
    [1 / 7, 1 / 3, 1, 1 / 5],  # robustez comparado com os outros atributos
    [1 / 2, 3, 5, 1]  # balanco_sd comparado com os outros atributos
]

'''
Prioridades:
1) f1_value
2) balanco_sd
3) f2_value
4) robustez
'''

decisao_ahp = ahp_classic(alternativas, atributos_par_a_par)
print(f"(AHP) Opcao selecionada: {decisao_ahp[0]}")
print(f"(AHP) Consistencia da matriz de comparacao par a par dos atributos: {decisao_ahp[2]}")
plot_melhor_solucao(probdata, decisao_ahp[1].solution)
best_ahp_idx = decisao_ahp[0]
best_ahp_sol = decisao_ahp[1]

# Exemplo: Pesos fixos (conforme indicado)
# f1_value=0.52047192, f2_value=0.1243774, robustez=0.05808436, balanco_sd=0.29706632
pesos_electre = [0.52047192, 0.1243774, 0.05808436, 0.29706632]

# Limiar de concordância e discordância
c_threshold = 0.75
d_threshold = 0.25

# Aplica ELECTRE I, plotando as matrizes e retornando outranking e survivors
outranking_mat, survivors = electre_1(
    solucoes=alternativas,
    pesos_atributos=pesos_electre,
    c_threshold=c_threshold,
    d_threshold=d_threshold
)

print("Matriz outranking:\n", outranking_mat)
print("Soluções survivors:", survivors)
for idx in survivors:
    print(f"Plotando survivor do ELECTRE no idx={idx}")
    sol = alternativas[idx]
    plot_melhor_solucao(probdata, sol.solution)

print("\n=== COMPARANDO AHP x ELECTRE I ===")
if best_ahp_idx in survivors:
    print("A melhor solução do AHP também está entre as sobreviventes do ELECTRE I.")
else:
    print("A melhor solução do AHP NÃO está entre as sobreviventes do ELECTRE I (divergência).")


plot_solutions_front(
    solucoes=alternativas,
    idx_ahp=best_ahp_idx,
    survivors=survivors,
    titulo="Fronteira de Soluções Avaliadas (AHP x ELECTRE)"
)


