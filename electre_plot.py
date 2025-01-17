import numpy as np
import matplotlib.pyplot as plt


def plot_matrix_values(matrix, title="", x_label="j", y_label="i", fmt=".3f"):
    """
    Plota a 'matrix' exibindo explicitamente o valor numérico de cada célula.

    Parâmetros:
    -----------
    matrix : np.ndarray
        Matriz bidimensional a ser plotada.
    title : str
        Título do gráfico.
    x_label : str
        Rótulo (label) do eixo x.
    y_label : str
        Rótulo (label) do eixo y.
    fmt : str
        Formatação para exibir os números (ex: ".2f", ".3f", etc.).
    """
    # Dimensões da matriz
    nrows, ncols = matrix.shape

    # Cria figura e eixo
    fig, ax = plt.subplots(figsize=(max(6, ncols), max(4, nrows)))  # ajusta tamanho automaticamente

    # Define título e rótulos
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Configura limites dos eixos para caber todas as células
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)

    # Cria grade (linhas) para dividir células
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1, alpha=0.2)

    # Configura ticks principais (para exibir rótulos nos eixos)
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(np.arange(ncols))
    ax.set_yticklabels(np.arange(nrows))

    # Para cada célula (i, j), escreve o valor numérico
    for i in range(nrows):
        for j in range(ncols):
            val = matrix[i, j]
            ax.text(
                j, i,
                f"{val:{fmt}}",  # exibe com formatação (ex: .3f)
                ha="center", va="center",
                color="black",
                fontsize=10
            )

    # Remove traços dos eixos (opcional, fica mais "tabela")
    ax.spines[:].set_visible(False)

    plt.show()