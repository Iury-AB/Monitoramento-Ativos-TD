import matplotlib.pyplot as plt

def plotar_tabela_comparacao(matriz_comparacao, nomes_atributos):
    """
    Plota uma tabela representando a matriz de comparação par a par dos atributos.

    Parâmetros:
        matriz_comparacao (list of list): Matriz de comparação par a par.
        nomes_atributos (list): Lista com os nomes dos atributos.
    """
    # Criar a tabela
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')

    # Formatar os valores para exibição
    tabela_formatada = [[f"{v:.2f}" for v in linha] for linha in matriz_comparacao]

    # Adicionar a tabela
    tabela = ax.table(
        cellText=tabela_formatada,
        rowLabels=nomes_atributos,
        colLabels=nomes_atributos,
        cellLoc='center',
        loc='center'
    )

    # Ajustar o layout
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 1.2)  # Ajustar o tamanho da tabela

    plt.title("Matriz de Comparação Par a Par dos Atributos", fontsize=14)
    plt.show()

# Exemplo de uso
atributos_par_a_par = [
    [1, 5, 7, 2],       
    [1/5, 1, 3, 1/3],  
    [1/7, 1/3, 1, 1/5], 
    [1/2, 3, 5, 1]
]
atributos = ['f1_value', 'f2_value', 'robustez', 'balanco_sd']

plotar_tabela_comparacao(atributos_par_a_par, atributos)
