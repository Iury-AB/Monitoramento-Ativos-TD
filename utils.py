def base_mais_proxima(dicionario, valor):
    return min(dicionario, key=lambda k: abs(dicionario[k] - valor))