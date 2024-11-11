import numpy as np
import pandas as pd

# Carrega o arquivo CSV em um DataFrame do pandas
df = pd.read_csv('probdata.csv', sep = ";")

# Converte o DataFrame em uma matriz numpy
matriz = df.to_numpy()

print(matriz)