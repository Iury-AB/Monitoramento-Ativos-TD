import matplotlib.pyplot as plt
import numpy as np

# Define os coeficientes (angular e linear) para 5 retas diferentes
retas = [
    {"angular": 0.5, "linear": 1},  # y = 0.5x + 1
    {"angular": -0.3, "linear": 2},  # y = -0.3x + 2
    {"angular": 1, "linear": -1},  # y = x - 1
    {"angular": -1, "linear": 0},  # y = -x
    {"angular": 0.2, "linear": -0.5},  # y = 0.2x - 0.5
]

# Gera valores de x
x = np.linspace(-10, 10, 100)

# Cria a figura
fig, ax = plt.subplots(figsize=(8, 6))

# Adiciona as retas no gráfico
for reta in retas:
    y = reta["angular"] * x + reta["linear"]
    ax.plot(x, y, label=f"y = {reta['angular']}x + {reta['linear']}")

# Configura os eixos
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Eixo X
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Eixo Y

# Configurações adicionais
ax.set_title("Gráfico de 5 Retas Sobrepostas")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)

# A variável fig contém a figura armazenada
# Para exibir, você ainda pode usar plt.show()
plt.show()
