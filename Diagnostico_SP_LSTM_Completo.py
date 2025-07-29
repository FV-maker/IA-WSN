"""Genera gráficos de diagnóstico para el modelo SP-LSTM.

Se cargan las predicciones realizadas por el modelo y se comparan con
los valores reales del dataset. También se leen las métricas de
evaluación almacenadas en un archivo de texto para graficarlas.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Cargar datos con predicciones y calcular error absoluto ===
df = pd.read_csv("modelo_splstm_dinamicoSPP/dataset_con_predicciones.csv")

# Error absoluto entre el throughput real e inferido por la red
df["error_abs"] = np.abs(df["throughput_instantaneo"] - df["throughput_predicho"])
# Cargar métricas desde archivo de texto
with open("modelo_splstm_dinamicoSPP/metricas.txt") as f:
    lineas = f.readlines()
mae = float(lineas[0].split(":")[1])  # Error absoluto medio
mse = float(lineas[1].split(":")[1])  # Error cuadrático medio
r2 = float(lineas[2].split(":")[1])   # Coeficiente de determinación

# === Histograma del throughput predicho ===
# Permite observar la distribución de las predicciones del modelo
plt.figure(figsize=(6, 4))
plt.hist(df["throughput_predicho"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribución del Throughput Predicho")
plt.xlabel("Throughput")
plt.ylabel("Frecuencia")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("grafico_histograma_throughput.png")
plt.show()

# === Dispersión real vs. predicho ===
# Cada punto representa una muestra coloreada según su error absoluto
plt.figure(figsize=(6, 4))
scatter = plt.scatter(
    df["throughput_instantaneo"],
    df["throughput_predicho"],
    c=df["error_abs"],
    cmap="viridis",
    alpha=0.5
)
plt.plot([0, 1], [0, 1], 'r--', label="Ideal (y = x)")
plt.colorbar(scatter, label="Error Absoluto")
plt.xlabel("Throughput Real")
plt.ylabel("Throughput Predicho")
plt.title("Dispersión: Real vs Predicho (Color por Error)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("grafico_dispersion_coloreado.png")
plt.show()

# === Gráfico de métricas generales ===
# Resumen de MAE, MSE y R² obtenidos durante la evaluación
plt.figure(figsize=(5, 4))
metricas = [mae, mse, r2]
nombres = ['MAE', 'MSE', 'R²']
plt.bar(nombres, metricas, color=['orange', 'green', 'blue'])
plt.title("Métricas del Modelo SP-LSTM")
plt.ylabel("Valor")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("grafico_metricas_lstm.png")
plt.show()
