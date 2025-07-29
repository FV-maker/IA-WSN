import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Definir rutas de carpetas
carpeta_ia = "resultados_simulacion_dinamica_ia"
carpeta_baseline = "resultados_baseline"

# === Cargar archivos IA
df_consumo_ia = pd.read_csv(f"{carpeta_ia}/consumo.csv")
df_latencia_ia = pd.read_csv(f"{carpeta_ia}/latencia.csv")
with open(f"{carpeta_ia}/pdr.txt", "r") as f:
    pdr_ia = float(f.read().strip())

# === Cargar archivos Baseline
df_consumo_base = pd.read_csv(f"{carpeta_baseline}/consumo.csv")
df_latencia_base = pd.read_csv(f"{carpeta_baseline}/latencia.csv")
with open(f"{carpeta_baseline}/pdr.txt", "r") as f:
    pdr_base = float(f.read().strip())

# === Extraer métricas
consumo_vals = [df_consumo_base["consumo_promedio"].mean(), df_consumo_ia["consumo_promedio"].mean()]
latencia_vals = [df_latencia_base["latencia_promedio"].mean(), df_latencia_ia["latencia_promedio"].mean()]
pdr_vals = [pdr_base * 100, pdr_ia * 100]

labels = ['Baseline', 'IA']

# === Función para crear cada figura
def crear_grafico(valores, ylabel, titulo, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, valores, color=["gray", "steelblue"])

    # Ajustes de eje y para dar espacio arriba
    max_val = max(valores)
    ax.set_ylim(0, max_val * 1.15)

    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.02 * max_val),  # offset relativo para que no se salga
            f"{height:.2f}",
            ha='center', va='bottom'
        )

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# === Crear gráficos individuales
crear_grafico(consumo_vals, "Joules", "Consumo Energético", "figura_consumo.png")
crear_grafico(latencia_vals, "Milisegundos", "Saltos promedio", "figura_latencia.png")
crear_grafico(pdr_vals, "Porcentaje (%)", "Packet Delivery Ratio (PDR)", "figura_pdr.png")

print(" Figuras generadas: figura_consumo.png, figura_latencia.png, figura_pdr.png")
