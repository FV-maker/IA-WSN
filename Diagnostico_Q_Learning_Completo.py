"""Graficar y analizar resultados del entrenamiento Q-Learning.

Este script lee los archivos generados en la fase de entrenamiento
(tabla Q y recompensas por episodio) y produce varias figuras para
evaluar la distribución de los valores de la tabla Q y la evolución
de la recompensa obtenida durante el aprendizaje.
"""

import pandas as pd
import matplotlib.pyplot as plt

# === Cargar tabla Q ===
# Cargamos la tabla Q generada durante el entrenamiento
q_table = pd.read_csv("resultados_q_learning_splstm/tabla_q.csv")

# Hacemos una copia para manipularla sin alterar el DataFrame original
df = q_table.copy()

# === Histograma de valores Q ===
# Permite observar la distribución general de la tabla Q
plt.figure(figsize=(6, 4))
plt.hist(q_table["Q"], bins=30, color="purple", alpha=0.7)
plt.title("Distribución de Valores Q")
plt.xlabel("Valor Q")
plt.ylabel("Frecuencia")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("grafico_histograma_q.png", dpi=300)
plt.show()

# === Preparación para el boxplot ===
# Se extrae la parte numérica del identificador de cada nodo de origen
df["origen_num"] = df["origen"].astype(str).str.extract("(\d+)")

# Se eliminan valores no numéricos y se convierten a enteros
df = df[pd.to_numeric(df["origen_num"], errors="coerce").notna()]
df["origen_num"] = df["origen_num"].astype(int)

# Agrupamos los nodos de origen en grupos de diez para una visualización más clara
df["grupo_origen"] = df["origen_num"] // 10 * 10

# === Boxplot por grupos de nodos ===
# Solo se grafica si existen valores numéricos válidos
if not df.empty:
    plt.figure(figsize=(10, 5))
    df.boxplot(column="Q", by="grupo_origen", grid=False)
    plt.title("Boxplot de Valores Q por Grupo de Nodos Origen")
    plt.suptitle("")
    plt.xlabel("Grupo de Nodos (por decenas)")
    plt.ylabel("Valor Q")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("grafico_boxplot_q_grupos.png", dpi=300)
    plt.show()
else:
    print(" No hay datos numéricos válidos en 'origen' para agrupar.")
# === Cargar recompensas por episodio ===
recompensas = pd.read_csv("resultados_q_learning_splstm/recompensas.csv")

# === Evolución de la recompensa ===
# Se comprueba que el CSV tenga las columnas requeridas
if "episodio" in recompensas.columns and "recompensa" in recompensas.columns:
    plt.figure(figsize=(8, 4))
    
    # Gráfica de la recompensa acumulada por episodio
    plt.plot(
        recompensas["episodio"],
        recompensas["recompensa"],
        marker="o", markersize=3, linewidth=1, label="Recompensa total"
    )
    
    # Línea horizontal con el promedio global de recompensa
    promedio = recompensas["recompensa"].mean()
    plt.axhline(
        y=promedio,
        color="red",
        linestyle="--",
        label=f"Promedio ({promedio:.2f})"
    )
    
    # Etiquetas y ajustes finales
    plt.title("Evolución de la Recompensa por Episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Total")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafico_recompensa_vs_episodio.png", dpi=300)
    plt.show()
else:
    print(" El archivo recompensas.csv no contiene columnas 'episodio' y 'recompensa'.")
