import pandas as pd
import matplotlib.pyplot as plt

# === Cargar tabla Q ===
q_table = pd.read_csv("resultados_q_learning_splstm/tabla_q.csv")
df = q_table.copy()

# === Histograma de valores Q ===
plt.figure(figsize=(6, 4))
plt.hist(q_table["Q"], bins=30, color="purple", alpha=0.7)
plt.title("Distribución de Valores Q")
plt.xlabel("Valor Q")
plt.ylabel("Frecuencia")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("grafico_histograma_q.png", dpi=300)
plt.show()

# Extraer solo el número desde la columna 'origen'
df["origen_num"] = df["origen"].astype(str).str.extract("(\d+)")

# Eliminar nulos y convertir a entero
df = df[pd.to_numeric(df["origen_num"], errors="coerce").notna()]
df["origen_num"] = df["origen_num"].astype(int)

# Agrupar por decenas
df["grupo_origen"] = df["origen_num"] // 10 * 10

# Graficar si hay datos válidos
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
    print("⚠️ No hay datos numéricos válidos en 'origen' para agrupar.")
# === Cargar recompensas por episodio ===
recompensas = pd.read_csv("resultados_q_learning_splstm/recompensas.csv")

# Validar columnas esperadas
if "episodio" in recompensas.columns and "recompensa" in recompensas.columns:
    plt.figure(figsize=(8, 4))
    
    # Graficar la recompensa por episodio
    plt.plot(recompensas["episodio"], recompensas["recompensa"],
             marker='o', markersize=3, linewidth=1, label='Recompensa total')
    
    # Calcular y graficar la línea de promedio
    promedio = recompensas["recompensa"].mean()
    plt.axhline(y=promedio, color='red', linestyle='--', label=f'Promedio ({promedio:.2f})')
    
    plt.title("Evolución de la Recompensa por Episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Total")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafico_recompensa_vs_episodio.png", dpi=300)
    plt.show()
else:
    print("⚠️ El archivo recompensas.csv no contiene columnas 'episodio' y 'recompensa'.")
