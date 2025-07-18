import pandas as pd
import numpy as np
import math
import os

# === Configuración ===
TIEMPO_SIMULACION = 1000                         # Número de instantes de simulación
RANGO_COMUNICACION = 55                          # Rango de comunicación entre nodos (metros)
NOISE_STD = 0.1                                  # Desviación estándar del ruido (simulación)
VENTANA_PROMEDIO = 5                             # Ventana para el promedio móvil
TOPO_PATH = "Data/topologia/topologia.csv"       # Ruta a la topología de nodos
OUTPUT_DIR = "Dataset"                   # Carpeta de salida para el dataset
os.makedirs(OUTPUT_DIR, exist_ok=True)           # Crea la carpeta si no existe

# === Cargar topología ===
df_nodos = pd.read_csv(TOPO_PATH)                # Carga la topología de nodos desde CSV

# === Crear enlaces válidos ===
enlaces = []
for i, nodo_i in df_nodos.iterrows():
    for j, nodo_j in df_nodos.iterrows():
        if nodo_i["nodo"] != nodo_j["nodo"]:    # Evita enlaces de un nodo consigo mismo
            distancia = math.dist((nodo_i["x"], nodo_i["y"]), (nodo_j["x"], nodo_j["y"]))
            if distancia <= RANGO_COMUNICACION: # Solo enlaces dentro del rango permitido
                enlaces.append({
                    "link_id": f"{nodo_i['nodo']}_{nodo_j['nodo']}",  # Identificador único del enlace
                    "origen": nodo_i["nodo"],                         # Nodo origen
                    "destino": nodo_j["nodo"],                        # Nodo destino
                    "distancia": distancia                            # Distancia entre nodos
                })

# === Simulación por tiempo y enlace ===
registros = []
for t in range(TIEMPO_SIMULACION):              # Para cada instante de tiempo
    for enlace in enlaces:                      # Para cada enlace válido
        base = np.exp(-0.05 * enlace["distancia"])      # Throughput base según distancia
        ruido = np.random.normal(0, NOISE_STD)          # Ruido gaussiano
        throughput = max(base + ruido, 0.0)             # Throughput instantáneo (no negativo)
        registros.append({
            "link_id": enlace["link_id"],
            "time": t,
            "origen": enlace["origen"],
            "destino": enlace["destino"],
            "distancia": enlace["distancia"],
            "throughput_instantaneo": throughput
        })

# === Crear DataFrame ===
df = pd.DataFrame(registros)                    # Convierte los registros en un DataFrame

# === Calcular promedio móvil y throughput medio por enlace ===
df["moving_avg_throughput"] = (
    df.groupby("link_id")["throughput_instantaneo"]
    .transform(lambda x: x.rolling(window=VENTANA_PROMEDIO, min_periods=1).mean())
)
df["Time_Average_Throughput"] = (
    df.groupby("link_id")["throughput_instantaneo"]
    .transform(lambda x: x.expanding().mean())
)

# === Exportar ===
df.to_csv(f"{OUTPUT_DIR}/Dataset.csv", index=False)     # Exporta el dataset enriquecido
print(f" Dataset dinámico enriquecido generado: {len(df)} muestras.")

# === Clasificación discreta del throughput ===
def clasificar(val):
    if val < 0.05:
        return "muy_bajo"
    elif val < 0.12:
        return "bajo"
    elif val < 0.20:
        return "medio"
    elif val < 0.32:
        return "alto"
    else:
        return "muy_alto"

df["clase"] = df["throughput_instantaneo"].apply(clasificar)  # Clasifica cada muestra

# === Balanceo por undersampling ===
min_count = df["clase"].value_counts().min()                  # Encuentra la clase menos representada
print(f"📊 Clases antes del balanceo:\n{df['clase'].value_counts()}")
df_balanceado = (
    df.groupby("clase", group_keys=False)
    .apply(lambda x: x.sample(min_count, random_state=42))    # Toma igual número de muestras por clase
    .reset_index(drop=True)
)
print(f"📊 Clases después del balanceo:\n{df_balanceado['clase'].value_counts()}")

# === Guardar dataset balanceado ===
df_balanceado.drop(columns=["clase"]).to_csv(f"{OUTPUT_DIR}/Dataset.csv", index=False)  # Exporta el dataset balanceado
print(f" Dataset balanceado exportado a: {OUTPUT_DIR}/Dataset.csv")
