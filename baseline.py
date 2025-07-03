import pandas as pd
import numpy as np
import os
import networkx as nx
import random
import logging

# === Configuración general ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# === CONFIGURACIÓN ===
TOPO_PATH = "Data/topologia/topologia_nodos.csv"         # Ruta a la topología de nodos
OUTPUT_DIR = "resultados_baseline"                     # Carpeta de salida para resultados
os.makedirs(OUTPUT_DIR, exist_ok=True)                          # Crea la carpeta si no existe

RANGO_COMUNICACION = 55                                         # Rango de comunicación entre nodos (m)
MAX_SALTOS = 12                                                 # Máximo número de saltos permitidos
ITERACIONES = 10                                                # Número de iteraciones de simulación

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# === CARGA TOPOLOGÍA ===
df_topo = pd.read_csv(TOPO_PATH)                                # Carga la topología de nodos
G = nx.Graph()
for _, row in df_topo.iterrows():
    G.add_node(row["nodo"], tipo=row["tipo"], pos=(row["x"], row["y"]))  # Agrega nodos al grafo
for i, ni in df_topo.iterrows():
    for j, nj in df_topo.iterrows():
        if ni["nodo"] != nj["nodo"]:
            d = np.linalg.norm([ni["x"] - nj["x"], ni["y"] - nj["y"]])   # Calcula distancia entre nodos
            if d <= RANGO_COMUNICACION:
                G.add_edge(ni["nodo"], nj["nodo"], weight=d)             # Agrega arista si están dentro del rango

sink = df_topo[df_topo["tipo"] == "sink"]["nodo"].iloc[0]                # Nodo sink
nodos = df_topo[df_topo["tipo"] == "sensor"]["nodo"].tolist()            # Lista de sensores

# === FUNCIÓN THROUGHPUT BASELINE ===
def estimar_throughput(distancia):
    if distancia <= 15:
        return 0.95
    elif distancia <= 30:
        return 0.7
    elif distancia <= 45:
        return 0.5
    elif distancia <= 55:
        return 0.3
    else:
        return 0.05

# === SIMULACIÓN ===
resultados = []
total_latencia = 0
total_consumo = 0
exitos = 0

for nodo in nodos:
    actual = nodo
    latencia = 0
    consumo = 0
    exitoso = False
    t = 0

    for salto in range(MAX_SALTOS):
        vecinos = list(G.neighbors(actual))
        if not vecinos:
            break

        # Selecciona el vecino más cercano al sink (menor número de saltos)
        siguiente = min(
            vecinos,
            key=lambda x: nx.shortest_path_length(G, x, sink) if nx.has_path(G, x, sink) else float('inf')
        )

        # === Estimar throughput y fallo dinámico ===
        d = np.linalg.norm(np.array(G.nodes[actual]["pos"]) - np.array(G.nodes[siguiente]["pos"]))
        throughput_estimado = estimar_throughput(d)
        p_fail_local = min(1.0, 0.2 * (1 - throughput_estimado))  # Probabilidad de fallo según throughput

        if random.random() < p_fail_local:
            break  # Simula un fallo de transmisión

        consumo += d
        latencia += 1
        actual = siguiente
        t += 1

        if actual == sink:
            exitos += 1
            exitoso = True
            break

    resultados.append({
        "nodo_origen": nodo,
        "exito": exitoso,
        "saltos": latencia,
        "consumo_total": consumo
    })
    total_latencia += latencia
    total_consumo += consumo

# === MÉTRICAS ===
pdr = exitos / len(nodos)                                         # Porcentaje de entregas exitosas
latencia_prom = total_latencia / len(nodos)                       # Latencia promedio
consumo_prom = total_consumo / len(nodos)                         # Consumo promedio

# === EXPORTAR RESULTADOS ===
pd.DataFrame(resultados).to_csv(f"{OUTPUT_DIR}/simulacion.csv", index=False)           # Resultados por nodo
with open(f"{OUTPUT_DIR}/pdr.txt", "w") as f:
    f.write(str(pdr))                                                                  # Guarda el PDR
pd.DataFrame({"latencia_promedio": [latencia_prom]}).to_csv(f"{OUTPUT_DIR}/latencia.csv", index=False)
pd.DataFrame({"consumo_promedio": [consumo_prom]}).to_csv(f"{OUTPUT_DIR}/consumo.csv", index=False)

logging.info(f"✅ Simulación baseline dinámica completada | PDR = {pdr:.2f} | Latencia = {latencia_prom:.2f} | Consumo = {consumo_prom:.2f}")
