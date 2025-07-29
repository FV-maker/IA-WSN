"""Genera la topología de la red de sensores para las simulaciones."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

# === Parámetros del escenario agrícola ===

NUM_SENSORES = 150                           # Número de sensores a desplegar
AREA_X, AREA_Y = 300, 300                    # Dimensiones de la granja (en metros)
RANGO_COMUNICACION = 55                      # Rango de comunicación de cada sensor (en metros)
SEED = 42                                    # Semilla para reproducibilidad
MIN_GRADO = 3                                # Grado mínimo de conectividad por nodo (excepto sink)
OUTPUT_DIR = "Data/topologia"                # Carpeta de salida para archivos generados
os.makedirs(OUTPUT_DIR, exist_ok=True)       # Crea la carpeta si no existe

np.random.seed(SEED)                         # Fija la semilla para la generación aleatoria

# === Generar sensores ===
sensores = []
for i in range(NUM_SENSORES):
    sensores.append({
        "nodo": f"S{i}",                     # Nombre del nodo sensor
        "x": np.random.uniform(0, AREA_X),   # Posición X aleatoria dentro del área
        "y": np.random.uniform(0, AREA_Y),   # Posición Y aleatoria dentro del área
        "tipo": "sensor"                     # Tipo de nodo
    })

# === Sink en el centro de la granja ===
sink = {
    "nodo": "sink",                          # Nombre del nodo sink
    "x": AREA_X / 2,                         # Posición X centrada
    "y": AREA_Y / 2,                         # Posición Y centrada
    "tipo": "sink"                           # Tipo de nodo
}

nodos = sensores + [sink]                    # Lista completa de nodos (sensores + sink)
df_topo = pd.DataFrame(nodos)                # DataFrame con la topología inicial

# === Validar conectividad ===
def verificar_topologia(df, rango, min_grado):
    G = nx.Graph()
    # Agrega nodos con sus posiciones
    for _, row in df.iterrows():
        G.add_node(row["nodo"], pos=(row["x"], row["y"]))
    # Agrega aristas si la distancia es menor o igual al rango de comunicación
    for i, ni in df.iterrows():
        for j, nj in df.iterrows():
            if ni["nodo"] != nj["nodo"]:
                d = np.hypot(ni["x"] - nj["x"], ni["y"] - nj["y"])
                if d <= rango:
                    G.add_edge(ni["nodo"], nj["nodo"])
    grados = dict(G.degree())
    # Retorna el grafo, si es conexo y cumple el grado mínimo, y los grados de los nodos
    return G, nx.is_connected(G) and all(g >= min_grado for n, g in grados.items() if n != "sink"), grados

# === Generar hasta lograr topología válida ===
intentos = 0
while True:
    G, ok, grados = verificar_topologia(df_topo, RANGO_COMUNICACION, MIN_GRADO)
    if ok:
        break
    intentos += 1
    print(f"[Reintento #{intentos}]")
    # Reubica los sensores aleatoriamente si la topología no es válida
    for i in range(NUM_SENSORES):
        df_topo.loc[i, "x"] = np.random.uniform(0, AREA_X)
        df_topo.loc[i, "y"] = np.random.uniform(0, AREA_Y)

# === Guardar CSV ===
df_topo.to_csv(f"{OUTPUT_DIR}/topologia_nodos.csv", index=False)    # Guarda la topología en un archivo CSV
print(f" Topología agrícola generada correctamente en {intentos} intentos.")

# === Visualización ===
plt.figure(figsize=(10, 10))
for _, row in df_topo.iterrows():
    if row["tipo"] == "sensor":
        plt.plot(row["x"], row["y"], "b.", markersize=6)            # Dibuja sensores en azul
    else:
        plt.plot(row["x"], row["y"], "ro", markersize=10, label="Sink")  # Dibuja el sink en rojo

nx.draw_networkx_edges(G, pos=nx.get_node_attributes(G, 'pos'), alpha=0.2)  # Dibuja las conexiones
plt.title("Topología WSN Granja Inteligente – 150 sensores")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mapa_topologia.png", dpi=300)     # Guarda la imagen de la topología
plt.show()                                                          # Muestra la figura en pantalla
