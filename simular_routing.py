"""Simula el enrutamiento inteligente usando SP‑LSTM y Q‑Learning."""

import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf
import joblib
import random
import os
import logging
from splstm_2 import SP_LSTM

# === Configuración general ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# === Configuración ===
TOPO_PATH = "Data/topologia/topologia_nodos.csv"                  # Ruta a la topología de nodos
QTABLE_PATH = "resultados_q_learning_splstm/tabla_q.csv"                 # Ruta a la tabla Q aprendida
MODEL_PATH = "modelo_SP_lstm/modelo_entrenado.keras"          # Ruta al modelo SP-LSTM entrenado
SCALER_PATH = "modelo_SP_lstm/scaler_x.save"                  # Ruta al scaler de variables auxiliares
OUTPUT_DIR = "resultados_simulacion_ia"                       # Carpeta de salida para resultados
os.makedirs(OUTPUT_DIR, exist_ok=True)                                   # Crea la carpeta si no existe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === Cargar topología ===
df_topo = pd.read_csv(TOPO_PATH)                                         # Carga la topología de nodos
G = nx.Graph()
for _, row in df_topo.iterrows():
    G.add_node(row["nodo"], tipo=row["tipo"], pos=(row["x"], row["y"]))  # Agrega nodos al grafo
for i, ni in df_topo.iterrows():
    for j, nj in df_topo.iterrows():
        if ni["nodo"] != nj["nodo"]:
            d = np.linalg.norm([ni["x"] - nj["x"], ni["y"] - nj["y"]])   # Calcula distancia entre nodos
            if d <= 55:
                G.add_edge(ni["nodo"], nj["nodo"], weight=d)             # Agrega arista si están dentro del rango

sink = df_topo[df_topo["tipo"] == "sink"]["nodo"].iloc[0]                # Nodo sink
nodos = df_topo[df_topo["tipo"] == "sensor"]["nodo"].tolist()            # Lista de sensores
link_ids = list(set([f"{a}_{b}" for a in df_topo["nodo"] for b in df_topo["nodo"] if a != b]))
link2code = {lid: i for i, lid in enumerate(link_ids)}                   # Diccionario enlace a código numérico

# === Cargar modelo y scaler ===
modelo = tf.keras.models.load_model(MODEL_PATH, custom_objects={"SP_LSTM": SP_LSTM})  # Carga el modelo SP-LSTM
scaler = joblib.load(SCALER_PATH)                                                     # Carga el scaler
df_q = pd.read_csv(QTABLE_PATH)                                                       # Carga la tabla Q
Q = {(row["origen"], row["accion"]): row["Q"] for _, row in df_q.iterrows()}          # Diccionario Q

# === Parámetros de simulación ===
ITERACIONES = 10
MAX_SALTOS = 12

resultados = []
cache_pred = {}

# Función para predecir el throughput de un enlace usando el modelo SP-LSTM
def predecir_throughput(a, b, t):
    """Obtiene una predicción de throughput para el enlace (a,b) en el tiempo t."""
    key = (a, b, t)
    if key in cache_pred:
        return cache_pred[key]

    lid = f"{a}_{b}"
    code = link2code.get(lid, 0)

    input_seq = np.array([[[0.5]]])  # dummy para moving_avg
    input_aux = pd.DataFrame([[code, t, 0.5, 0.5]], columns=[
        "link_id_code", "time", "moving_avg_throughput", "Time_Average_Throughput"
    ])
    input_aux_scaled = scaler.transform(input_aux.values)  # Normaliza las variables auxiliares

    y_pred = modelo.predict([input_seq, input_aux_scaled], verbose=0)[0][0]
    val = max(y_pred, 0.01)
    cache_pred[key] = val
    return val

# === Simulación ===
exitos, total_latencia, total_consumo = 0, 0, 0

for nodo in nodos:
    actual = nodo
    t = 0
    latencia = 0
    consumo = 0
    exitoso = False

    for salto in range(MAX_SALTOS):
        vecinos = list(G.neighbors(actual))
        if not vecinos:
            break

        candidatos = [(v, Q.get((actual, v), -np.inf)) for v in vecinos]
        candidatos = [c for c in candidatos if c[1] > -np.inf]
        if not candidatos:
            break

        accion = max(candidatos, key=lambda x: x[1])[0]

        throughput = predecir_throughput(actual, accion, t)
        p_fail_local = min(1.0, 0.2 * (1 - throughput))  # más throughput, menos fallos

        if random.random() < p_fail_local:
            break

        latencia += 1
        consumo += np.linalg.norm(np.array(G.nodes[actual]["pos"]) - np.array(G.nodes[accion]["pos"]))
        actual = accion
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

# === Métricas globales ===
pdr = exitos / len(nodos)                                        # Porcentaje de entregas exitosas
latencia_prom = total_latencia / len(nodos)                      # Latencia promedio
consumo_prom = total_consumo / len(nodos)                        # Consumo promedio

# === Guardar resultados ===
pd.DataFrame(resultados).to_csv(f"{OUTPUT_DIR}/simulacion.csv", index=False)           # Resultados por nodo
with open(f"{OUTPUT_DIR}/pdr.txt", "w") as f:
    f.write(str(pdr))                                                                  # Guarda el PDR
pd.DataFrame({"latencia_promedio": [latencia_prom]}).to_csv(f"{OUTPUT_DIR}/latencia.csv", index=False)
pd.DataFrame({"consumo_promedio": [consumo_prom]}).to_csv(f"{OUTPUT_DIR}/consumo.csv", index=False)

logging.info(f"Simulación completada | PDR = {pdr:.2f}, Latencia = {latencia_prom:.2f}, Consumo = {consumo_prom:.2f}")
