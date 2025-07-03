import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
import networkx as nx
import random
import logging
import time
from splstm_2 import SP_LSTM  # Asegúrate de que este archivo esté correctamente adaptado

# === Configuración general ===
TOPO_PATH = "Data/topologia/topologia_nodos.csv"           # Ruta a la topología de nodos
MODEL_PATH = "modelo_SP_lstm/modelo_entrenado.keras"   # Ruta al modelo SP-LSTM entrenado
SCALER_PATH = "modelo_SP_lstm/scaler_x.save"           # Ruta al scaler de variables auxiliares
QLEARN_DIR = "resultados_q_learning_splstm"                       # Carpeta de salida para resultados Q-Learning
os.makedirs(QLEARN_DIR, exist_ok=True)                            # Crea la carpeta si no existe

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# === Cargar topología ===
df_topo = pd.read_csv(TOPO_PATH)                                  # Carga la topología de nodos
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
link_ids = list(set([f"{a}_{b}" for a in df_topo["nodo"] for b in df_topo["nodo"] if a != b]))  # Todos los enlaces posibles
link2code = {lid: i for i, lid in enumerate(link_ids)}                   # Diccionario enlace a código numérico

# === Cargar modelo SP-LSTM y scaler ===
logging.info("Cargando modelo SP-LSTM...")
modelo = tf.keras.models.load_model(MODEL_PATH, custom_objects={"SP_LSTM": SP_LSTM})  # Carga el modelo
scaler = joblib.load(SCALER_PATH)                                                     # Carga el scaler

# === Inicialización Q-Learning ===
Q = {}                  # Tabla Q vacía
rewards = []            # Lista para guardar recompensas por episodio
cache_pred = {}         # Caché para predicciones de throughput

# Función para discretizar el throughput predicho
def discretizar(val):
    if val < 0.05: return "muy_bajo"
    elif val < 0.12: return "bajo"
    elif val < 0.20: return "medio"
    elif val < 0.32: return "alto"
    else: return "muy_alto"

# Función para predecir el throughput de un enlace usando el modelo SP-LSTM
def predecir_throughput(a, b, t):
    key = (a, b, t)
    if key in cache_pred:
        return cache_pred[key]

    lid = f"{a}_{b}"
    code = link2code.get(lid, 0)

    input_seq = np.array([[[0.5]]])  # Valor dummy para la secuencia
    x_df = pd.DataFrame([[code, t, 0.5, 0.5]],
        columns=["link_id_code", "time", "moving_avg_throughput", "Time_Average_Throughput"])
    x_scaled = scaler.transform(x_df.values)  # Normaliza las variables auxiliares

    y_pred = modelo.predict([input_seq, x_scaled], verbose=0)[0][0]
    val = max(y_pred, 0.01)  # Evita throughput 0
    cache_pred[key] = val
    return val

# === Hiperparámetros ===
EPISODIOS = 300
MAX_SALTOS = 12
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.95
EPSILON_MIN = 0.1
P_FAIL = 0.05

logging.info("Iniciando entrenamiento Q-Learning con SP-LSTM...")
start_global = time.time()

# === Entrenamiento Q-Learning ===
for ep in range(EPISODIOS):
    start_ep = time.time()
    total_reward = 0

    for origen in nodos:
        actual = origen
        t = 0

        for salto in range(MAX_SALTOS):
            vecinos = list(G.neighbors(actual))
            if not vecinos:
                break

            if random.random() < P_FAIL:
                break  # Simula un fallo de enlace

            # Selección de acción (vecino) usando política epsilon-greedy
            if random.random() < EPSILON:
                accion = min(vecinos, key=lambda v: nx.shortest_path_length(G, v, sink))
            else:
                candidatos = [(v, Q.get((actual, v), 0)) for v in vecinos]
                accion = max(candidatos, key=lambda x: x[1])[0]

            # Cálculo de recompensa
            reward = 10 - 0.5 * salto if accion == sink else -1
            futuro = G.neighbors(accion)
            max_q = max([Q.get((accion, v), 0) for v in futuro], default=0)
            # Actualización de la tabla Q
            Q[(actual, accion)] = Q.get((actual, accion), 0) + ALPHA * (reward + GAMMA * max_q - Q.get((actual, accion), 0))

            actual = accion
            t += 1
            total_reward += reward

            if actual == sink:
                break

    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)  # Decaimiento de epsilon
    rewards.append(total_reward)

    dur_ep = time.time() - start_ep
    logging.info(f"[EP {ep+1:03}] Recompensa total = {total_reward:.1f} | Epsilon = {EPSILON:.3f} | Tiempo = {dur_ep:.1f}s")

dur_total = time.time() - start_global
logging.info(f"✅ Entrenamiento finalizado en {dur_total/60:.2f} minutos.")

# === Guardar resultados ===
pd.DataFrame({"episodio": list(range(EPISODIOS)), "recompensa_total": rewards}).to_csv(f"{QLEARN_DIR}/recompensas.csv", index=False)
pd.DataFrame([{"origen": k[0], "accion": k[1], "Q": v} for k, v in Q.items()]).to_csv(f"{QLEARN_DIR}/tabla_q.csv", index=False)
