"""Entrena el modelo SP-LSTM para estimar el throughput entre nodos."""

import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from splstm_2 import SP_LSTM


# === Rutas y configuración ===
# Ubicación del dataset de entrenamiento y carpeta de salida
DATA_PATH = "Dataset/Dataset.csv"
MODEL_DIR = "modelo_SP_lstm"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Cargar y preparar los datos ===
df = pd.read_csv(DATA_PATH)
# Codificamos el identificador del enlace a un valor numérico
df["link_id_code"] = df["link_id"].astype("category").cat.codes

# === Variables de entrada
# Datos de entrada secuenciales y auxiliares
X_seq = df[["moving_avg_throughput"]].values.reshape(-1, 1, 1)
X_aux = df[["link_id_code", "time", "moving_avg_throughput", "Time_Average_Throughput"]].values
y = df["throughput_instantaneo"].values

# === Escalado
scaler = MinMaxScaler()
X_aux_scaled = scaler.fit_transform(X_aux)
# Guardamos el scaler para usarlo durante la inferencia
joblib.dump(scaler, f"{MODEL_DIR}/scaler_X.save")

# === División de datos
X_seq_train, X_seq_test, X_aux_train, X_aux_test, y_train, y_test = train_test_split(
    X_seq, X_aux_scaled, y, test_size=0.2, random_state=None, shuffle=True
)

# === Construcción del modelo
input_seq = tf.keras.Input(shape=(1, 1))
input_aux = tf.keras.Input(shape=(X_aux.shape[1],))
# Capa recurrente personalizada que procesa la secuencia
x = SP_LSTM(units=32, input_dim=1, return_sequences=False)(input_seq)
# Concatenamos las características auxiliares
x = tf.keras.layers.concatenate([x, input_aux])
# Capas densas para producir la predicción final
x = tf.keras.layers.Dense(16, activation='relu')(x)
salida = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=[input_seq, input_aux], outputs=salida)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === Entrenamiento
history = model.fit(
    [X_seq_train, X_aux_train],
    y_train,
    validation_data=([X_seq_test, X_aux_test], y_test),
    epochs=10,
    batch_size=64,
    verbose=1,
    shuffle=False  # clave para resultados deterministas
)

# === Guardar modelo
model.save(f"{MODEL_DIR}/modelo_entrenado.keras")
print("SP-LSTM entrenado correctamente.")

# === Curva de entrenamiento
plt.plot(history.history['loss'], label='MSE (Train)')
plt.plot(history.history['val_loss'], label='MSE (Val)')
plt.title("Curva de Aprendizaje SP-LSTM")
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/grafico_mse.png")
plt.close()

# === Evaluación final ===
y_pred = model.predict([X_seq_test, X_aux_test])
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with open(f"{MODEL_DIR}/metricas.txt", "w") as f:
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"R2 Score: {r2:.6f}\n")

print("Evaluación completada. Métricas guardadas en 'metricas.txt'.")
