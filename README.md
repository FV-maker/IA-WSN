# IA-WSN

## Tecnologías Utilizadas

- Python 3.10+
- TensorFlow 2.14
- NumPy, Pandas, Matplotlib, Scikit-Learn
- Algoritmos: SP-LSTM personalizado, Q-Learning

## Flujo de Ejecución

1. **Generar topología**  
   Ejecutar `generar_topologia.py` para construir una red de 150 nodos en una zona de 300x300 metros.

2. **Simular dataset**  
   Ejecutar `generar_dataset.py` para simular enlaces con tráfico mixto y ruido gaussiano.

3. **Entrenar SP-LSTM**  
   Ejecutar `entrenar_SP_lstm.py` para predecir el throughput de enlaces a partir del dataset generado.

4. **Entrenar Q-Learning**  
   Ejecutar `q_sp.py` para crear la tabla Q usando las predicciones del modelo SP-LSTM.

5. **Simular enrutamiento inteligente**  
   Ejecutar `simular_routing_fail.py` para evaluar el desempeño del sistema IA.

6. **Simular baseline**  
   Ejecutar `baseline_fail.py` como comparación sin inteligencia artificial.

7. **Diagnóstico y análisis**  
   Ejecutar `Diagnostico_SP_LSTM_Completo.py` y `Diagnostico_Q_Learning_Completo.py` para generar métricas y gráficos.

## Resultados Principales

- MAE del modelo SP-LSTM: 0.0646  
- R² del modelo SP-LSTM: 0.7904  
- PDR Baseline: 73.33 %  
- PDR con SP-LSTM + Q-Learning: 83.33 %  
- Incremento moderado en saltos y consumo energético a cambio de mayor confiabilidad

## Alcances y Limitaciones

Este proyecto fue desarrollado íntegramente en entorno simulado. No contempla:

- Implementación física en hardware o sensores reales.
- Conexión con infraestructuras industriales o sistemas productivos.
- Análisis de costos de implementación real.
- Validación con tráfico real o topologías dinámicas.

## Líneas Futuras

- Incorporar variabilidad topológica y patrones de tráfico no estacionarios.
- Aplicar técnicas de curriculum learning y selección activa de muestras.
- Ajustar la política de recompensa para optimizar el equilibrio entre consumo y confiabilidad.
- Evaluar generalización con entornos reales o híbridos.

## Autor
Fabián Venegas Mesas

Este proyecto corresponde a un trabajo de titulación en Ingeniería Civil Eléctrica con mención en Telecomunicaciones. Uso estrictamente académico.


