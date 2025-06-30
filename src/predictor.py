import pandas as pd
from joblib import load
import os

# 1. Carga el modelo entrenado
model_path = r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\results\best_model_BAP_1d.joblib"
model = load(model_path)

# 2. Carga tus datos preprocesados NO normalizados (escala real)
data_path = r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data\preprocesed_normalizada\BAP_1d.csv"
data = pd.read_csv(data_path, index_col=0, parse_dates=True)

# 3. Selecciona la última fila para predecir el siguiente día
X_cols = ["Open", "High", "Low", "Volume"] + \
    [col for col in data.columns if col.startswith("Close_lag_")] + \
    [col for col in data.columns if col.startswith("SMA_")]

last_row = data[X_cols].iloc[[-1]]

# 4. Predice el precio del siguiente día
pred_next_day = model.predict(last_row)[0]

# 5. Obtén el precio actual real (último valor de 'Close')
last_close = data["Close"].iloc[-1]
last_fecha = data.index[-1]

# 6. Ajuste empírico: suma una fracción del cambio último real (puedes ajustar el "factor" para calibrar)
# Ejemplo: usar la diferencia real entre el último y el penúltimo cierre
last_close_prev = data["Close"].iloc[-2]
ajuste = 0.5 * (last_close - last_close_prev)   # Usa 0.5 como ejemplo, puedes probar 1, 0.3, etc.
pred_next_day_ajustada = pred_next_day + ajuste

print(f"Fecha actual: {last_fecha}")
print(f"Precio de cierre actual: {last_close:.4f}")
print(f"Predicción base para el siguiente día: {pred_next_day:.4f}")
print(f"Predicción AJUSTADA para el siguiente día: {pred_next_day_ajustada:.4f}")
