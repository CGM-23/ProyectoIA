import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(input_file, scaled_dir, descaled_dir, n=3):
    """
    Preprocesa los datos, guarda normalizados en scaled_dir y desescalados en descaled_dir.
    """
    try:
        data = pd.read_csv(input_file, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
        data.ffill(inplace=True)

        # === Ventanas deslizantes (lags) ===
        lag_cols = []
        for i in range(1, n+1):
            col = f'Close_lag_{i}'
            data[col] = data['Close'].shift(i)
            lag_cols.append(col)

        # === SMA ===
        data[f'SMA_{n}'] = data['Close'].rolling(window=n).mean()

        # Eliminar filas con nulos solo en las columnas lag
        data = data.dropna(subset=lag_cols).copy()

        # === Normalización ===
        scaler = MinMaxScaler()
        features_to_scale = ["Close", "High", "Low", "Open", "Volume"]

        scaler.fit(data[features_to_scale])
        data_scaled = data.copy()
        data_scaled[features_to_scale] = scaler.transform(data[features_to_scale])

        # === Guarda los datos normalizados (0 a 1) ===
        os.makedirs(scaled_dir, exist_ok=True)
        scaled_file = os.path.join(scaled_dir, os.path.basename(input_file))
        data_scaled.to_csv(scaled_file)
        print(f"Datos normalizados guardados en {scaled_file}")

        # === Guarda los datos desescalados (valores originales) ===
        os.makedirs(descaled_dir, exist_ok=True)
        data_descaled = data_scaled.copy()
        data_descaled[features_to_scale] = scaler.inverse_transform(data_scaled[features_to_scale])
        descaled_file = os.path.join(descaled_dir, f"descaled_{os.path.basename(input_file)}")
        data_descaled.to_csv(descaled_file)
        print(f"Datos desescalados guardados en {descaled_file}")

    except Exception as e:
        print(f"Error al preprocesar {input_file}: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data")
    scaled_dir = os.path.join(data_dir, "preprocesed_normalizada")
    descaled_dir = os.path.join(data_dir, "preprocessed")
    n = 4

    for interval in ["1d", "1wk", "1mo"]:
        input_file = os.path.join(data_dir, f"BAP_{interval}.csv")
        if os.path.exists(input_file):
            preprocess_data(input_file, scaled_dir=scaled_dir, descaled_dir=descaled_dir, n=n)
        else:
            print(f"El archivo {input_file} no existe.")
