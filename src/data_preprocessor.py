import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(input_file, output_dir=r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data\preprocessed"):
    """
    Carga los datos, los limpia, normaliza y guarda los datos preprocesados.

    Args:
        input_file (str): Ruta al archivo CSV de entrada.
        output_dir (str): Directorio donde se guardará el archivo CSV preprocesado.
    """
    try:
        # Cargar datos, usando la primera fila como encabezado (header=0) y saltando las filas 1 y 2 (Ticker y Date)
        data = pd.read_csv(input_file, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)

        # Manejar valores faltantes (usando forward fill)
        data.ffill(inplace=True)

        # Normalizar los datos (excepto el volumen, que tiene una escala diferente)
        scaler = MinMaxScaler()
        data_scaled = data.copy()
        features_to_scale = ["Close", "High", "Low", "Open", "Volume"]

        # Guardamos el scaler para poder invertir la normalización más tarde
        scaler.fit(data[features_to_scale])

        # Normalizamos los datos
        data_scaled[features_to_scale] = scaler.transform(data[features_to_scale])

        # Guardar los datos preprocesados
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        data_scaled.to_csv(output_file)
        print(f"Datos preprocesados y guardados en {output_file}")

        # Para obtener los valores desnormalizados, podemos aplicar inverse_transform
        data_descaled = data_scaled.copy()
        data_descaled[features_to_scale] = scaler.inverse_transform(data_scaled[features_to_scale])

        # Guardamos los datos desescalados (si es necesario)
        descaled_output_file = os.path.join(output_dir, f"descaled_{os.path.basename(input_file)}")
        data_descaled.to_csv(descaled_output_file)
        print(f"Datos desescalados guardados en {descaled_output_file}")

    except Exception as e:
        print(f"Error al preprocesar {input_file}: {e}")

if __name__ == "__main__":
    # Ajustar el directorio de datos a la ubicación del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data")
    preprocessed_dir = os.path.join(data_dir, "preprocessed")

    for interval in ["1d", "1wk", "1mo"]:
        input_file = os.path.join(data_dir, f"BAP_{interval}.csv")
        if os.path.exists(input_file):
            preprocess_data(input_file, output_dir=preprocessed_dir)
        else:
            print(f"El archivo {input_file} no existe.")
