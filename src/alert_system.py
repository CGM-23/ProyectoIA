import yfinance as yf
import pandas as pd
import joblib
import os
from datetime import datetime
from pathlib import Path

def generate_alerts(data_file, model_file, threshold=0.03):
    """
    Carga los datos y el modelo, realiza predicciones y genera alertas
    basadas en cambios significativos del precio de las acciones en tiempo real.

    Args:
        data_file (str): Ruta al archivo CSV de datos preprocesados.
        model_file (str): Ruta al archivo del modelo entrenado.
        threshold (float): Porcentaje de cambio para considerar una alerta (e.g., 0.03 para 3%).
    """
    try:
        # Cargar los datos preprocesados (últimos datos disponibles)
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)

        # Verificar los últimos registros de los datos para depuración
        print("\nÚltimos registros de los datos:")
        print(data.tail())  # Imprimir las últimas filas para ver cómo está el formato de los datos

        # Cargar el modelo entrenado
        model = joblib.load(model_file)

        # Realizar predicciones
        features = ["Open", "High", "Low", "Volume"]
        X = data[features]
        data["Predicted_Close"] = model.predict(X)

        # Calcular el cambio porcentual entre precios consecutivos
        data["Daily_Change"] = data["Close"].pct_change()
        data["Predicted_Change"] = data["Predicted_Close"].pct_change()

        # Obtener la fecha actual para compararla con los datos
        current_date = datetime.now().date()  # Solo la fecha, sin hora

        print(f"\n--- Alertas para {os.path.basename(data_file)} ---")
        last_alert = None  # Variable para evitar alertas duplicadas
        alert_found = False  # Variable para verificar si se ha encontrado una alerta

        for index, row in data.iterrows():
            alert_message = ""

            # Solo generar alertas para el día actual
            if index.date() == current_date:  # Comparar solo con la fecha actual
                if pd.notna(row["Daily_Change"]):  # Solo si hay un cambio calculado
                    # Verificar si el cambio en el precio real es significativo
                    if abs(row["Daily_Change"]) > threshold:
                        alert_found = True
                        if row["Daily_Change"] > 0:
                            alert_message = f'Alerta: {index.date()} - Subida significativa del {row["Daily_Change"]*100:.2f}% en el valor real.'
                        else:
                            alert_message = f'Alerta: {index.date()} - Bajada significativa del {abs(row["Daily_Change"])*100:.2f}% en el valor real.'
                
                # Alerta sobre el cambio en el precio predicho por el modelo
                if pd.notna(row["Predicted_Change"]):
                    if abs(row["Predicted_Change"]) > threshold:
                        alert_found = True
                        if row["Predicted_Change"] > 0:
                            alert_message = f'Alerta (Predicción): {index.date()} - Subida significativa del {row["Predicted_Change"]*100:.2f}% en el valor predicho.'
                        else:
                            alert_message = f'Alerta (Predicción): {index.date()} - Bajada significativa del {abs(row["Predicted_Change"])*100:.2f}% en el valor predicho.'

            # Solo imprimir alerta si no ha sido generada antes para el mismo día
            if alert_message and alert_message != last_alert:
                print(alert_message)
                last_alert = alert_message  # Actualizar la última alerta generada

        # Si no se encontró ninguna alerta, mostrar el mensaje correspondiente
        if not alert_found:
            last_real_price = data["Close"].iloc[-1]  # Último valor de la columna 'Close'
            print(f"\nNo hubo subida o bajada significativa hoy. El valor actual de la acción es: {last_real_price:.2f}")

    except FileNotFoundError as e:
        print(f"Archivo no encontrado: {e}")
    except Exception as e:
        print(f"Error al generar alertas para {data_file}: {e}")

if __name__ == '__main__':
    # Definir las rutas correctas para los archivos de datos preprocesados y el modelo entrenado
    data_file = Path("c:/Users/PC/Desktop/EJERCICIOS_PROGRA/python/ProyectoIA/data/preprocessed/descaled_BAP_1d.csv")  
    model_file = Path("c:/Users/PC/Desktop/EJERCICIOS_PROGRA/python/ProyectoIA/data/models/BAP_1d_linear_regression_model.joblib")  # Ruta al archivo del modelo entrenado (ajusta según el modelo que quieras usar)
    threshold = 0.03  # Umbral de cambio para generar alerta (3% en este caso)

    # Llamar a la función para generar las alertas
    generate_alerts(data_file, model_file, threshold)
