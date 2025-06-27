import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_and_save_model(input_file, output_dir=r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data\models"):
    """
    Carga los datos preprocesados, entrena un modelo de regresión y lo guarda.

    Args:
        input_file (str): Ruta al archivo CSV de datos preprocesados.
        output_dir (str): Directorio donde se guardará el modelo entrenado.
    """
    try:
        data = pd.read_csv(input_file, index_col=0, parse_dates=True)

        # Usar las columnas de características y la columna "Close" como objetivo
        features = ["Open", "High", "Low", "Volume"]
        target = "Close"

        X = data[features]
        y = data[target]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo de Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE: {mse:.6f}")
        print(f"R2 Score: {r2:.6f}")

        # Guardar el modelo entrenado
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        model_filename = os.path.join(output_dir, f"{base_filename}_random_forest_model.joblib")
        joblib.dump(model, model_filename)
        print(f"Modelo guardado en {model_filename}")

    except Exception as e:
        print(f"Error al entrenar y guardar el modelo para {input_file}: {e}")

if __name__ == "__main__":
    preprocessed_data_dir = r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data\preprocessed"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_preprocessed_data_dir = os.path.join(script_dir, preprocessed_data_dir)

    for interval in ["1d", "1wk", "1mo"]:
        input_file = os.path.join(absolute_preprocessed_data_dir, f"BAP_{interval}.csv")
        if os.path.exists(input_file):
            train_and_save_model(input_file)
        else:
            print(f"El archivo {input_file} no existe.")
