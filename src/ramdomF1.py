import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os
import json
from joblib import dump

def train_and_save_model(input_file, results_dir):
    """
    Carga los datos preprocesados, entrena un modelo Random Forest con parámetros fijos y guarda los resultados en un CSV.
    """
    try:
        data = pd.read_csv(input_file, index_col=0, parse_dates=True)

        # Selección automática de features
        feature_cols = ["Open", "High", "Low", "Volume"]
        lag_cols = [col for col in data.columns if col.startswith("Close_lag_")]
        sma_cols = [col for col in data.columns if col.startswith("SMA_")]
        features = feature_cols + lag_cols + sma_cols

        target = "Close"
        X = data[features]
        y = data[target]

        # Separar los datos (sin mezclar temporalidad)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Parámetros fijos
        fixed_params = {
            'n_estimators': 50,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }

        rf = RandomForestRegressor(**fixed_params)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print("Parámetros usados:", fixed_params)
        print(f"R2 en el conjunto de prueba: {r2:.4f}")
        print(f"MSE en el conjunto de prueba: {mse:.6f}")

        # Guardar resultados en CSV
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(
            results_dir, f"results_{os.path.basename(input_file).replace('.csv', '')}.csv"
        )
        result_dict = {
            'input_file': input_file,
            'params': json.dumps(fixed_params),
            'R2': r2,
            'MSE': mse
        }
        results_df = pd.DataFrame([result_dict])
        results_df.to_csv(result_file, index=False)
        print(f"Resultados guardados en {result_file}")

        # Guardar el modelo entrenado
        model_file = os.path.join(results_dir, f"model_{os.path.basename(input_file).replace('.csv','')}.joblib")
        dump(rf, model_file)
        print(f"Modelo guardado en {model_file}")

    except Exception as e:
        print(f"Error al entrenar y guardar el modelo para {input_file}: {e}")

if __name__ == "__main__":
    preprocessed_data_dir = r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data\preprocesed_normalizada"
    # Cambia el nombre de la carpeta aquí:
    results_dir = r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\results_rf_fijos"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_preprocessed_data_dir = os.path.join(script_dir, preprocessed_data_dir)
    absolute_results_dir = os.path.join(script_dir, results_dir)

    for interval in ["1d", "1wk", "1mo"]:
        input_file = os.path.join(absolute_preprocessed_data_dir, f"BAP_{interval}.csv")
        if os.path.exists(input_file):
            train_and_save_model(input_file, absolute_results_dir)
        else:
            print(f"El archivo {input_file} no existe.")
