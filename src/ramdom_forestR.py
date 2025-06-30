import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os
import json
from joblib import dump

def find_best_parameters(input_file, results_dir):
    """
    Carga los datos preprocesados, busca los mejores parámetros y guarda los resultados en un CSV usando validación cruzada temporal.
    Imprime los R² de cada fold para el mejor modelo.
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

        # División temporal: 80% entrenamiento, 20% prueba
        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Definir validación cruzada temporal (no mezcla el pasado con el futuro)
        tscv = TimeSeriesSplit(n_splits=5)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt']  # Usa 'sqrt' para regresión
        }

        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=0, scoring='r2')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        # Extraer los R2 de cada split (fold) para el mejor modelo
        best_index = grid_search.best_index_
        split_test_scores = []
        # cv_results_['split0_test_score'], etc., son arrays con la score para cada split y cada combinación de params
        for i in range(tscv.get_n_splits()):
            split_test_scores.append(grid_search.cv_results_[f'split{i}_test_score'][best_index])
        
        split_test_scores = np.array(split_test_scores)
        promedio_r2 = split_test_scores.mean()
        std_r2 = split_test_scores.std()
        
        print("\nResultados de Validación Cruzada (TimeSeriesSplit):")
        for i, score in enumerate(split_test_scores):
            print(f"Fold {i+1} - R²: {score:.4f}")
        print(f"Promedio R²: {promedio_r2:.3f} ± {std_r2:.3f}\n")

        # Evaluar el modelo en el conjunto de prueba (20% más reciente)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print("Mejores parámetros encontrados:", best_params)
        print(f"R2 en el conjunto de prueba: {r2:.4f}")
        print(f"MSE en el conjunto de prueba: {mse:.6f}")

        # Guardar resultados en CSV
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(
            results_dir, f"results_{os.path.basename(input_file).replace('.csv', '')}.csv"
        )

        result_dict = {
            'input_file': input_file,
            'best_params': json.dumps(best_params),
            'R2_test': r2,
            'MSE_test': mse,
            'R2_folds': json.dumps(split_test_scores.tolist()),
            'R2_mean_cv': promedio_r2,
            'R2_std_cv': std_r2
        }
        results_df = pd.DataFrame([result_dict])
        results_df.to_csv(result_file, index=False)
        print(f"Resultados guardados en {result_file}")

        # Guardar el modelo entrenado
        model_file = os.path.join(results_dir, f"best_model_{os.path.basename(input_file).replace('.csv','')}.joblib")
        dump(best_model, model_file)
        print(f"Modelo guardado en {model_file}")

    except Exception as e:
        print(f"Error al buscar los mejores parámetros para {input_file}: {e}")

if __name__ == "__main__":
    preprocessed_data_dir = r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data\preprocesed_normalizada"
    results_dir = r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\results"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_preprocessed_data_dir = os.path.join(script_dir, preprocessed_data_dir)
    absolute_results_dir = os.path.join(script_dir, results_dir)

    for interval in ["1d", "1wk", "1mo"]:
        input_file = os.path.join(absolute_preprocessed_data_dir, f"BAP_{interval}.csv")
        if os.path.exists(input_file):
            find_best_parameters(input_file, absolute_results_dir)
        else:
            print(f"El archivo {input_file} no existe.")
