#vamos a mostrar la matriz de correlacion de los datos preprocesados
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_correlation_matrix(input_file, output_dir):
    try:
        # Cargar los datos preprocesados
        data = pd.read_csv(input_file, index_col=0, parse_dates=True)

        # Calcular la matriz de correlación
        corr = data.corr()

        # Crear un mapa de calor
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
        plt.title("Matriz de Correlación")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        # Guardar la figura
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"correlation_matrix_{os.path.basename(input_file).replace('.csv', '')}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Matriz de correlación guardada en {output_file}")

    except Exception as e:
        print(f"Error al generar la matriz de correlación para {input_file}: {e}")
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data\preprocesed_normalizada")
    output_dir = os.path.join(script_dir, r"C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\results")

    # Cambia el nombre del archivo según sea necesario
    input_file = os.path.join(data_dir, "BAP_1d.csv")

    plot_correlation_matrix(input_file, output_dir)