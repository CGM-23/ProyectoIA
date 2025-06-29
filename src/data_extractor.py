import yfinance as yf
import pandas as pd
import os

def extract_data(ticker, start_date=None, end_date=None, interval='1d', output_dir=r'C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data'):
    """
    Extrae datos históricos de un ticker usando yfinance y los guarda en un archivo CSV.

    Args:
        ticker (str): Símbolo del ticker (ej. 'BAP').
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'. Si es None, extrae desde el inicio.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'. Si es None, extrae hasta la fecha actual.
        interval (str): Intervalo de los datos (ej. '1d' para diario, '1wk' para semanal, '1mo' para mensual).
        output_dir (str): Directorio donde se guardará el archivo CSV.
    """
    print(f"Extrayendo datos para {ticker}...")
    try:
        # Ajustar el directorio de salida relativo a la ubicación del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_output_dir = os.path.join(script_dir, output_dir)

        # Descargar los datos usando yfinance
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        # Verificar si se descargaron datos
        if data.empty:
            print(f"No se encontraron datos para {ticker} con el intervalo {interval} en el rango especificado.")
            return

        # Crear el directorio si no existe
        os.makedirs(absolute_output_dir, exist_ok=True)
        
        # Definir la ruta del archivo y guardar los datos en CSV
        file_path = os.path.join(absolute_output_dir, f'{ticker}_{interval}.csv')
        data.to_csv(file_path)
        print(f"Datos guardados en {file_path}")
    except Exception as e:
        print(f"Error al extraer datos para {ticker}: {e}")

if __name__ == '__main__': 
    # Extraer los datos para diferentes intervalos de tiempo
    extract_data('BAP', '2020-01-30', interval='1d', output_dir=r'C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data') #data diaria
    extract_data('BAP', '2010-12-01', interval='1wk', output_dir=r'C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data') #data semanal
    extract_data('BAP', '1996-01-01', interval='1mo', output_dir=r'C:\Users\PC\Desktop\EJERCICIOS_PROGRA\python\ProyectoIA\data') #data mensual
