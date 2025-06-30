### ***Descripción de los scripts***
**link:**https://dnzteosk.gensparkspace.com/
- **alert_system**  
  Detecta movimientos atípicos en la acción (por ejemplo, variaciones diarias fuera de lo normal).  
  **_FALTA MEJORARLO_**

- **data_extractor**  
  - Automatiza la extracción de históricos bursátiles desde Yahoo Finance.
  - Permite especificar el ticker, rango de fechas e intervalo (diario, semanal o mensual).
  - Los datos descargados se guardan en archivos CSV listos para su análisis posterior.
  - **Datos diarios:** Desde `2020-01-30`
  - **Datos semanales:** Desde `2010-12-01`
  - **Datos mensuales:** Desde `1996-01-01`

- **data_preprocessor**  
  - Elimina nulos, crea variables rezagadas (lags), calcula indicadores técnicos y normaliza los datos para facilitar el modelado.
  - Cálculo de medias móviles simples (SMA).
  - Creación de variables lag (ventanas deslizantes).

- **ramdom_forestR**  
  - Entrena modelos, busca los mejores parámetros mediante GridSearch y reporta métricas de desempeño.
  - Guarda las métricas y los mejores parámetros en archivos CSV.
  - Exporta el modelo entrenado en formato `.joblib` para posible uso futuro.
