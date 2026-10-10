# Wine Quality — MLOps (Semana 4)

## Descripción

Proyecto individual para entrenar y comparar modelos **RandomForestRegressor** que predicen la variable `quality` del conjunto de datos **Wine Quality Red**. El entrenamiento se ejecuta desde `train.py`, con hiperparámetros configurables por línea de comandos, y registra los resultados con **MLflow**.

## Archivos del repositorio

- `train.py`: carga de datos, preprocesamiento, entrenamiento, evaluación y registro de experimentos.
- `requirements.txt`: dependencias del proyecto con versiones especificadas.
- `winequality-red.csv`: datos de entrada (si su redistribución está permitida).

> El archivo `winequality-red.csv` debe estar disponible en el directorio desde el cual se ejecuta el comando, o indicarse mediante `--data`.

## Instalación

Se recomienda crear un entorno virtual con una versión de Python compatible con las dependencias fijadas.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

En Windows, activa el entorno con `.venv\Scripts\activate`.

## Ejecución

Desde la carpeta raíz del repositorio:

```bash
python3 train.py --n_estimators 200 --max_depth 10 --min_samples_split 2 --random_state 42
```

Si el CSV está en otra ubicación:

```bash
python3 train.py --data ruta/al/winequality-red.csv --n_estimators 200 --max_depth 10 --min_samples_split 2 --random_state 42
```

También es posible ejecutar `python3 train.py` con los valores predeterminados del script: `n_estimators=100`, `max_depth=10`, `min_samples_split=2` y `random_state=42`.

## Experimentos y resultados

Se evaluaron cinco configuraciones con una partición de entrenamiento/prueba del 80 %/20 % y semilla 42. Se eliminaron los registros duplicados antes de entrenar.

| n_estimators | max_depth | min_samples_split | MAE | RMSE | R² |
|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 2 | 0.468208 | 0.617632 | 0.461472 |
| 100 | 5 | 2 | 0.474794 | 0.618021 | 0.460794 |
| **200** | **10** | **2** | **0.468232** | **0.614479** | **0.466957** |
| 200 | 15 | 4 | 0.466594 | 0.615673 | 0.464883 |
| 300 | 20 | 6 | 0.465630 | 0.616993 | 0.462587 |

La mejor configuración **según RMSE** fue `200 / 10 / 2`. La configuración `300 / 20 / 6` obtuvo el menor MAE, por lo que la elección depende de la métrica priorizada.

## Seguimiento con MLflow

El script registra los hiperparámetros, las métricas **MAE, RMSE y R²**, el artefacto `metrics.json` y el modelo entrenado dentro del experimento `Wine_Quality_RF_Experiments`.

Después de ejecutar el script, desde la misma carpeta puedes iniciar la interfaz de MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abre <http://127.0.0.1:5000> para consultar las corridas. La base de datos y los artefactos se generan localmente y no es necesario versionarlos en Git.

## Reproducibilidad

El parámetro `--random_state` fija la semilla de la división de datos y del modelo. Dos ejecuciones con la misma configuración y semilla produjeron las mismas métricas mostradas a cuatro decimales en el entorno utilizado.

## Fuente de datos

UCI Machine Learning Repository — **Wine Quality**: <https://archive.ics.uci.edu/dataset/186/wine+quality>.
