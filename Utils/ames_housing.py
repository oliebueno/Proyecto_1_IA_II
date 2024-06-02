import pandas as pd
import numpy as np
import sys

# Lee archivos .csv y retorna el marco de datos


def read_file(filr_path):
    try:
        df = pd.read_csv(filr_path)
        return df
    except:
        print("No se puede leer el archivo")
        sys.exit(-1)


# Normaliza los datos de las columnas indicadas


def normalize(df, columnas):

    df_normalizado = df.copy()

    for columna in columnas:
        media = df[columna].mean()
        desviacion_std = df[columna].std()
        # Evitar la división por cero
        desviacion_std = desviacion_std if desviacion_std > 0 else 1
        df_normalizado[columna] = (df[columna] - media) / desviacion_std

    return df_normalizado

# Funciones para calcular las métricas

# Función que calcula el sesgo promedio


def bias(y_real, y_pred):
    return np.mean(y_pred - y_real)


# Función que calcula la desviación máxima


def maximum_d(y_real, y_pred):
    return np.max(np.abs(y_real - y_pred))

# Función que calcula el promedio de la desviación media absoluta


def mad(y_real, y_pred):
    return np.mean(np.abs(y_real - y_pred))

# Función que calcula el promedio del error cuadrático medio


def mse(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2)
