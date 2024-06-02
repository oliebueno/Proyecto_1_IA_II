import sys
import numpy as np
import Utils.ames_housing as ames
import matplotlib.pyplot as plt
from linear_regression import LinearRegression as ln
from sklearn.model_selection import train_test_split
from Utils.parser import parser


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Uso: python main.py <nombre_archivo1> <nombre_archivo2> <nombre_archivo2>")
    else:

        # -----------Parte 2.1.1

        # Lectura del primer archivo
        x, y = parser(sys.argv[1], 1)

        # Crea una instancia de la clase LinearRegression
        lr_model = ln(x, y)

        # Hiperparámetros para el descenso de gradiente
        alpha = 0.0000001
        epsilon = 1e-6
        max_iterations = 1000

        # Ejecuta el descenso de gradiente para obtener los parámetros óptimos
        theta_optimal, cost_history = lr_model.gradient_descend(
            alpha, epsilon, max_iterations)

        # a) Curva de Convergencia (J() vs. Iteraciones)
        plt.plot(range(len(cost_history)), cost_history)
        plt.xlabel("Iteraciones")
        plt.ylabel("Costo J()")
        plt.title("Curva de Convergencia - 2.1.1 a)")
        plt.show()

        # b) Scatterplot de los Datos con la Curva de Regresión Lineal
        plt.scatter(x[:, 0], y, label="Datos originales")
        y_pred = theta_optimal[0] + theta_optimal[1] * x
        plt.plot(x, y_pred, color='red', label="Regresión lineal")
        plt.xlabel("Brain Weight")
        plt.ylabel("Body Weight")
        plt.title("Regresión Lineal - 2.1.1 b)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # -----------Parte 2.1.2

        # Crea una instancia de la clase LinearRegression
        lr_model_2 = ln(x, y)

        # Normaliza los datos
        x_norma = lr_model_2.normalize()

        # Ejecuta el descenso de gradiente para obtener los parámetros óptimos con alfa 0.001
        theta_optimal, cost_history = lr_model_2.gradient_descend(
            alpha=0.001, epsilon=1e-6, max_ite=10000)

        # a) Curva de Convergencia (J() vs. Iteraciones)
        plt.plot(range(len(cost_history)), cost_history)
        plt.xlabel("Iteraciones")
        plt.ylabel("Costo J()")
        plt.title("Curva de Convergencia, alfa = 0.001 - 2.1.2 a)")
        plt.show()

        # b) Scatterplot de los Datos con la Curva de Regresión Lineal
        plt.scatter(x_norma[:, 0], y, label="Datos originales")
        y_pred = theta_optimal[0] + theta_optimal[1] * x_norma
        plt.plot(x_norma, y_pred, color='red', label="Regresión lineal")
        plt.xlabel("Brain Weight")
        plt.ylabel("Body Weight")
        plt.title("Regresión Lineal 2.1.2 b)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Crea una instancia de la clase LinearRegression
        lr_model_2 = ln(x, y)

        # Normaliza los datos
        x_norma = lr_model_2.normalize()

        # Ejecuta el descenso de gradiente para obtener los parámetros óptimos con alfa 0.0000001
        theta_optimal, cost_history = lr_model_2.gradient_descend(
            alpha=0.0000001, epsilon=1e-6, max_ite=10000)

        # a) Curva de Convergencia (J() vs. Iteraciones)
        plt.plot(range(len(cost_history)), cost_history)
        plt.xlabel("Iteraciones")
        plt.ylabel("Costo J()")
        plt.title("Curva de Convergencia, alfa = 0.0000001 - 2.1.2 a)")
        plt.show()

        # -------------Parte 2.2.1

        # Lectura del segundo archivo
        x1, y1 = parser(sys.argv[2], 1)

        # Para datos sin normalizar
        lr_model_3 = ln(x1, y1)
        theta_optimal, cost_history = lr_model_3.gradient_descend(
            alpha=0.001, epsilon=1e-6, max_ite=10000)

        # Para datos normalizados
        lr_model_4 = ln(x1, y1)
        lr_model_4.normalize()  # Normaliza las características
        theta_optimal_norm, cost_history_norm = lr_model_4.gradient_descend(
            alpha=0.001, epsilon=1e-6, max_ite=10000)

        # Graficar ambas curvas
        plt.plot(range(len(cost_history)), cost_history,
                 label='Sin Normalizar', linestyle='-', color='blue')

        plt.xlabel('Iteraciones')
        plt.ylabel('Costo J()')
        plt.title('Curvas de Convergencia para alpha = 0.001 - 2.2.1')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.plot(range(len(cost_history_norm)), cost_history_norm,
                 label='Normalizado', linestyle='-', color='red')
        plt.xlabel('Iteraciones')
        plt.ylabel('Costo J()')
        plt.title('Curvas de Convergencia para alpha = 0.001 Normalizado - 2.2.1')
        plt.legend()
        plt.grid(True)
        plt.show()

        # ----------Parte 2.2.2
        # Lista de valores de alpha para probar
        alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 1]

        # Inicializa la figura de matplotlib
        plt.figure(figsize=(10, 8))

        # Ejecuta el descenso de gradiente para cada valor de alpha y grafica la curva de convergencia
        for alpha in alphas:
            lr_model = ln(x1, y1)
            lr_model.normalize()
            _, cost_history = lr_model.gradient_descend(
                alpha=alpha, epsilon=1e-6, max_ite=10000)
            plt.plot(range(len(cost_history)),
                     cost_history, label=f'Alpha = {alpha}')

        plt.xlabel('Iteraciones')
        plt.ylabel('Costo J()')
        plt.title('Curvas de Convergencia para Diferentes Valores de Alpha - 2.2.2')
        plt.legend()
        plt.grid(True)
        plt.show()

        # ----------Parte 3
        # Lectura del tercer archivo

        df = ames.read_file(sys.argv[3])

        # ----------Sección a)
        # Limpieza de datos

        # Filtrar por la condición de venta "normal"
        df = df[df['Sale Condition'] == 'Normal']

        # Eliminar casas con un área de vida (GR LIVE AREA) superior a 1500 pies cuadrados
        df = df[df['Gr Liv Area'] <= 1500]

        # ----------Sección b)
        # Normalización de los datos

        col_data = ['Total Bsmt SF', 'Gr Liv Area', 'Lot Area', 'Garage Cars',
                    'Fireplaces', 'Year Built', 'Garage Area', 'Bedroom AbvGr', 'SalePrice']

        col_to_norm = ['Total Bsmt SF', 'Gr Liv Area', 'Lot Area', 'Garage Cars',
                       'Fireplaces', 'Year Built', 'Garage Area', 'Bedroom AbvGr']

        # Se selecionan los datos a usar
        df = df[col_data]

        # normaliza los datos
        df = ames.normalize(df, col_to_norm)

        # ---------Sección c)
        # Se definen las características y la variable dependiente
        x_ames = df.drop('SalePrice', axis=1)
        y_ames = df['SalePrice']

        # Dividir los datos en conjuntos de entrenamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(
            x_ames, y_ames, test_size=0.2, random_state=42)

        # ---------Sección d)

        # Rasgos para los modelos

        # Se definen las columnas para cada modelo
        col_mod_0 = ['Total Bsmt SF', 'Gr Liv Area']
        col_mod_1 = ['Total Bsmt SF', 'Gr Liv Area', 'Lot Area', 'Garage Cars',
                     'Fireplaces']
        col_mod_2 = ['Total Bsmt SF', 'Gr Liv Area', 'Year Built',
                     'Garage Area', 'Bedroom AbvGr']

        # Valores de las métricas
        bias = []
        md = []
        mad = []
        mse = []

        # Modelo 0

        x_train_0 = x_train[col_mod_0]
        x_test_0 = x_test[col_mod_0]

        # Se entrena el modelo
        model_0 = ln(x_train_0, y_train)
        theta_optimal_norm, cost_history_norm = model_0.gradient_descend(
            alpha=0.001, epsilon=1e-6, max_ite=10000)

        # Se hacen las predicciones
        y_pred_0 = model_0.h_0(x_test_0)

        # Se calculan las métricas
        bias.append(ames.bias(y_pred_0, y_test))
        md.append(ames.maximum_d(y_pred_0, y_test))
        mad.append(ames.mad(y_pred_0, y_test))
        mse.append(ames.mse(y_pred_0, y_test))

        # Modelo 1

        x_train_1 = x_train[col_mod_1]
        x_test_1 = x_test[col_mod_1]

        # Se entrena el modelo
        model_1 = ln(x_train_1, y_train)
        theta_optimal_norm, cost_history_norm = model_1.gradient_descend(
            alpha=0.001, epsilon=1e-6, max_ite=10000)

        # Se hacen las predicciones
        y_pred_1 = model_1.h_0(x_test_1)

        # Se calculan las métricas
        bias.append(ames.bias(y_pred_1, y_test))
        md.append(ames.maximum_d(y_pred_1, y_test))
        mad.append(ames.mad(y_pred_1, y_test))
        mse.append(ames.mse(y_pred_1, y_test))

        # Modelo 2

        x_train_2 = x_train[col_mod_2]
        x_test_2 = x_test[col_mod_2]

        # Se entrena el modelo
        model_2 = ln(x_train_2, y_train)
        theta_optimal_norm, cost_history_norm = model_2.gradient_descend(
            alpha=0.001, epsilon=1e-6, max_ite=10000)

        # Se hacen las predicciones
        y_pred_2 = model_2.h_0(x_test_2)

        # Se calculan las métricas
        bias.append(ames.bias(y_pred_2, y_test))
        md.append(ames.maximum_d(y_pred_2, y_test))
        mad.append(ames.mad(y_pred_2, y_test))
        mse.append(ames.mse(y_pred_2, y_test))

        # Se crean los gráficos de barras para cada métrica

        models = ['Modelo 0', 'Modelo 1', 'Modelo 2']
        pos = np.arange(len(models))
        ancho = 0.2

        # Sesgo

        # Crear gráfico de barras
        plt.grid(axis='y', linestyle='-', linewidth=0.5)
        plt.bar(models, bias)

        # Añadir título y etiquetas a los ejes
        plt.title('Sesgo')
        plt.xlabel('Modelos')
        plt.ylabel('Sale Price $')

        # Mostrar el gráfico
        plt.show()

        # Desviación máxima

        # Crear gráfico de barras
        plt.grid(axis='y', linestyle='-', linewidth=0.5)
        plt.bar(models, md)

        # Añadir título y etiquetas a los ejes
        plt.title('Desviación máxima')
        plt.xlabel('Modelos')
        plt.ylabel('Sale Price $')

        # Mostrar el gráfico
        plt.show()

        # Desviación media absoluta

        # Crear gráfico de barras
        plt.grid(axis='y', linestyle='-', linewidth=0.5)
        plt.bar(models, mad)

        # Añadir título y etiquetas a los ejes
        plt.title('Desviación media absoluta')
        plt.xlabel('Modelos')
        plt.ylabel('Sale Price $')

        # Mostrar el gráfico
        plt.show()

        # Error cuadrático medio

        # Crear gráfico de barras
        plt.grid(axis='y', linestyle='-', linewidth=0.5)
        plt.bar(models, mse)

        # Añadir título y etiquetas a los ejes
        plt.title('Error cuadrático medio')
        plt.xlabel('Modelos')
        plt.ylabel('Sale Price $^2')

        # Mostrar el gráfico
        plt.show()
