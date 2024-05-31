import sys
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression as ln
from Utils.parser import parser

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Uso: python main.py <nombre_archivo1> <nombre_archivo2>")
    else:

        # Parte 2.1.1

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
        plt.xlabel("Body Weight")
        plt.ylabel("Brain Weight")
        plt.title("Regresión Lineal - 2.1.1 b)")
        plt.legend()
        plt.grid(True)
        plt.show()

        print("Resultados 1:")
        print(f"Parámetros óptimos theta: {theta_optimal}")

        # Parte 2.1.2--------------------------------------

        # Crea una instancia de la clase LinearRegression
        lr_model_2 = ln(x, y)

        # Hiperparámetros para el descenso de gradiente
        alpha = 0.0000001
        epsilon = 1e-6
        max_iterations = 1000

        # Normaliza los datos
        x_norma = lr_model_2.normalize()

        # Ejecuta el descenso de gradiente para obtener los parámetros óptimos
        theta_optimal, cost_history = lr_model_2.gradient_descend(
            alpha, epsilon, max_iterations)

        # a) Curva de Convergencia (J() vs. Iteraciones)
        plt.plot(range(len(cost_history)), cost_history)
        plt.xlabel("Iteraciones")
        plt.ylabel("Costo J()")
        plt.title("Curva de Convergencia - 2.1.2 a)")
        plt.show()

        # b) Scatterplot de los Datos con la Curva de Regresión Lineal
        plt.scatter(x_norma[:, 0], y, label="Datos originales")
        y_pred = theta_optimal[0] + theta_optimal[1] * x_norma
        plt.plot(x_norma, y_pred, color='red', label="Regresión lineal")
        plt.xlabel("Body Weight")
        plt.ylabel("Brain Weight")
        plt.title("Regresión Lineal 2.1.2 b)")
        plt.legend()
        plt.grid(True)
        plt.show()

        print("Resultados 1 normalizados:")
        print(f"Parámetros óptimos theta: {theta_optimal}")

        # -------------Parte 2.2.1

        # Lectura del primer archivo
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
