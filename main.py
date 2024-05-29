import sys
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression as ln
from Utils.parser import parser

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Uso: python main.py <nombre_archivo>")
    else:
        x, y = parser(sys.argv[1])

        # Crea una instancia de la clase LinearRegression
        lr_model = ln(x, y)

        # Hiperparámetros para el descenso de gradiente
        alpha = 0.000001
        epsilon = 1e-6
        max_iterations = 1000

        # Ejecuta el descenso de gradiente para obtener los parámetros óptimos
        theta_optimal, cost_history = lr_model.gradient_descend(
            alpha, epsilon, max_iterations)

        # a) Curva de Convergencia (J() vs. Iteraciones)
        plt.plot(range(len(cost_history)), cost_history)
        plt.xlabel("Iteraciones")
        plt.ylabel("Costo J()")
        plt.title("Curva de Convergencia")
        plt.show()

        # b) Scatterplot de los Datos con la Curva de Regresión Lineal
        plt.scatter(x[:, 0], y, label="Datos originales")
        y_pred = theta_optimal[0] + theta_optimal[1] * x
        plt.plot(x, y_pred, color='red', label="Regresión lineal")
        plt.xlabel("Característica independiente")
        plt.ylabel("Datos dependientes")
        plt.title("Regresión Lineal")
        plt.legend()
        plt.grid(True)
        plt.show()

        print("Resultados:")
        print(f"Parámetros óptimos theta: {theta_optimal}")
        print(cost_history)
