import sys
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
        alpha = 0.01
        epsilon = 1e-6
        max_iterations = 1000

        # Ejecuta el descenso de gradiente para obtener los parámetros óptimos
        theta_optimal, cost_history = lr_model.gradient_descend(
            alpha, epsilon, max_iterations)

        # Imprime los parámetros óptimos
        print("Parámetros óptimos (theta):", theta_optimal)

        # Grafica los datos originales
        plt.scatter(x[:, 1], y, label="Datos originales")

        # Calcula los valores predichos usando los parámetros óptimos
        x_pred = np.linspace(min(x[:, 1]), max(x[:, 1]), 100)
        y_pred = theta_optimal[0] + theta_optimal[1] * x_pred

        # Grafica la línea de regresión
        plt.plot(x_pred, y_pred, color='red', label="Regresión lineal")

        plt.xlabel("Característica independiente")
        plt.ylabel("Datos dependientes")
        plt.title("Regresión Lineal")
        plt.legend()
        plt.grid(True)
        plt.show()
