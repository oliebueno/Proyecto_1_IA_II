import numpy as np
import math

# Clase que implementa la regresión lineal


class LinearRegression:

    # Inicialización de la clase
    def __init__(self, x, y):
        self.x = np.c_[np.ones(len(x)), np.array(x)]
        self.y = np.array(y)
        self.theta_actual = np.zeros(self.x.shape[1])
        self.theta_prev = np.zeros(self.x.shape[1])

    # Función que calcula la norm
    def norma2(self, x):
        return np.linalg.norm(x)

    # Hipotesis
    def h_0(self, x):
        x = np.c_[np.ones(len(x)), np.array(x)]
        return np.dot(x, self.theta_actual)

    # Función de costo
    def cost_function(self):
        return (1 / (2 * len(self.y))) * np.sum((np.dot(self.x, self.theta_actual) - self.y) ** 2)

    # Función para normalizar los datos
    def normalize(self):

        x_norm = self.x[:, 1:]
        mean = np.mean(x_norm, axis=0)
        std = np.std(x_norm, axis=0)

        # Evita la división por cero
        std[std == 0] = 1

        x_norm = (x_norm - mean) / std

        self.x = np.c_[self.x[:, 0], x_norm]
        return x_norm

    # Función que implementa el descenso del gradiente
    def gradient_descend(self, alpha, epsilon, max_ite):

        cost_history = [self.cost_function()]
        count = 0

        while True:
            gradient = (1 / len(self.y)) * \
                self.x.T.dot(self.x.dot(self.theta_actual) - self.y)
            self.theta_actual = self.theta_actual - alpha * gradient
            cost_history.append(self.cost_function())

            # Se comprueba la convergencia y el número de iteraciones
            if self.norma2(self.theta_prev - self.theta_actual) < epsilon or count > max_ite:
                break

            self.theta_prev = np.copy(self.theta_actual)
            count += 1

        return self.theta_actual, cost_history
