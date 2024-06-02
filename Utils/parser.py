import numpy as np

# Función para leer los datos

# Lee archivos y asigna  las caracteristicas y las variable dependiente


def parser(file_path, dependent_feature=0):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Se ignoran los encabezados, solo por ahora
    lines = lines[1:]

    x = []  # Características independientes
    y = []  # Datos dependientes

    for line in lines:
        parts = line.strip().split()
        if dependent_feature == 1:
            # Asume que la característica dependiente está en la última columna
            y.append(float(parts[-1]))
            x.append([float(value) for value in parts[1:-1]])
        else:
            # Asume que la característica dependiente está en la primera columna
            y.append(float(parts[1]))
            x.append([float(value) for value in parts[2:]])

    return np.array(x), np.array(y)
