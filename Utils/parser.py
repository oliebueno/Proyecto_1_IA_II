import numpy as np

# Función para leer los datos


def parser(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Se ignoran los encabezados, solo por ahora
    lines = lines[1:]

    x = []  # Caracteísticas independientes
    y = []  # Datos dependientes

    for line in lines:
        parts = line.strip().split()
        if len(parts) > 2:
            y.append(float(parts[1]))
            x.append([float(value) for value in parts[2:]])

    return np.array(x), np.array(y)
