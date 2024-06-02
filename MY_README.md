# Detalles del repositorio

## Descripción :rocket:
Este repositorio contiene la implementación del algoritmo de Descenso del Gradiente para resolver una Regresión Lineal Múltiple. El proyecto está desarrollado en Python y utiliza distintos datos para probar el algoritmo.

## Contenido del Repositorio :coffee:
- `Utils`: Carpeta que contiene implementaciones de utilidad.
- - `parser.py`: Parser para leer y procesar los datos para las dos primeros archivos de datos.
- - `ames_housing.py`: Contiene un parser para leer y procesar los datos para las dos primeros archivos de datos, además de funciones útiles para calcular métricas de los datos.
- `Data`: Carpeta que contiene los conjuntos de datos utilizados.
- `linear_regression.py`: Contiene una clase para ejecutar la Regresión Lineal Multiple.
- `main.py`: Contiene el programa principal donde ejecuta Regresión Lineal Multiple para cada uno de los datos, adémas de generar gráficas para cada uno de los resultados.
- `MY_README.md`: Este archivo.

## Requisitos
- Python 3.x
- Bibliotecas: numpy, matplotlib, pandas y sklearn.

## Instalación
Para instalar las dependencias del proyecto, ejecute el siguiente comando:

```bash
pip install numpy matplotlib pandas scikit-learn
```

## Ejecución
Para ejecutar el algoritmo de Descenso del Gradiente, utilice el siguiente comando:

```bash
python main.py <archivo_1> <archivo_2> <archivo_3>
```

- `archivo_1`: Archivo que contiene los datos de la parte 2.1
- `archivo_2`: Archivo que contiene los datos de la parte 2.2
- `archivo_3`: Archivo que contiene los datos de la parte 3
