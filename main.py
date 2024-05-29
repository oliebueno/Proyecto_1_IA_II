import sys
from linear_regression import LinearRegression as ln
from Utils.parser import parser

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Uso: python main.py <nombre_archivo>")
    else:
        x, y = parser(sys.argv[1])

        l_regression = ln(x, y)

        l_regression.gradient_descend(1, 1, 1)
