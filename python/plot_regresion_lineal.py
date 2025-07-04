import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Lee el CSV de resultados generado por el modelo en C
    df = pd.read_csv('stats/resultados_lr.csv')
    df.columns = df.columns.str.strip()  # Limpia espacios en los nombres de columna
    x = df['x'].values
    y_real = df['y_realista'].values
    y_pred = df['y_prediccion'].values

    # Ordenar por x para graficar la línea correctamente
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_pred_sorted = y_pred[idx]

    plt.figure(figsize=(10, 6))
    # Puntos reales
    plt.scatter(x, y_real, label='Datos reales', color='blue')
    # Línea de predicción del modelo en C
    plt.plot(x_sorted, y_pred_sorted, label='Regresión', color='red', linewidth=2)

    plt.xlabel('Longitud del Pétalo')
    plt.ylabel('Ancho del Pétalo')
    plt.title('Regresión Lineal: Longitud vs Ancho del Pétalo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/regresion_lineal.png")

if __name__ == '__main__':
    main()
