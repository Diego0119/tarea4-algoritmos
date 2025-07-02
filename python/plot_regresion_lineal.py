import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = 'stats/resultados_lr.csv'  
data = pd.read_csv(csv_path)

data.columns = [col.lower() for col in data.columns]

# Ajuste para tus columnas: 'x', 'y_real', 'y_predicho'
if 'x' in data.columns and 'y_real' in data.columns and 'y_predicho' in data.columns:
    x = data['x']
    y = data['y_real']
else:
    print('Columnas detectadas:', data.columns)
    raise ValueError('Ajusta los nombres de columna en el script.')

# Ajuste por mínimos cuadrados para graficar la recta de regresión
coef = np.polyfit(x, y, 1)  # Ajuste lineal: grado 1
x_line = np.linspace(x.min(), x.max(), 100)
y_line = np.polyval(coef, x_line)

plt.figure(figsize=(10,6))
plt.scatter(x, y, color='blue', label='Datos reales')
plt.plot(x_line, y_line, color='red', label='Regresion (ajuste lineal)')
plt.xlabel('Longitud del Petalo')
plt.ylabel('Ancho del Petalo')
plt.title('Regresion Lineal: Longitud vs Ancho del Petalo')
plt.legend()
plt.grid(True)
plt.savefig('plots/regresion_lineal.png')
plt.show()
