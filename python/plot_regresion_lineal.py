import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'stats/resultados_lr.csv'  
df = pd.read_csv(filename)

df.columns = df.columns.str.strip()

coefficients = df['X'].dropna().values 
predictions = df['Y predicción'].dropna().values 

X = np.linspace(0, 20, num=30)
y = 0.5 * X + 0.1 + np.random.randn(30) * 0.2  

# Graficar los puntos de datos reales
plt.scatter(X, y, color='blue', label='Datos reales')

# Graficar la línea de regresión utilizando los coeficientes
plt.plot(X, coefficients[0] * X + coefficients[1], color='red', label='Regresión')

# Añadir etiquetas y título
plt.xlabel('Longitud del Pétalo')
plt.ylabel('Ancho del Pétalo')
plt.title('Regresión Lineal: Longitud vs Ancho del Pétalo')

# Añadir leyenda
plt.legend()

# Agregar cuadrículas al fondo
plt.grid(True)

# Guardar el gráfico como imagen
plt.savefig('plots/regresion_lineal.png')

# Mostrar el gráfico
plt.show()
