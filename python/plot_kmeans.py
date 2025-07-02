import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris_path = '../data/iris.csv'
iris = pd.read_csv(iris_path)

clusters_path = '../stats/resultados_kmeans.csv'
clusters = pd.read_csv(clusters_path, on_bad_lines='skip')
clusters = clusters.iloc[:len(iris)] 

assert len(iris) == len(clusters), "El dataset y las asignaciones deben tener el mismo número de muestras."

iris['cluster'] = clusters['Cluster']

plt.figure(figsize=(10, 6))

colors = ['red', 'green', 'blue', 'purple', 'orange']

for cluster_id in sorted(iris['cluster'].unique()):
    cluster_points = iris[iris['cluster'] == cluster_id]
    plt.scatter(cluster_points['petal_length'], cluster_points['petal_width'],
                label=f'Cluster {cluster_id}', s=50, alpha=0.6)

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Clusters K-Means en Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()
