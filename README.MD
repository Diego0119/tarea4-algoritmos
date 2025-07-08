# Algoritmos de Aprendizaje automático desde C

---

# Integrantes 👨‍💻

### 1. Miguel Loaiza 🤓
- Estudiante de Ingeniería Informática
- [EhMigueh](https://github.com/EhMigueh)
- **miloaiza@umag.cl**

### 2. Diego Sanhueza 🤓
- Estudiante de Ingeniería Informática
- [Diego0119] (https://github.com/Diego0119)
- **disanhue@umag.cl**

### 3. Duvan Figueroa 🤓
- Estudiante de Ingeniería Informática
- [HisokaMorow1 ] (https://github.com/HisokaMorow1)
- **dufiguer@umag.cl**

---

# Descripción ✏️

Este proyecto implementa tres algoritmos de aprendizaje automático en C: **K-Nearest Neighbors (KNN)**, **Regresión Lineal** y **K-Means Clustering**.   
  
El sistema está diseñado para procesar datasets en formato CSV y proporciona análisis completos con métricas de evaluación, matrices de confusión y exportación de resultados. Todos los algoritmos incluyen manejo de errores y optimizaciones. 

---

# Estructura de Directorios 📁

```bash
tarea4-algoritmos/
├── src/ # Código fuente principal
│ ├── main.c # Punto de entrada del programa
│ ├── commands.c # Procesamiento de argumentos CLI
│ ├── k-nn.c # Implementación KNN con múltiples métricas
│ ├── lr.c # Regresión lineal con descenso de gradiente
│ ├── k-means.c # Clustering K-Means con K-Means++
│ ├── load.c # Carga y normalización de datos CSV
│ ├── assist.c # Funciones auxiliares y ayuda
│ └── exeptions.c # Manejo centralizado de errores
├── incs/ # Archivos de cabecera
│ ├── k-nn.h # Definiciones para KNN
│ ├── lr.h # Estructuras de regresión lineal
│ ├── k-means.h # Tipos de datos para K-Means
│ └── errors.h # Declaraciones de manejo de errores
├── obj/ # Archivos objeto compilados
├── build/ # Ejecutable final
├── data/ # Datasets de entrada (iris.csv)
├── stats/ # Resultados exportados en CSV
├── python/ # Scripts de visualización
│ ├── plot_kmeans.py # Gráficos de clustering
│ └── plot_regresion_lineal.py # Visualización de regresión
└── docs/ # Documentación del proyecto
```
---

# Generación de Gráficos 📊

## K-Means Clustering 

El proyecto incluye scripts de Python para visualizar los resultados de los algoritmos:  

```bash 
cd python  
python plot_kmeans.py  
```

Genera un plot mostrando los clusters encontrados por K-Means usando las dimensiones petal_length vs petal_width del dataset Iris.

## Regresión Lineal

```bash 
cd python  
python plot_regresion_lineal.py
```


Requisitos para gráficos:

Python 3.x
pandas
matplotlib
numpy

---

# Requisitos de Compilación 🛠️

Dependencias del Sistema

GCC: Compilador GNU C
Make: Herramienta de automatización de compilación
libm: Biblioteca matemática estándar de C
Valgrind: (Opcional) Para análisis de memoria

Configuración del Compilador

CFLAGS: -Wall -Wextra -Wpedantic -O3 -g -Wno-stringop-truncation
LDFLAGS: -Wall -lm -g
Optimización: Nivel O3 para máximo rendimiento
Debug: Símbolos de depuración incluidos

---

# Ejecución del Programa ▶️

## Compilación

```bash
make all
```

## Comandos Disponibles

```bash
./build/program.out -h    # Mostrar ayuda  
```

## K-Nearest Neighbors

```bash
./build/program.out -k ./data/iris.csv 1
```

- Parámetros: <archivo_csv> <k>
- k: Número de vecinos (debe ser impar y > 0)
- Métricas: Euclidiana, Manhattan, y versiones ponderadas

## Regresión Lineal

```bash
./build/program.out -l ./data/iris.csv 0.01 2000 1e-8
```

- Parámetros: <archivo_csv> <learning_rate> <max_iter> <tolerancia>
- Método: Descenso de gradiente con regularización Ridge
- Métricas: MSE, MAE, R²


## K-Means Clustering

```bash
./build/program.out -m ./data/iris.csv 3 100 1e-4
```

- Parámetros: <archivo_csv> <k> <max_iter> <tolerancia>
- Inicialización: K-Means++ para mejor convergencia
- Salida: Matriz de confusión y asignaciones de cluster

## Ejecución Automatizada 

```bash
make run-all        # Ejecutar todos los algoritmos  
make run-all-val    # Ejecutar con análisis de memoria Valgrind  
make run-knn        # Solo KNN  
make run-lr         # Solo Regresión Lineal    
make run-km         # Solo K-Means
```