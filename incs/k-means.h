/*
 * @file: k-means.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Cabecera general de funciones auxiliares.
 */

#ifndef KMEANS_H
#define KMEANS_H

#include "matrix.h"

typedef struct
{
    Matrix *centroids; // Matriz de centroides KxN
    int *labels;       // Vector de asignación de clusters para cada punto
    int max_iters;     // Iteraciones máximas
    double tolerance;  // Criterio de convergencia
} KMeansResult;

KMeansResult *kmeans_fit(Matrix *data, int k, int max_iters, double tol);
void kmeans_free(KMeansResult *result);

#endif
