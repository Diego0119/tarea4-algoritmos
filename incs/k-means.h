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

Matrix *initialize_centroids(Matrix *data, int k);
double euclidean_distance(const double *a, const double *b, int length);
void assign_clusters(Matrix *data, Matrix *centroids, int *labels);
void update_centroids(Matrix *data, Matrix *centroids, int *labels, int k);
int has_converged(Matrix *old, Matrix *new, double tol);

#endif
