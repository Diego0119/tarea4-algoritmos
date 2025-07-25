/*
 * @file: k-means.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 23/06/2025
 * @description: Cabecera general de funciones auxiliares para el algoritmo K-Means.
 */

#ifndef KMEANS_H
#define KMEANS_H

#include "matrix.h"
#include "csv.h"

typedef struct
{
    Matrix *centroids; // Matriz de centroides KxN
    int *labels;       // Vector de asignación de clusters para cada punto
    int max_iters;     // Iteraciones máximas
    double tolerance;  // Criterio de convergencia
} KMeansResult;

// Funciones del algoritmo K-Means
KMeansResult *kmeans_fit(Matrix *, int, int, double);
void kmeans_free(KMeansResult *);
Matrix *initialize_centroids(Matrix *, int);
double euclidean_distance(const double *, const double *, int);
void assign_clusters(Matrix *, Matrix *, int *);
void update_centroids(Matrix *, Matrix *, int *, int);
int has_converged(Matrix *, Matrix *, double);
Matrix *initialize_centroids_kmeans_pp(Matrix *, int);
void print_confusion_matrix_kmeans(Matrix *, int *, int);
void export_results_kmeans_to_csv(Matrix *, int *, int, const char *);
void exec_kmeans(CSVData *, int, int, double);

#endif
