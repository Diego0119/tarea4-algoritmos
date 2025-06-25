/*
 * @file: knn.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 24/06/2025
 * @description: Cabecera general de funciones auxiliares para el algoritmo K-Nearest Neighbors (KNN).
 */

#ifndef KNN_H
#define KNN_H

#include "matrix.h"
#include "csv.h"

typedef struct
{
    Matrix *X_train; // Datos de entrenamiento
    Matrix *y_train; // Etiquetas de entrenamiento
    int k;           // Número de vecinos a considerar
} KNNClassifier;

typedef struct
{
    double distance; // Distancia euclidiana
    double label;    // Etiqueta de la clase
} DistanceLabel;

// Funciones del algoritmo K-Nearest Neighbors (KNN)
void exec_knn(CSVData *, int);
KNNClassifier *knn_create(int);
void knn_fit(KNNClassifier *, Matrix *, Matrix *);
Matrix *knn_predict(KNNClassifier *, Matrix *);
void knn_free(KNNClassifier *);
void quicksort(DistanceLabel *, int, int);
int partition(DistanceLabel *, int, int);
double euclidean_distance(const double *, const double *, int);

#endif