/*
 * @file: k-means.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 24/06/2025
 * @description: Desarrollo del algoritmo K-Means para clustering de datos.
 */

#include "libs.h"
#include "k-means.h"
#include "config.h"
#include "matrix.h"

// Crea una copia aleatoria de K puntos del dataset como centroides iniciales
Matrix *initialize_centroids(Matrix *data, int k)
{
    Matrix *centroids = matrix_create(k, data->cols);

    for (int i = 0; i < k; i++)
    {
        int random_idx = rand() % data->rows;

        for (int j = 0; j < data->cols; j++)
            centroids->data[i][j] = data->data[random_idx][j];
    }

    return centroids;
}

// Asigna cada punto al centroide mas cercano
void assign_clusters(Matrix *data, Matrix *centroids, int *labels)
{
    for (int i = 0; i < data->rows; i++)
    {
        double min_dist = DBL_MAX;
        int best_cluster = -1;

        for (int c = 0; c < centroids->rows; c++)
        {
            double dist = euclidean_distance(data->data[i], centroids->data[c], data->cols);

            if (dist < min_dist)
            {
                min_dist = dist;
                best_cluster = c;
            }
        }

        labels[i] = best_cluster;
    }
}

// Actualiza los centroides como el promedio de sus puntos asignados
void update_centroids(Matrix *data, Matrix *centroids, int *labels, int k)
{
    int *counts = calloc(k, sizeof(int));

    for (int i = 0; i < k; i++)
        for (int j = 0; j < data->cols; j++)
            centroids->data[i][j] = 0.0;

    for (int i = 0; i < data->rows; i++)
    {
        int cluster = labels[i];
        counts[cluster]++;

        for (int j = 0; j < data->cols; j++)
            centroids->data[cluster][j] += data->data[i][j];
    }

    for (int i = 0; i < k; i++)
    {
        if (counts[i] == 0)
            continue; // Evitar division por 0

        for (int j = 0; j < data->cols; j++)
            centroids->data[i][j] /= counts[i];
    }

    free(counts);
}

// Verifica si los centroides han cambiado muy poco (criterio de convergencia)
int has_converged(Matrix *old, Matrix *new, double tol)
{
    for (int i = 0; i < old->rows; i++)
        if (euclidean_distance(old->data[i], new->data[i], old->cols) > tol)
            return 0; // No convergio

    return 1; // Convergioo
}

// Ajusta el algoritmo kmeans
KMeansResult *kmeans_fit(Matrix *data, int k, int max_iters, double tol)
{
    // Matrix *centroids = initialize_centroids(data, k);
    Matrix *centroids = initialize_centroids_kmeans_pp(data, k); // Acá se aplica la optimizacion
    Matrix *old_centroids = matrix_create(k, data->cols);
    int *labels = malloc(sizeof(int) * data->rows);

    for (int iter = 0; iter < max_iters; iter++)
    {
        assign_clusters(data, centroids, labels);

        // Guardar copia de centroides actuales
        for (int i = 0; i < k; i++)
            for (int j = 0; j < data->cols; j++)
                old_centroids->data[i][j] = centroids->data[i][j];

        update_centroids(data, centroids, labels, k);

        if (has_converged(old_centroids, centroids, tol))
            break;
    }

    matrix_free(old_centroids);

    KMeansResult *result = malloc(sizeof(KMeansResult));
    result->centroids = centroids;
    result->labels = labels;
    result->max_iters = max_iters;
    result->tolerance = tol;

    return result;
}

// Libera la memoria utilizada por el resultado del K-Means
void kmeans_free(KMeansResult *result)
{
    matrix_free(result->centroids);
    free(result->labels);
    free(result);
}

// Optimizacion
Matrix *initialize_centroids_kmeans_pp(Matrix *data, int k)
{
    Matrix *centroids = matrix_create(k, data->cols);

    // Primer centroide aleatorio
    int first_idx = rand() % data->rows;
    for (int j = 0; j < data->cols; j++)
        centroids->data[0][j] = data->data[first_idx][j];

    double *distances = malloc(sizeof(double) * data->rows);

    for (int c = 1; c < k; c++)
    {
        double sum = 0.0;

        // Para cada punto calcula la minima distancia al cuadrado a centroides ya seleccionados
        for (int i = 0; i < data->rows; i++)
        {
            double min_dist = DBL_MAX;

            for (int m = 0; m < c; m++)
            {
                double dist = euclidean_distance(data->data[i], centroids->data[m], data->cols);

                if (dist < min_dist)
                    min_dist = dist;
            }

            distances[i] = min_dist * min_dist; // Distancia al cuadrado
            sum += distances[i];
        }

        // Nuevo centroide con probabilidad proporcional a distancia al cuadrado
        double r = ((double)rand() / RAND_MAX) * sum;
        double acc = 0.0;
        int next_idx = 0;

        for (int i = 0; i < data->rows; i++)
        {
            acc += distances[i];

            if (acc >= r)
            {
                next_idx = i;
                break;
            }
        }

        // Copiar punto seleccionado como nuevo centroide
        for (int j = 0; j < data->cols; j++)
            centroids->data[c][j] = data->data[next_idx][j];
    }

    free(distances);

    return centroids;
}
