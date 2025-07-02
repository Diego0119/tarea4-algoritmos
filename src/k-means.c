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
#include "csv.h"
#include "errors.h"

void exec_kmeans(CSVData *csv_data, int k, int max_iters, double tol)
{
    fprintf(stdout, CYAN_COLOR "Algoritmo K-Means\n\n" RESET_COLOR);

    Matrix *X = csv_data->data;
    Matrix *y_true = csv_data->labels;

    fprintf(stdout, "Conjunto completo: %d muestras x %d características\n", X->rows, X->cols);

    KMeansResult *result = kmeans_fit(X, k, max_iters, tol);
    if (!result)
        kmeans_fit_error(__FILE__, __LINE__, X);

    fprintf(stdout, GREEN_COLOR "\nK-Means ajustado correctamente (k=%d, iter máx=%d, tol=%.5f)\n\n" RESET_COLOR, k, max_iters, tol);

    fprintf(stdout, YELLOW_COLOR "Primeras 5 asignaciones:\n\n" RESET_COLOR);
    for (int i = 0; i < 5 && i < X->rows; i++)
        fprintf(stdout, "Real: %.0f, Cluster asignado: %d\n", y_true->data[i][0], result->labels[i]);

    fprintf(stdout, CYAN_COLOR "\nMatriz de Confusión K-Means:\n\n" RESET_COLOR);
    print_confusion_matrix_kmeans(y_true, result->labels, k);

    export_results_kmeans_to_csv(y_true, result->labels, k, "stats/resultados_kmeans.csv");
    fprintf(stdout, GREEN_COLOR "\nResultados exportados a stats/resultados_kmeans.csv\n\n" RESET_COLOR);

    kmeans_free(result);
}

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

void print_confusion_matrix_kmeans(Matrix *y_true, int *y_pred, int k)
{
    if (!y_true || !y_pred || y_true->rows <= 0)
        return;

    int classes[MAX_CLASSES];
    int num_classes = 0;

    for (int i = 0; i < y_true->rows; i++)
    {
        int current_class = (int)(y_true->data[i][0] + 0.5);
        int found = 0;
        for (int j = 0; j < num_classes; j++)
        {
            if (classes[j] == current_class)
            {
                found = 1;
                break;
            }
        }
        if (!found && num_classes < MAX_CLASSES)
        {
            classes[num_classes] = current_class;
            num_classes++;
        }
    }

    for (int i = 0; i < num_classes - 1; i++)
        for (int j = i + 1; j < num_classes; j++)
            if (classes[i] > classes[j])
            {
                int temp = classes[i];
                classes[i] = classes[j];
                classes[j] = temp;
            }

    int confusion_matrix[MAX_CLASSES][MAX_CLASSES] = {0};

    for (int i = 0; i < y_true->rows; i++)
    {
        int true_idx = -1;
        int pred_idx = y_pred[i];

        int true_class = (int)(y_true->data[i][0] + 0.5);
        for (int j = 0; j < num_classes; j++)
        {
            if (classes[j] == true_class)
            {
                true_idx = j;
                break;
            }
        }

        if (true_idx >= 0 && pred_idx >= 0 && pred_idx < k)
            confusion_matrix[true_idx][pred_idx]++;
    }

    printf("\n" CYAN_COLOR "Matriz de Confusion (K-Means):\n" RESET_COLOR);
    printf("         ");
    for (int i = 0; i < k; i++)
        printf("Pred %d  ", i);
    printf("\n");

    for (int i = 0; i < num_classes; i++)
    {
        printf("Real %d | ", classes[i]);
        for (int j = 0; j < k; j++)
        {
            if (confusion_matrix[i][j] == 0)
                printf("%6d  ", confusion_matrix[i][j]);
            else if (j == i) // diagonal, aciertos
                printf(GREEN_COLOR "%6d" RESET_COLOR "  ", confusion_matrix[i][j]);
            else
                printf(RED_COLOR "%6d" RESET_COLOR "  ", confusion_matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void export_results_kmeans_to_csv(Matrix *y_true, int *y_pred, int k, const char *filename)
{
    if (!y_true || !y_pred || y_true->rows <= 0)
        return;

    FILE *file = fopen(filename, "w");
    if (!file)
        open_file_error(__FILE__, __LINE__, filename);

    double classes[MAX_CLASSES];
    int num_classes = 0;

    for (int i = 0; i < y_true->rows; i++)
    {
        double current_class = y_true->data[i][0];
        int found = 0;
        for (int j = 0; j < num_classes; j++)
            if (classes[j] == current_class)
            {
                found = 1;
                break;
            }
        if (!found && num_classes < MAX_CLASSES)
        {
            classes[num_classes] = current_class;
            num_classes++;
        }
    }

    for (int i = 0; i < num_classes - 1; i++)
        for (int j = i + 1; j < num_classes; j++)
            if (classes[i] > classes[j])
            {
                double temp = classes[i];
                classes[i] = classes[j];
                classes[j] = temp;
            }

    int confusion_matrix[MAX_CLASSES][MAX_CLASSES] = {0};

    for (int i = 0; i < y_true->rows; i++)
    {
        int true_idx = -1, pred_idx = y_pred[i];

        for (int j = 0; j < num_classes; j++)
            if (classes[j] == y_true->data[i][0])
            {
                true_idx = j;
                break;
            }

        if (true_idx >= 0 && pred_idx >= 0 && pred_idx < k)
            confusion_matrix[true_idx][pred_idx]++;
    }

    fprintf(file, "Clase,Cluster\n");
    for (int i = 0; i < y_true->rows; i++)
        fprintf(file, "%.0f,%d\n", y_true->data[i][0], y_pred[i]);

    fprintf(file, "\nMatriz de Confusion\n");
    fprintf(file, "Real/Pred");
    for (int i = 0; i < k; i++)
        fprintf(file, ",%d", i);
    fprintf(file, "\n");

    for (int i = 0; i < num_classes; i++)
    {
        fprintf(file, "%.0f", classes[i]);
        for (int j = 0; j < k; j++)
            fprintf(file, ",%d", confusion_matrix[i][j]);
        fprintf(file, "\n");
    }

    fclose(file);
}
