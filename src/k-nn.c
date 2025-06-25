/*
 * @file: knn.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 24/06/2025
 * @description: Desarrollo del algoritmo K-Nearest Neighbors (KNN) para clasificación de datos.
 */

#include "libs.h"
#include "k-nn.h"
#include "config.h"
#include "csv.h"
#include "errors.h"

// Aplicar algoritmo K-Vecinos Más Cercanos (KNN) al conjunto de datos Iris
void exec_knn(CSVData *csv_data, int k)
{
    fprintf(stdout, CYAN_COLOR "K-Vecinos Más Cercanos (KNN)\n\n" RESET_COLOR);

    // Dividir en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    Matrix *X_train, *y_train, *X_test, *y_test;
    if (!train_test_split(csv_data->data, csv_data->labels, 0.2, &X_train, &y_train, &X_test, &y_test))
        train_test_split_error(__FILE__, __LINE__);

    fprintf(stdout, "Conjunto de entrenamiento: %d muestras x %d características\n", X_train->rows, X_train->cols);
    fprintf(stdout, "Conjunto de prueba: %d muestras x %d características\n", X_test->rows, X_test->cols);

    // Crear y entrenar el modelo KNN
    KNNClassifier *knn = knn_create(k);
    if (!knn)
        create_knn_classifier_error(__FILE__, __LINE__, X_train, y_train, X_test, y_test, knn);

    // Entrenar el modelo (knn_fit es void, no devuelve valor)
    knn_fit(knn, X_train, y_train);

    fprintf(stdout, CYAN_COLOR "\nUsando Métrica de Distancia Euclidiana\n\n" RESET_COLOR);

    // Realizar predicciones con distancia Euclidiana
    Matrix *y_pred_eucledian = knn_predict(knn, X_test, 0); // 0 para usar distancia euclidiana
    if (!y_pred_eucledian)
        predict_knn_error(__FILE__, __LINE__, X_train, y_train, X_test, y_test, knn);

    // Calcular precisión con métrica euclidiana (porcentaje de predicciones correctas)
    int correct_euclidean = 0;
    for (int i = 0; i < y_test->rows; i++)
        if (y_test->data[i][0] == y_pred_eucledian->data[i][0])
            correct_euclidean++;

    double precision_euclidean = (double)correct_euclidean / y_test->rows;

    fprintf(stdout, YELLOW_COLOR "Primeras 5 predicciones:\n\n" RESET_COLOR);
    for (int i = 0; i < 5 && i < y_test->rows; i++)
        fprintf(stdout, "Real: %.0f, Predicción: %.0f\n", y_test->data[i][0], y_pred_eucledian->data[i][0]);

    fprintf(stdout, GREEN_COLOR "\nPrecisión del modelo KNN (k=%d, Euclidiana): %.4f\n\n" RESET_COLOR, k, precision_euclidean);

    fprintf(stdout, CYAN_COLOR "Usando Métrica de Distancia Manhattan\n\n" RESET_COLOR);

    // Realizar predicciones con distancia Manhattan
    Matrix *y_pred_manhattan = knn_predict(knn, X_test, 1); // 1 para usar distancia Manhattan
    if (!y_pred_manhattan)
        predict_knn_error(__FILE__, __LINE__, X_train, y_train, X_test, y_test, knn);

    // Calcular precisión con métrica manhattan (porcentaje de predicciones correctas)
    int correct_manhattan = 0;
    for (int i = 0; i < y_test->rows; i++)
        if (y_test->data[i][0] == y_pred_manhattan->data[i][0])
            correct_manhattan++;

    double precision_manhattan = (double)correct_manhattan / y_test->rows;

    fprintf(stdout, YELLOW_COLOR "Primeras 5 predicciones:\n\n" RESET_COLOR);
    for (int i = 0; i < 5 && i < y_test->rows; i++)
        fprintf(stdout, "Real: %.0f, Predicción: %.0f\n", y_test->data[i][0], y_pred_manhattan->data[i][0]);

    fprintf(stdout, GREEN_COLOR "\nPrecisión del modelo KNN (k=%d, Manhattan): %.4f\n\n" RESET_COLOR, k, precision_manhattan);

    matrix_free(y_pred_eucledian);
    matrix_free(y_pred_manhattan);
}

// Crea un clasificador KNN con el número de vecinos k especificado
KNNClassifier *knn_create(int k)
{
    if (k <= 0) // Verificar que k sea positivo
        return NULL;

    KNNClassifier *knn = (KNNClassifier *)malloc(sizeof(KNNClassifier)); // Asignar memoria para el clasificador KNN
    if (!knn)
        return NULL;

    // Inicializar el clasificador KNN
    knn->k = k;
    knn->X_train = NULL;
    knn->y_train = NULL;

    return knn;
}

// Ajustar el clasificador KNN con los datos de entrenamiento X y las etiquetas y
void knn_fit(KNNClassifier *knn, Matrix *X, Matrix *y)
{
    if (!knn || !X || !y) // Verificar que los parámetros sean válidos
        return;

    // Almacenar referencias a los datos de entrenamiento
    knn->X_train = X;
    knn->y_train = y;
}

// Predecir las etiquetas para los datos de entrada X utilizando el clasificador KNN
Matrix *knn_predict(KNNClassifier *knn, Matrix *X, int distance_metric)
{
    if (!knn || !X || !knn->X_train || !knn->y_train) // Verificar que el clasificador y los datos sean válidos
        return NULL;

    // Crear una matriz para almacenar las predicciones
    Matrix *predictions = matrix_create(X->rows, 1);
    if (!predictions)
        return NULL;

    // Para cada muestra en X, encontrar los k vecinos más cercanos y predecir la etiqueta
    for (int i = 0; i < X->rows; i++)
    {
        DistanceLabel *distances = (DistanceLabel *)malloc(knn->X_train->rows * sizeof(DistanceLabel));
        if (!distances)
        {
            matrix_free(predictions);
            return NULL;
        }

        // Calculo distancia euclidiana a todas las muestras de entrenamiento
        for (int j = 0; j < knn->X_train->rows; j++)
        {
            if (distance_metric == 0)
                distances[j].distance = euclidean_distance(X->data[i], knn->X_train->data[j], X->cols); // Calcular la distancia Euclidiana
            else if (distance_metric == 1)
                distances[j].distance = manhattan_distance(X->data[i], knn->X_train->data[j], X->cols); // Calcular la distancia Manhattan

            distances[j].label = knn->y_train->data[j][0]; // Almacenar la etiqueta correspondiente
        }

        // Ordenar las distancias para encontrar los k vecinos más cercanos
        quicksort(distances, 0, knn->X_train->rows - 1);

        // Encontrar la clase mayoritaria entre los k vecinos más cercanos, contar ocurrencias de cada clase
        int classes[MAX_CLASSES];
        int counts[MAX_CLASSES];
        int num_clases = 0;

        for (int k = 0; k < MAX_CLASSES; k++)
        {
            classes[k] = -1;
            counts[k] = 0;
        }

        // Contar votos de los k vecinos más cercanos
        for (int k_idx = 0; k_idx < knn->k && k_idx < knn->X_train->rows; k_idx++)
        {
            double current_label = distances[k_idx].label; // Etiqueta del vecino actual

            // Verificar si la clase ya está registrada
            int class_found = 0;
            for (int c = 0; c < num_clases; c++)
                if (classes[c] == current_label) // Si la clase ya está registrada
                {
                    counts[c]++;
                    class_found = 1;
                    break;
                }

            // Si la clase no está registrada, se añade
            if (!class_found && num_clases < MAX_CLASSES)
            {
                classes[num_clases] = current_label;
                counts[num_clases] = 1;
                num_clases++;
            }
        }

        // Encontrar la clase con más votos
        double predicted_class = classes[0];
        int max_votes = counts[0];

        for (int c = 1; c < num_clases; c++)
            if (counts[c] > max_votes) // Si la clase actual tiene más votos
            {
                max_votes = counts[c];
                predicted_class = classes[c];
            }

        predictions->data[i][0] = predicted_class; // Asignar la predicción a la matriz de resultados

        free(distances);
    }

    return predictions; // Devolver las predicciones
}

// Liberar la memoria utilizada por el clasificador KNN
void knn_free(KNNClassifier *knn)
{
    if (!knn) // Clasificador KNN es válido
        return;

    free(knn);
}

// Función QuickSort para ordenar por distancia
void quicksort(DistanceLabel *arr, int low, int high)
{
    if (low < high)
    {
        int pivot = partition(arr, low, high);
        quicksort(arr, low, pivot - 1);
        quicksort(arr, pivot + 1, high);
    }
}

// Partición para QuickSort
int partition(DistanceLabel *arr, int low, int high)
{
    double pivot = arr[high].distance;
    int i = (low - 1);

    for (int j = low; j < high; j++)
    {
        if (arr[j].distance <= pivot)
        {
            i++;
            DistanceLabel temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    DistanceLabel temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return i + 1;
}

// Función para calcular la distancia euclidiana entre dos vectores
double euclidean_distance(const double *x1, const double *x2, int n)
{
    double sum = 0.0; // Inicializar la suma de las diferencias al cuadrado

    for (int i = 0; i < n; i++)
    {
        double diff = x1[i] - x2[i]; // Calcular la diferencia entre los elementos
        sum += diff * diff;          // Sumar el cuadrado de la diferencia
    }

    return sqrt(sum); // Devolver la raíz cuadrada de la suma de las diferencias al cuadrado
}

double manhattan_distance(const double *x1, const double *x2, int n)
{
    double sum = 0.0; // Inicializar la suma de las diferencias absolutas

    for (int i = 0; i < n; i++)
    {
        double diff = fabs(x1[i] - x2[i]); // Calcular la diferencia absoluta entre los elementos
        sum += diff;                       // Sumar la diferencia absoluta
    }

    return sum; // Devolver la suma de las diferencias absolutas
}