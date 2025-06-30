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
    fprintf(stdout, CYAN_COLOR "K-Vecinos Mas Cercanos (KNN)\n\n" RESET_COLOR);

    // Dividir en conjuntos de entrenamiento, validación y prueba (60% entrenamiento, 20% validación y 20% prueba)
    Matrix *X_train, *y_train, *X_valid, *y_valid, *X_test, *y_test;
    if (!train_valid_test_split(csv_data->data, csv_data->labels, 0.2, 0.2, &X_train, &y_train, &X_valid, &y_valid, &X_test, &y_test))
        train_valid_test_split_error(__FILE__, __LINE__);

    fprintf(stdout, "Conjunto de entrenamiento: %d muestras x %d caracteristicas\n", X_train->rows, X_train->cols);
    fprintf(stdout, "Conjunto de validacion: %d muestras x %d caracteristicas\n", X_valid->rows, X_valid->cols);
    fprintf(stdout, "Conjunto de prueba: %d muestras x %d caracteristicas\n", X_test->rows, X_test->cols);

    // Crear y entrenar el modelo KNN
    KNNClassifier *knn = knn_create(k);
    if (!knn)
        create_knn_classifier_error(__FILE__, __LINE__, X_train, y_train, X_test, y_test, knn);

    // Entrenar el modelo (knn_fit es void, no devuelve valor)
    knn_fit(knn, X_train, y_train);

    fprintf(stdout, CYAN_COLOR "\nUsando Metrica de Distancia Euclidiana\n\n" RESET_COLOR);

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
        fprintf(stdout, "Real: %.0f, Prediccion: %.0f\n", y_test->data[i][0], y_pred_eucledian->data[i][0]);

    fprintf(stdout, GREEN_COLOR "\nPrecision del modelo KNN (k=%d, Euclidiana): %.4f\n\n" RESET_COLOR, k, precision_euclidean);

    // Mostrar matriz de confusión para distancia euclidiana
    fprintf(stdout, CYAN_COLOR "Matriz de Confusion (Euclidiana):\n\n" RESET_COLOR);
    print_confusion_matrix(y_test, y_pred_eucledian);

    fprintf(stdout, CYAN_COLOR "Usando Metrica de Distancia Manhattan\n\n" RESET_COLOR);

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
        fprintf(stdout, "Real: %.0f, Prediccion: %.0f\n", y_test->data[i][0], y_pred_manhattan->data[i][0]);

    fprintf(stdout, GREEN_COLOR "\nPrecision del modelo KNN (k=%d, Manhattan): %.4f\n\n" RESET_COLOR, k, precision_manhattan);

    // Mostrar matriz de confusión para distancia manhattan
    fprintf(stdout, CYAN_COLOR "Matriz de Confusion (Manhattan):\n\n" RESET_COLOR);
    print_confusion_matrix(y_test, y_pred_manhattan);

    export_results_knn_to_csv(y_test, y_pred_eucledian, k, "Euclidiana", "stats/resultados_knn_euclidiana.csv");
    export_results_knn_to_csv(y_test, y_pred_manhattan, k, "Manhattan", "stats/resultados_knn_manhattan.csv");
    fprintf(stdout, GREEN_COLOR "Resultados exportados a la carpeta stats.\n\n" RESET_COLOR);

    matrix_free(y_pred_eucledian);
    matrix_free(y_pred_manhattan);
    matrix_free(X_train);
    matrix_free(y_train);
    matrix_free(X_test);
    matrix_free(y_test);
    knn_free(knn);
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

// Función para calcular y mostrar la matriz de confusión
void print_confusion_matrix(Matrix *y_true, Matrix *y_pred)
{
    if (!y_true || !y_pred || y_true->rows != y_pred->rows)
        return;

    double classes[MAX_CLASSES];
    int num_classes = 0;

    // Buscar clases únicas en las etiquetas verdaderas
    for (int i = 0; i < y_true->rows; i++)
    {
        double current_class = y_true->data[i][0];
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

    // Ordenar las clases con Bubble Sort
    for (int i = 0; i < num_classes - 1; i++)
        for (int j = i + 1; j < num_classes; j++)
            if (classes[i] > classes[j]) // Si la clase i es mayor que la clase j
            {
                double temp = classes[i];
                classes[i] = classes[j]; // Intercambiar clases
                classes[j] = temp;
            }

    // Crear matriz de confusión
    int confusion_matrix[MAX_CLASSES][MAX_CLASSES];
    for (int i = 0; i < num_classes; i++)
        for (int j = 0; j < num_classes; j++)
            confusion_matrix[i][j] = 0;

    // Llenar la matriz de confusión
    for (int i = 0; i < y_true->rows; i++)
    {
        int true_idx = -1, pred_idx = -1;

        // Encontrar índice de la clase verdadera
        for (int j = 0; j < num_classes; j++)
            if (classes[j] == y_true->data[i][0])
            {
                true_idx = j;
                break;
            }

        // Encontrar índice de la clase predicha
        for (int j = 0; j < num_classes; j++)
            if (classes[j] == y_pred->data[i][0])
            {
                pred_idx = j;
                break;
            }

        if (true_idx >= 0 && pred_idx >= 0)
            confusion_matrix[true_idx][pred_idx]++;
    }

    fprintf(stdout, "         ");
    for (int i = 0; i < num_classes; i++)
        fprintf(stdout, "Pred %.0f  ", classes[i]);
    fprintf(stdout, "\n");

    for (int i = 0; i < num_classes; i++)
    {
        fprintf(stdout, "Real %.0f | ", classes[i]);
        for (int j = 0; j < num_classes; j++)
        {
            if (i == j) // Diagonal principal (predicciones correctas)
                fprintf(stdout, GREEN_COLOR "%6d" RESET_COLOR "  ", confusion_matrix[i][j]);
            else // Predicciones incorrectas
                fprintf(stdout, RED_COLOR "%6d" RESET_COLOR "  ", confusion_matrix[i][j]);
        }
        fprintf(stdout, "\n");
    }

    fprintf(stdout, "\n");

    for (int i = 0; i < num_classes; i++)
    {
        int tp = confusion_matrix[i][i]; // Verdaderos positivos
        int fp = 0, fn = 0;              // Falsos positivos y falsos negativos

        // Calcular falsos positivos (columna i, excluyendo diagonal)
        for (int j = 0; j < num_classes; j++)
            if (j != i)
                fp += confusion_matrix[j][i];

        // Calcular falsos negativos (fila i, excluyendo diagonal)
        for (int j = 0; j < num_classes; j++)
            if (j != i)
                fn += confusion_matrix[i][j];

        // Calcular precisión y recall
        double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
        double f1_score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0;

        fprintf(stdout, "Clase %.0f: Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n", classes[i], precision, recall, f1_score);
    }

    fprintf(stdout, "\n");
}

// Función para exportar resultados de KNN a un archivo CSV
void export_results_knn_to_csv(Matrix *y_true, Matrix *y_pred, int k, const char *method_name, const char *filename)
{
    if (!y_true || !y_pred || y_true->rows != y_pred->rows)
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
        if (!found && num_classes < 10)
        {
            classes[num_classes] = current_class;
            num_classes++;
        }
    }

    // Ordenar clases
    for (int i = 0; i < num_classes - 1; i++)
        for (int j = i + 1; j < num_classes; j++)
            if (classes[i] > classes[j])
            {
                double temp = classes[i];
                classes[i] = classes[j];
                classes[j] = temp;
            }

    // Matriz de confusión
    int confusion_matrix[10][10] = {0};
    for (int i = 0; i < y_true->rows; i++)
    {
        int true_idx = -1, pred_idx = -1;
        for (int j = 0; j < num_classes; j++)
        {
            if (classes[j] == y_true->data[i][0])
                true_idx = j;
            if (classes[j] == y_pred->data[i][0])
                pred_idx = j;
        }
        if (true_idx >= 0 && pred_idx >= 0)
            confusion_matrix[true_idx][pred_idx]++;
    }

    // Escribir encabezados en el archivo CSV
    fprintf(file, "Metodo,K,Clases\n");
    fprintf(file, "%s,%d,", method_name, k);
    for (int i = 0; i < num_classes; i++)
        fprintf(file, "%.0f%s", classes[i], (i < num_classes - 1) ? ";" : "\n");

    // Métricas por clase
    fprintf(file, "Clase,Precision,Recall,F1-Score\n");
    for (int i = 0; i < num_classes; i++)
    {
        int tp = confusion_matrix[i][i]; // Verdaderos positivos
        int fp = 0, fn = 0;              // Falsos positivos y falsos negativos

        for (int j = 0; j < num_classes; j++) // Calcular falsos positivos (columna i, excluyendo diagonal)
            if (j != i)
                fp += confusion_matrix[j][i];

        // Calcular falsos negativos (fila i, excluyendo diagonal)
        for (int j = 0; j < num_classes; j++)
            if (j != i)
                fn += confusion_matrix[i][j];

        double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
        double f1_score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0;

        fprintf(file, "%.0f,%.4f,%.4f,%.4f\n", classes[i], precision, recall, f1_score);
    }

    // Matriz de confusión
    fprintf(file, "\nMatriz de Confusion\n");
    fprintf(file, "Real/Pred");
    for (int i = 0; i < num_classes; i++)
        fprintf(file, ",%.0f", classes[i]);
    fprintf(file, "\n");

    for (int i = 0; i < num_classes; i++)
    {
        fprintf(file, "%.0f", classes[i]);
        for (int j = 0; j < num_classes; j++)
            fprintf(file, ",%d", confusion_matrix[i][j]);
        fprintf(file, "\n");
    }

    fclose(file);
}
