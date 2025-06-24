/*
 * @file: knn.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 24/06/2025
 * @description: Desarrollo del algoritmo K-Nearest Neighbors (KNN) para clasificación de datos.
 */

#include "libs.h"
#include "knn.h"
#include "config.h"
#include "csv.h"
#include "errors.h"

// Aplicar algoritmo K-Vecinos Más Cercanos (KNN) al conjunto de datos Iris
void exec_knn(CSVData *csv_data)
{
    fprintf(stdout, CYAN_COLOR "K-Vecinos Más Cercanos (KNN)\n\n" RESET_COLOR);

    // Dividir en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    Matrix *X_train, *y_train, *X_test, *y_test;
    if (!train_test_split(csv_data->data, csv_data->labels, 0.2, &X_train, &y_train, &X_test, &y_test))
        train_test_split_error(__FILE__, __LINE__);

    fprintf(stdout, "Conjunto de entrenamiento: %d muestras x %d características\n", X_train->rows, X_train->cols);
    fprintf(stdout, "Conjunto de prueba: %d muestras x %d características\n", X_test->rows, X_test->cols);

    // Crear y entrenar el modelo KNN
    int k = 3;
    KNNClassifier *knn = knn_create(k);
    if (!knn)
        create_knn_classifier_error(__FILE__, __LINE__, X_train, y_train, X_test, y_test, knn);

    // Entrenar el modelo (knn_fit es void, no devuelve valor)
    knn_fit(knn, X_train, y_train);

    // Realizar predicciones
    Matrix *y_pred = knn_predict(knn, X_test);
    if (!y_pred)
        predict_knn_error(__FILE__, __LINE__, X_train, y_train, X_test, y_test, knn);

    // Calcular precisión (porcentaje de predicciones correctas)
    int correct = 0;
    for (int i = 0; i < y_test->rows; i++)
        if (y_test->data[i][0] == y_pred->data[i][0])
            correct++;

    double precision = (double)correct / y_test->rows;

    fprintf(stdout, "Precisión del modelo KNN (k=%d): %.4f\n", k, precision);

    // Mostrar algunas predicciones
    fprintf(stdout, "\nPrimeras 5 predicciones:\n");
    for (int i = 0; i < 5 && i < y_test->rows; i++)
        fprintf(stdout, "Real: %.0f, Predicción: %.0f\n", y_test->data[i][0], y_pred->data[i][0]);

    matrix_free(y_pred);
}

// Crea un clasificador KNN con el número de vecinos k especificado
KNNClassifier *knn_create(int k)
{
    return NULL;
}

// Ajustar el clasificador KNN con los datos de entrenamiento X y las etiquetas y
void knn_fit(KNNClassifier *knn, Matrix *X, Matrix *y)
{
}

// Predecir las etiquetas para los datos de entrada X utilizando el clasificador KNN
Matrix *knn_predict(KNNClassifier *knn, Matrix *X)
{
    return NULL;
}

// Liberar la memoria utilizada por el clasificador KNN
void knn_free(KNNClassifier *knn)
{
}