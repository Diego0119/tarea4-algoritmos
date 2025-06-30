/*
 * @file: linear-regression.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 25/06/2025
 * @description: desarrollo del algoritmo de regresion lineal para prediccion de valores continuos.
 */

#include "libs.h"
#include "lr.h"
#include "config.h"
#include "csv.h"
#include "errors.h"

// aplicar algoritmo de regresion lineal al conjunto de datos
void exec_linear_regression(CSVData *csv_data, double learning_rate, int max_iterations, double tolerance)
{
    fprintf(stdout, CYAN_COLOR "=== regresion lineal ===\n" RESET_COLOR);

    // dividir en conjuntos de entrenamiento y prueba (60% entrenamiento, 20% validación?? y 20% prueba)
    Matrix *X_train, *y_train, *X_valid, *y_valid, *X_test, *y_test;
    double test_ratio = 0.2;
    double valid_ratio = 0.2; // No sé si se usa en la regresión lineal
    if (!train_valid_test_split(csv_data->data, csv_data->labels, valid_ratio, test_ratio, &X_train, &y_train, &X_valid, &y_valid, &X_test, &y_test))
        train_valid_test_split_error(__FILE__, __LINE__);

    // normalizar las caracteristicas para mejorar la convergencia
    double *means = (double *)calloc(X_train->cols, sizeof(double));
    double *stds = (double *)calloc(X_train->cols, sizeof(double));

    if (!means || !stds)
    {
        if (means)
            free(means);
        if (stds)
            free(stds);
        matrix_free(X_train);
        matrix_free(y_train);
        matrix_free(X_test);
        matrix_free(y_test);
        return;
    }

    // calcular medias y desviaciones estandar del conjunto de entrenamiento
    for (int j = 0; j < X_train->cols; j++)
    {
        for (int i = 0; i < X_train->rows; i++)
            means[j] += X_train->data[i][j];
        means[j] /= X_train->rows;

        for (int i = 0; i < X_train->rows; i++)
            stds[j] += (X_train->data[i][j] - means[j]) * (X_train->data[i][j] - means[j]);
        stds[j] = sqrt(stds[j] / X_train->rows);

        // evitar division por cero
        if (stds[j] < 1e-8)
            stds[j] = 1.0;
    }

    // normalizar conjuntos de entrenamiento y prueba
    normalize_features_with_stats(X_train, means, stds, 0);
    normalize_features_with_stats(X_test, means, stds, 0);

    // crear el modelo de regresion lineal
    LinearRegression *lr = linear_regression_create(X_train->cols, learning_rate, max_iterations, tolerance);
    if (!lr)
        create_linear_regression_error(__FILE__, __LINE__, X_train, y_train, X_test, y_test, lr);

    // entrenar el modelo
    RegressionMetrics *metrics = NULL;
    linear_regression_fit(lr, X_train, y_train, &metrics);

    // realizar predicciones en el conjunto de prueba
    Matrix *y_pred = linear_regression_predict(lr, X_test);
    if (!y_pred)
        predict_linear_regression_error(__FILE__, __LINE__, X_train, y_train, X_test, y_test, lr);

    // calcular metricas en el conjunto de prueba
    double test_mse = linear_regression_mse(y_test, y_pred);
    double test_r2 = linear_regression_r2_score(y_test, y_pred);

    // mostrar resultados en formato simple y limpio
    if (lr->weights->rows == 1)
    {
        // regresion lineal simple (una sola caracteristica)
        fprintf(stdout, "coeficiente: %.4f\n", lr->weights->data[0][0]);
    }
    else
    {
        // regresion lineal multiple (varias caracteristicas)
        fprintf(stdout, "coeficientes: ");
        for (int i = 0; i < lr->weights->rows; i++)
        {
            fprintf(stdout, "%.4f", lr->weights->data[i][0]);
            if (i < lr->weights->rows - 1)
                fprintf(stdout, ", ");
        }
        fprintf(stdout, "\n");
    }

    fprintf(stdout, "intercepto: %.4f\n", lr->bias);
    fprintf(stdout, "error cuadratico medio: %.4f\n", test_mse);
    fprintf(stdout, "coeficiente r²: %.4f\n", test_r2);

    // liberar memoria
    free(means);
    free(stds);
    matrix_free(y_pred);
    linear_regression_free(lr);
    regression_metrics_free(metrics);
    matrix_free(X_train);
    matrix_free(y_train);
    matrix_free(X_test);
    matrix_free(y_test);
}

// crear un modelo de regresion lineal
LinearRegression *linear_regression_create(int n_features, double learning_rate, int max_iterations, double tolerance)
{
    LinearRegression *lr = (LinearRegression *)malloc(sizeof(LinearRegression));
    if (!lr)
        return NULL;

    // inicializar pesos con valores aleatorios pequenos
    lr->weights = matrix_create(n_features, 1);
    if (!lr->weights)
    {
        free(lr);
        return NULL;
    }

    // inicializar pesos con inicializacion xavier/glorot para mejor convergencia
    double limit = sqrt(6.0 / (n_features + 1)); // xavier initialization
    // xavier o glorot initialization = inicializar peso de una red
    for (int i = 0; i < n_features; i++)
        lr->weights->data[i][0] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * limit;

    lr->bias = 0.0;
    lr->learning_rate = learning_rate;
    lr->max_iterations = max_iterations;
    lr->tolerance = tolerance;

    return lr;
}

// entrenar el modelo de regresion lineal usando gradiente descendente
void linear_regression_fit(LinearRegression *lr, Matrix *X, Matrix *y, RegressionMetrics **metrics)
{
    if (!lr || !X || !y)
        return;

    // crear estructura para metricas
    *metrics = (RegressionMetrics *)malloc(sizeof(RegressionMetrics));
    if (!*metrics)
        return;

    (*metrics)->mse_history = (double *)malloc(lr->max_iterations * sizeof(double));
    (*metrics)->cost_history = (double *)malloc(lr->max_iterations * sizeof(double));
    if (!(*metrics)->mse_history || !(*metrics)->cost_history)
    {
        regression_metrics_free(*metrics);
        *metrics = NULL;
        return;
    }

    double prev_mse = DBL_MAX;
    int iteration = 0;

    // gradiente descendente
    for (iteration = 0; iteration < lr->max_iterations; iteration++)
    {
        // calcular predicciones usando multiplicacion de matrices: y_pred = X * weights
        Matrix *predictions = matrix_multiply(X, lr->weights);
        if (!predictions)
        {
            regression_metrics_free(*metrics);
            *metrics = NULL;
            return;
        }

        // agregar bias a todas las predicciones
        for (int i = 0; i < predictions->rows; i++)
            predictions->data[i][0] += lr->bias;

        // calcular mse
        double mse = linear_regression_mse(y, predictions);
        (*metrics)->mse_history[iteration] = mse;
        (*metrics)->cost_history[iteration] = mse / 2.0; // funcion de costo

        // verificar convergencia
        if (iteration > 0 && fabs(prev_mse - mse) < lr->tolerance)
        {
            matrix_free(predictions);
            break;
        }

        // actualizar pesos y bias usando gradiente descendente con operaciones matriciales
        gradient_descent_step(lr, X, y, predictions);

        prev_mse = mse;
        matrix_free(predictions);
    }

    // calcular metricas finales
    Matrix *final_predictions = matrix_multiply(X, lr->weights);
    if (final_predictions)
    {
        // agregar bias
        for (int i = 0; i < final_predictions->rows; i++)
            final_predictions->data[i][0] += lr->bias;

        (*metrics)->n_iterations = iteration + 1;
        (*metrics)->final_mse = linear_regression_mse(y, final_predictions);
        (*metrics)->r2_score = linear_regression_r2_score(y, final_predictions);

        matrix_free(final_predictions);
    }
}

// realizar predicciones con el modelo entrenado
Matrix *linear_regression_predict(LinearRegression *lr, Matrix *X)
{
    if (!lr || !X)
        return NULL;

    // usar multiplicacion de matrices: y_pred = X * weights
    Matrix *predictions = matrix_multiply(X, lr->weights);
    if (!predictions)
        return NULL;

    // agregar bias a todas las predicciones
    for (int i = 0; i < predictions->rows; i++)
        predictions->data[i][0] += lr->bias;

    return predictions;
}

// liberar memoria del modelo de regresion lineal
void linear_regression_free(LinearRegression *lr)
{
    if (!lr)
        return;

    if (lr->weights)
        matrix_free(lr->weights);

    free(lr);
}

// calcular el error cuadratico medio (mse)
double linear_regression_mse(Matrix *y_true, Matrix *y_pred)
{
    if (!y_true || !y_pred || y_true->rows != y_pred->rows)
        return -1.0;

    double sum = 0.0;
    for (int i = 0; i < y_true->rows; i++)
    {
        double error = y_true->data[i][0] - y_pred->data[i][0];
        sum += error * error;
    }

    return sum / y_true->rows;
}

// calcular el coeficiente de determinacion r²
double linear_regression_r2_score(Matrix *y_true, Matrix *y_pred)
{
    if (!y_true || !y_pred || y_true->rows != y_pred->rows)
        return -1.0;

    // calcular la media de y_true
    double mean_y = 0.0;
    for (int i = 0; i < y_true->rows; i++)
        mean_y += y_true->data[i][0];
    mean_y /= y_true->rows;

    // calcular suma de cuadrados total y residual
    double ss_tot = 0.0; // suma total de cuadrados
    double ss_res = 0.0; // suma residual de cuadrados

    for (int i = 0; i < y_true->rows; i++)
    {
        double y_val = y_true->data[i][0];
        double y_pred_val = y_pred->data[i][0];

        ss_tot += (y_val - mean_y) * (y_val - mean_y);
        ss_res += (y_val - y_pred_val) * (y_val - y_pred_val);
    }

    // r² = 1 - (ss_res / ss_tot)
    if (ss_tot == 0.0)
        return 1.0; // caso perfecto donde todas las y son iguales

    return 1.0 - (ss_res / ss_tot);
}

// paso de gradiente descendente usando operaciones matriciales
void gradient_descent_step(LinearRegression *lr, Matrix *X, Matrix *y, Matrix *predictions)
{
    if (!lr || !X || !y || !predictions)
        return;

    int m = X->rows; // numero de muestras

    // calcular errores: error = predictions - y
    Matrix *errors = matrix_subtract(predictions, y);
    if (!errors)
        return;

    // transponer X para calcular gradientes: X^T
    Matrix *X_transpose = matrix_transpose(X);
    if (!X_transpose)
    {
        matrix_free(errors);
        return;
    }

    // calcular gradientes de los pesos: gradients = X^T * errors
    Matrix *weight_gradients = matrix_multiply(X_transpose, errors);
    if (!weight_gradients)
    {
        matrix_free(errors);
        matrix_free(X_transpose);
        return;
    }

    // actualizar pesos: weights = weights - learning_rate * (gradients / m)
    for (int i = 0; i < lr->weights->rows; i++)
        lr->weights->data[i][0] -= lr->learning_rate * (weight_gradients->data[i][0] / m);

    // calcular gradiente del bias: bias_gradient = suma(errors) / m
    // bias = termino independiente
    double bias_gradient = 0.0;
    for (int i = 0; i < errors->rows; i++)
        bias_gradient += errors->data[i][0];

    // actualizar bias
    lr->bias -= lr->learning_rate * (bias_gradient / m);

    // liberar matrices temporales
    matrix_free(errors);
    matrix_free(X_transpose);
    matrix_free(weight_gradients);
}

// liberar memoria de las metricas de regresion
void regression_metrics_free(RegressionMetrics *metrics)
{
    if (!metrics)
        return;

    if (metrics->mse_history)
        free(metrics->mse_history);
    if (metrics->cost_history)
        free(metrics->cost_history);

    free(metrics);
}

// normalizar caracteristicas usando estadisticas del conjunto de entrenamiento
void normalize_features_with_stats(Matrix *X, double *means, double *stds, int compute_stats)
{
    if (!X)
        return;

    // si compute_stats es 1, calcular estadisticas (para X_train)
    // si compute_stats es 0, usar estadisticas proporcionadas (para X_test)
    if (compute_stats)
    {
        for (int j = 0; j < X->cols; j++)
        {
            double mean = 0.0;
            for (int i = 0; i < X->rows; i++)
                mean += X->data[i][j];
            mean /= X->rows;
            means[j] = mean;
            double std = 0.0;
            for (int i = 0; i < X->rows; i++)
            {
                double diff = X->data[i][j] - mean;
                std += diff * diff;
            }
            std = sqrt(std / X->rows);

            // evitar division por cero
            if (std < 1e-8)
                std = 1.0;
            stds[j] = std;
        }
    }

    // aplicar normalizacion usando las estadisticas
    for (int j = 0; j < X->cols; j++)
    {
        for (int i = 0; i < X->rows; i++)
            X->data[i][j] = (X->data[i][j] - means[j]) / stds[j];
    }
}
