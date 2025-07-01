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

// Seleccion de metodo y regularizacion
void exec_linear_regression(CSVData *csv_data, double learning_rate, int max_iterations, double tolerance)
{
    fprintf(stdout, CYAN_COLOR "Regresion Lineal (LR)\n\n" RESET_COLOR);

    Matrix *X_train, *y_train, *X_valid, *y_valid, *X_test, *y_test;

    double test_ratio = 0.2;
    double valid_ratio = 0.2;

    if (!train_valid_test_split(csv_data->data, csv_data->labels, valid_ratio, test_ratio, &X_train, &y_train, &X_valid, &y_valid, &X_test, &y_test))
        train_valid_test_split_error(__FILE__, __LINE__);

    printf("Conjunto de entrenamiento: %d muestras x %d caracteristicas\n", X_train->rows, X_train->cols);
    printf("Conjunto de validacion: %d muestras x %d caracteristicas\n", X_valid->rows, X_valid->cols);
    printf("Conjunto de prueba: %d muestras x %d caracteristicas\n", X_test->rows, X_test->cols);
    printf("\n");
    int n_samples_to_show = X_train->rows < 5 ? X_train->rows : 5;
    printf(YELLOW_COLOR "Primeras %d muestras del conjunto de entrenamiento:\n" RESET_COLOR, n_samples_to_show);
    printf("\n");   
    if (csv_data->has_header && csv_data->header) {
        printf("  [");
        for (int j = 0; j < X_train->cols; j++) {
            printf("%s", csv_data->header[j]);
            if (j < X_train->cols - 1) printf(", ");
        }
        printf("] -> Label\n");
    }
    for (int i = 0; i < n_samples_to_show; i++) {
        printf("Muestra %-2d:[", i + 1);
        for (int j = 0; j < X_train->cols; j++) {
            printf("%7.4f", X_train->data[i][j]);
            if (j < X_train->cols - 1) printf(", ");
        }
        printf("] -> %.4f\n", y_train->data[i][0]);
    }
    printf("\n");
    double *means = (double *)calloc(X_train->cols, sizeof(double));
    double *stds = (double *)calloc(X_train->cols, sizeof(double));
    printf("\n");
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

    for (int j = 0; j < X_train->cols; j++)
    {
        for (int i = 0; i < X_train->rows; i++)
            means[j] += X_train->data[i][j];

        means[j] /= X_train->rows;

        for (int i = 0; i < X_train->rows; i++)
            stds[j] += (X_train->data[i][j] - means[j]) * (X_train->data[i][j] - means[j]);

        stds[j] = sqrt(stds[j] / X_train->rows);

        if (stds[j] < 1e-8)
            stds[j] = 1.0;
    }

    normalize_features_with_stats(X_train, means, stds, 0);
    normalize_features_with_stats(X_test, means, stds, 0);

    LinearRegression *lr = linear_regression_create(X_train->cols, learning_rate, max_iterations, tolerance);
    if (!lr)
        create_linear_regression_error(__FILE__, __LINE__, X_train, y_train, X_test, y_test, lr);

    RegressionMetrics *metrics = NULL;

    if (strcmp(LR_METHOD, "normal") == 0)
    {
        int ridge = (strcmp(LR_REGULARIZATION, "ridge") == 0);
        linear_regression_fit_normal(lr, X_train, y_train, ridge, LR_LAMBDA);
    }
    else
        linear_regression_fit_regularized(lr, X_train, y_train, &metrics, LR_REGULARIZATION, LR_LAMBDA); 

    Matrix *y_pred = linear_regression_predict(lr, X_test);
    if (!y_pred)
        predict_linear_regression_error(__FILE__, __LINE__, X_train, y_train, X_test, y_test, lr);

    // Exportar resultados para graficar
    export_results_lr_to_csv(X_test, y_test, y_pred, "stats/resultados_lr.csv");

    double test_mse = linear_regression_mse(y_test, y_pred);
    double test_r2 = linear_regression_r2_score(y_test, y_pred);
    double test_mae = linear_regression_mae(y_test, y_pred);

    printf(CYAN_COLOR "Parámetros del modelo entrenado:\n" RESET_COLOR);
    printf("  Pesos (coeficientes): ");
    for (int i = 0; i < lr->weights->rows; i++) {
        printf("%.4f", lr->weights->data[i][0]);
        if (i < lr->weights->rows - 1) printf(", ");
    }
    printf("\n");
    printf("  Término independiente (intercepto): %.4f\n", lr->bias);

    printf(CYAN_COLOR "\nMétricas de evaluación en el conjunto de prueba:\n" RESET_COLOR);
    printf("  MSE : %.4f\n", test_mse);
    printf("  MAE : %.4f\n", test_mae);
    printf("  R2  : %.4f\n\n", test_r2);

    compare_batch_vs_minibatch(X_train, y_train, X_test, y_test, learning_rate, max_iterations, tolerance, LR_REGULARIZATION, LR_LAMBDA); // Comparacion de metodos

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

// Aplica el algoritmo de regresion lineal al conjunto de datos
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

    // Inicializar pesos con inicializacion xavier/glorot para mejor convergencia
    double limit = sqrt(6.0 / (n_features + 1)); // Xavier initialization

    // Xavier o glorot initialization = inicializar peso de una red
    for (int i = 0; i < n_features; i++)
        lr->weights->data[i][0] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * limit;

    lr->bias = 0.0;
    lr->learning_rate = learning_rate;
    lr->max_iterations = max_iterations;
    lr->tolerance = tolerance;

    return lr;
}

// Entrenar el modelo de regresion lineal usando gradiente descendente
void linear_regression_fit(LinearRegression *lr, Matrix *X, Matrix *y, RegressionMetrics **metrics)
{
    if (!lr || !X || !y)
        return;

    // Crear estructura para metricas
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

    // Gradiente descendente
    for (iteration = 0; iteration < lr->max_iterations; iteration++)
    {
        // Calcular predicciones usando multiplicacion de matrices: y_pred = X * weights
        Matrix *predictions = matrix_multiply(X, lr->weights);
        if (!predictions)
        {
            regression_metrics_free(*metrics);
            *metrics = NULL;
            return;
        }

        // Agregar bias a todas las predicciones
        for (int i = 0; i < predictions->rows; i++)
            predictions->data[i][0] += lr->bias;

        // Calcular mse
        double mse = linear_regression_mse(y, predictions);
        (*metrics)->mse_history[iteration] = mse;
        (*metrics)->cost_history[iteration] = mse / 2.0; // funcion de costo

        // Verificar convergencia
        if (iteration > 0 && fabs(prev_mse - mse) < lr->tolerance)
        {
            matrix_free(predictions);
            break;
        }

        // Actualizar pesos y bias usando gradiente descendente con operaciones matriciales
        gradient_descent_step(lr, X, y, predictions);

        prev_mse = mse;

        matrix_free(predictions);
    }

    // Calcular metricas finales
    Matrix *final_predictions = matrix_multiply(X, lr->weights);
    if (final_predictions)
    {
        for (int i = 0; i < final_predictions->rows; i++) // Agregar bias
            final_predictions->data[i][0] += lr->bias;

        (*metrics)->n_iterations = iteration + 1;
        (*metrics)->final_mse = linear_regression_mse(y, final_predictions);
        (*metrics)->r2_score = linear_regression_r2_score(y, final_predictions);

        matrix_free(final_predictions);
    }
}

// Realizar predicciones con el modelo entrenado
Matrix *linear_regression_predict(LinearRegression *lr, Matrix *X)
{
    if (!lr || !X)
        return NULL;

    // Usar multiplicacion de matrices: y_pred = X * weights
    Matrix *predictions = matrix_multiply(X, lr->weights);
    if (!predictions)
        return NULL;

    // Agregar bias a todas las predicciones
    for (int i = 0; i < predictions->rows; i++)
        predictions->data[i][0] += lr->bias;

    return predictions;
}

// Liberar memoria del modelo de regresion lineal
void linear_regression_free(LinearRegression *lr)
{
    if (!lr)
        return;

    if (lr->weights)
        matrix_free(lr->weights);

    free(lr);
}

// Calcular el error cuadratico medio (mse)
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

// Calcular el coeficiente de determinacion r²
double linear_regression_r2_score(Matrix *y_true, Matrix *y_pred)
{
    if (!y_true || !y_pred || y_true->rows != y_pred->rows)
        return -1.0;

    // Calcular la media de y_true
    double mean_y = 0.0;
    for (int i = 0; i < y_true->rows; i++)
        mean_y += y_true->data[i][0];
    mean_y /= y_true->rows;

    // Calcular suma de cuadrados total y residual
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
        return 1.0; // Caso perfecto donde todas las y son iguales

    return 1.0 - (ss_res / ss_tot);
}

// Calculo de MAE
double linear_regression_mae(Matrix *y_true, Matrix *y_pred)
{
    if (!y_true || !y_pred || y_true->rows != y_pred->rows)
        return -1.0;

    double sum = 0.0;

    for (int i = 0; i < y_true->rows; i++)
        sum += fabs(y_true->data[i][0] - y_pred->data[i][0]);

    return sum / y_true->rows;
}

// Ecuaciones normales (con y sin Ridge)
Matrix *matrix_identity(int n)
{
    Matrix *I = matrix_create(n, n);

    for (int i = 0; i < n; i++)
        I->data[i][i] = 1.0;

    return I;
}

// Inversion de matrices pequeñas por Gauss-Jordan (solo para ecuaciones normales)
int matrix_inverse(Matrix *A, Matrix *A_inv)
{
    int n = A->rows;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A_inv->data[i][j] = (i == j) ? 1.0 : 0.0;

    for (int i = 0; i < n; i++)
    {
        double pivot = A->data[i][i];

        if (fabs(pivot) < 1e-12) // Verificar si el pivote es cero
            return 0;

        for (int j = 0; j < n; j++)
        {
            A->data[i][j] /= pivot;
            A_inv->data[i][j] /= pivot;
        }

        for (int k = 0; k < n; k++)
        {
            if (k == i) // No hacer operaciones en la fila del pivote
                continue;

            double factor = A->data[k][i];

            for (int j = 0; j < n; j++) // Restar la fila del pivote multiplicada por el factor
            {
                A->data[k][j] -= factor * A->data[i][j];
                A_inv->data[k][j] -= factor * A_inv->data[i][j];
            }
        }
    }

    return 1;
}

// Gradiente descendente con regularizacion
void linear_regression_fit_normal(LinearRegression *lr, Matrix *X, Matrix *y, int ridge, double lambda)
{
    int n = X->cols;

    Matrix *X_T = matrix_transpose(X);
    Matrix *XTX = matrix_multiply(X_T, X);

    if (ridge)
    {
        Matrix *I = matrix_identity(n);

        for (int i = 0; i < n; i++)
            I->data[i][i] *= lambda;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                XTX->data[i][j] += I->data[i][j];

        matrix_free(I);
    }

    Matrix *XTy = matrix_multiply(X_T, y);
    Matrix *XTX_inv = matrix_create(n, n);
    Matrix *XTX_copy = matrix_create(n, n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            XTX_copy->data[i][j] = XTX->data[i][j];

    if (!matrix_inverse(XTX_copy, XTX_inv))
    {
        fprintf(stderr, "No se pudo invertir la matriz para ecuaciones normales.\n");

        matrix_free(X_T);
        matrix_free(XTX);
        matrix_free(XTy);
        matrix_free(XTX_inv);
        matrix_free(XTX_copy);

        return;
    }

    Matrix *w = matrix_multiply(XTX_inv, XTy);

    for (int i = 0; i < n; i++)
        lr->weights->data[i][0] = w->data[i][0];

    lr->bias = 0.0; // El bias se asume absorbido en X si se agrega columna de 1s

    matrix_free(X_T);
    matrix_free(XTX);
    matrix_free(XTy);
    matrix_free(XTX_inv);
    matrix_free(XTX_copy);
    matrix_free(w);
}

// Gradiente descendente con regularizacion
void linear_regression_fit_regularized(LinearRegression *lr, Matrix *X, Matrix *y, RegressionMetrics **metrics, const char *regularization, double lambda)
{
    if (!lr || !X || !y)
        return;

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

    for (iteration = 0; iteration < lr->max_iterations; iteration++)
    {
        Matrix *predictions = matrix_multiply(X, lr->weights);
        if (!predictions)
        {
            regression_metrics_free(*metrics);

            *metrics = NULL;

            return;
        }

        for (int i = 0; i < predictions->rows; i++)
            predictions->data[i][0] += lr->bias;

        double mse = linear_regression_mse(y, predictions);

        (*metrics)->mse_history[iteration] = mse;
        (*metrics)->cost_history[iteration] = mse / 2.0;

        if (iteration > 0 && fabs(prev_mse - mse) < lr->tolerance)
        {
            matrix_free(predictions);
            break;
        }

        int m = X->rows;

        Matrix *errors = matrix_subtract(predictions, y);
        Matrix *X_transpose = matrix_transpose(X);
        Matrix *weight_gradients = matrix_multiply(X_transpose, errors);

        for (int i = 0; i < lr->weights->rows; i++)
        {
            double reg_term = 0.0;

            if (strcmp(regularization, "ridge") == 0)
                reg_term = lambda * lr->weights->data[i][0];
            else if (strcmp(regularization, "lasso") == 0)
                reg_term = lambda * (lr->weights->data[i][0] > 0 ? 1 : -1);

            lr->weights->data[i][0] -= lr->learning_rate * ((weight_gradients->data[i][0] / m) + reg_term);
        }

        double bias_gradient = 0.0;

        for (int i = 0; i < errors->rows; i++)
            bias_gradient += errors->data[i][0];

        lr->bias -= lr->learning_rate * (bias_gradient / m);
        prev_mse = mse;

        matrix_free(errors);
        matrix_free(X_transpose);
        matrix_free(weight_gradients);
        matrix_free(predictions);
    }

    Matrix *final_predictions = matrix_multiply(X, lr->weights);
    if (final_predictions)
    {
        for (int i = 0; i < final_predictions->rows; i++)
            final_predictions->data[i][0] += lr->bias;

        (*metrics)->n_iterations = iteration + 1;
        (*metrics)->final_mse = linear_regression_mse(y, final_predictions);
        (*metrics)->r2_score = linear_regression_r2_score(y, final_predictions);

        matrix_free(final_predictions);
    }
}

// Mini-batch Gradient Descent
void linear_regression_fit_minibatch(LinearRegression *lr, Matrix *X, Matrix *y, RegressionMetrics **metrics, int batch_size, const char *regularization, double lambda)
{
    if (!lr || !X || !y)
        return;

    int m = X->rows;
    int n_batches = (m + batch_size - 1) / batch_size;

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

    for (iteration = 0; iteration < lr->max_iterations; iteration++)
    {
        for (int b = 0; b < n_batches; b++)
        {
            int start = b * batch_size;
            int end = (start + batch_size < m) ? (start + batch_size) : m;
            int cur_batch = end - start;

            Matrix *X_batch = matrix_create(cur_batch, X->cols);
            Matrix *y_batch = matrix_create(cur_batch, 1);

            for (int i = 0; i < cur_batch; i++)
            {
                for (int j = 0; j < X->cols; j++)
                    X_batch->data[i][j] = X->data[start + i][j];

                y_batch->data[i][0] = y->data[start + i][0];
            }

            Matrix *predictions = matrix_multiply(X_batch, lr->weights);

            for (int i = 0; i < predictions->rows; i++)
                predictions->data[i][0] += lr->bias;

            int mb = X_batch->rows;

            Matrix *errors = matrix_subtract(predictions, y_batch);
            Matrix *X_transpose = matrix_transpose(X_batch);
            Matrix *weight_gradients = matrix_multiply(X_transpose, errors);

            for (int i = 0; i < lr->weights->rows; i++)
            {
                double reg_term = 0.0;

                if (strcmp(regularization, "ridge") == 0)
                    reg_term = lambda * lr->weights->data[i][0];
                else if (strcmp(regularization, "lasso") == 0)
                    reg_term = lambda * (lr->weights->data[i][0] > 0 ? 1 : -1);

                lr->weights->data[i][0] -= lr->learning_rate * ((weight_gradients->data[i][0] / mb) + reg_term);
            }

            double bias_gradient = 0.0;

            for (int i = 0; i < errors->rows; i++)
                bias_gradient += errors->data[i][0];

            lr->bias -= lr->learning_rate * (bias_gradient / mb);

            matrix_free(errors);
            matrix_free(X_transpose);
            matrix_free(weight_gradients);
            matrix_free(predictions);
            matrix_free(X_batch);
            matrix_free(y_batch);
        }

        Matrix *full_pred = matrix_multiply(X, lr->weights);

        for (int i = 0; i < full_pred->rows; i++)
            full_pred->data[i][0] += lr->bias;

        double mse = linear_regression_mse(y, full_pred);

        (*metrics)->mse_history[iteration] = mse;
        (*metrics)->cost_history[iteration] = mse / 2.0;

        if (iteration > 0 && fabs(prev_mse - mse) < lr->tolerance)
        {
            matrix_free(full_pred);
            break;
        }

        prev_mse = mse;
        matrix_free(full_pred);
    }

    Matrix *final_predictions = matrix_multiply(X, lr->weights);

    if (final_predictions)
    {
        for (int i = 0; i < final_predictions->rows; i++)
            final_predictions->data[i][0] += lr->bias;

        (*metrics)->n_iterations = iteration + 1;
        (*metrics)->final_mse = linear_regression_mse(y, final_predictions);
        (*metrics)->r2_score = linear_regression_r2_score(y, final_predictions);

        matrix_free(final_predictions);
    }
}

// Comparación de métodos de entrenamiento: Batch vs Mini-batch
void compare_batch_vs_minibatch(Matrix *X_train, Matrix *y_train, Matrix *X_test, Matrix *y_test, double learning_rate, int max_iterations, double tolerance, const char *regularization, double lambda)
{
    RegressionMetrics *metrics_batch = NULL;
    RegressionMetrics *metrics_minibatch = NULL;

    int batch_size = 16; // Tamaño de mini-batch

    // Batch clasico
    LinearRegression *lr_batch = linear_regression_create(X_train->cols, learning_rate, max_iterations, tolerance);

    clock_t t1 = clock();
    linear_regression_fit_regularized(lr_batch, X_train, y_train, &metrics_batch, regularization, lambda);
    clock_t t2 = clock();

    Matrix *y_pred_batch = linear_regression_predict(lr_batch, X_test);

    double mse_batch = linear_regression_mse(y_test, y_pred_batch);
    double mae_batch = linear_regression_mae(y_test, y_pred_batch);
    double r2_batch = linear_regression_r2_score(y_test, y_pred_batch);
    double time_batch = (double)(t2 - t1) / CLOCKS_PER_SEC;

    // Mini-batch
    LinearRegression *lr_mb = linear_regression_create(X_train->cols, learning_rate, max_iterations, tolerance);

    t1 = clock();
    linear_regression_fit_minibatch(lr_mb, X_train, y_train, &metrics_minibatch, batch_size, regularization, lambda);
    t2 = clock();

    Matrix *y_pred_mb = linear_regression_predict(lr_mb, X_test);

    double mse_mb = linear_regression_mse(y_test, y_pred_mb);
    double mae_mb = linear_regression_mae(y_test, y_pred_mb);
    double r2_mb = linear_regression_r2_score(y_test, y_pred_mb);
    double time_mb = (double)(t2 - t1) / CLOCKS_PER_SEC;

    fprintf(stdout, CYAN_COLOR "Comparación de metodos Batch vs Mini Batch:\n" RESET_COLOR);
    fprintf(stdout, "\n%-10s | %-12s %-12s %-12s %-12s\n", "Metodo", "MSE", "MAE", "R2", "Tiempo(s)");
    fprintf(stdout, "%-10s | ", "Batch");
    fprintf(stdout, GREEN_COLOR "%-12.4f %-12.4f %-12.4f %-12.4f\n" RESET_COLOR, mse_batch, mae_batch, r2_batch, time_batch);
    fprintf(stdout, "%-10s | ", "Mini-batch");
    fprintf(stdout, RED_COLOR "%-12.4f %-12.4f %-12.4f %-12.4f\n\n" RESET_COLOR, mse_mb, mae_mb, r2_mb, time_mb);

    // Exportar resultados de Batch y Mini-batch a CSV
    export_results_lr_to_csv(X_test, y_test, y_pred_batch, "stats/resultados_lr_batch.csv");
    export_results_lr_to_csv(X_test, y_test, y_pred_mb, "stats/resultados_lr_minibatch.csv");

    matrix_free(y_pred_batch);
    matrix_free(y_pred_mb);
    linear_regression_free(lr_batch);
    linear_regression_free(lr_mb);
    regression_metrics_free(metrics_batch);
    regression_metrics_free(metrics_minibatch);
}

// Liberar memoria de las metricas de regresion
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

// Normalizar caracteristicas usando estadisticas del conjunto de entrenamiento
void normalize_features_with_stats(Matrix *X, double *means, double *stds, int compute_stats)
{
    if (!X)
        return;

    // Si compute_stats es 1, calcular estadisticas (para X_train)
    // Si compute_stats es 0, usar estadisticas proporcionadas (para X_test)
    if (compute_stats)
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

            // Evitar division por cero
            if (std < 1e-8)
                std = 1.0;

            stds[j] = std;
        }

    // Aplicar normalizacion usando las estadisticas
    for (int j = 0; j < X->cols; j++)
        for (int i = 0; i < X->rows; i++)
            X->data[i][j] = (X->data[i][j] - means[j]) / stds[j];
}

// Paso de gradiente descendente
void gradient_descent_step(LinearRegression *lr, Matrix *X, Matrix *y, Matrix *predictions)
{
    if (!lr || !X || !y || !predictions)
        return;

    int m = X->rows;

    Matrix *errors = matrix_subtract(predictions, y);
    if (!errors)
        return;

    Matrix *X_transpose = matrix_transpose(X);
    if (!X_transpose)
    {
        matrix_free(errors);
        return;
    }

    Matrix *weight_gradients = matrix_multiply(X_transpose, errors);
    if (!weight_gradients)
    {
        matrix_free(errors);
        matrix_free(X_transpose);
        return;
    }

    for (int i = 0; i < lr->weights->rows; i++)
        lr->weights->data[i][0] -= lr->learning_rate * (weight_gradients->data[i][0] / m);

    double bias_gradient = 0.0;

    for (int i = 0; i < errors->rows; i++)
        bias_gradient += errors->data[i][0];

    lr->bias -= lr->learning_rate * (bias_gradient / m);

    matrix_free(errors);
    matrix_free(X_transpose);
    matrix_free(weight_gradients);
}
