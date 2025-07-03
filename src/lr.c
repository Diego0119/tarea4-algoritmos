#include "lr.h"
#include "libs.h"
#include "config.h"
#include "csv.h"
#include "errors.h"

// Función para entrenar el modelo de regresión lineal usando descenso de gradiente o ecuaciones normales
void train_linear_regression(LinearRegression* model, Matrix* X, Matrix* y, int max_iter, double learning_rate, int use_normal_equations, double lambda, int regularization_type, double tolerance) {
    int m = X->rows;
    int n = X->cols;

    model->coefficients = matrix_create(n, 1);  // Inicializar los coeficientes
    model->num_features = n;

    // Inicializar coeficientes a cero
    for (int i = 0; i < n; i++) {
        model->coefficients->data[i][0] = 0.0;
    }

    if (use_normal_equations) {
        // Usar ecuaciones normales para resolver los coeficientes
        Matrix* Xt = matrix_transpose(X);
        Matrix* XtX = matrix_multiply(Xt, X);
        
        // Agregar regularización Ridge (L2)
        if (regularization_type == 1) {  // Ridge
            for (int i = 0; i < XtX->rows; i++) {
                XtX->data[i][i] += lambda;  // Agregar el término de regularización
            }
        }

        Matrix* XtX_inv = matrix_inverse(XtX);
        Matrix* XtY = matrix_multiply(Xt, y);
        model->coefficients = matrix_multiply(XtX_inv, XtY);

        matrix_free(Xt);
        matrix_free(XtX);
        matrix_free(XtX_inv);
        matrix_free(XtY);
    } else {
        // Descenso de gradiente con regularización
        for (int iter = 0; iter < max_iter; iter++) {
            Matrix* predictions = matrix_multiply(X, model->coefficients);
            Matrix* error = matrix_subtract(predictions, y);

            // Calcular el MSE
            double mse = calculate_mse(y, predictions);
            if (mse < tolerance) {
                printf("Convergencia alcanzada (MSE < tolerance). Iteración: %d\n", iter);
                break;  // Detener el entrenamiento si el error es suficientemente pequeño
            }

            for (int j = 0; j < n; j++) {
                double gradient = 0.0;
                for (int i = 0; i < m; i++) {
                    gradient += error->data[i][0] * X->data[i][j];
                }

                // Regularización Lasso (L1)
                if (regularization_type == 2) {  // Lasso
                    model->coefficients->data[j][0] -= (learning_rate / m) * (gradient + lambda * (model->coefficients->data[j][0] > 0 ? 1 : -1));
                }
                // Regularización Ridge (L2)
                else if (regularization_type == 1) {  // Ridge
                    model->coefficients->data[j][0] -= (learning_rate / m) * (gradient + lambda * model->coefficients->data[j][0]);
                } else {
                    model->coefficients->data[j][0] -= (learning_rate / m) * gradient;
                }
            }

            matrix_free(predictions);
            matrix_free(error);
        }
    }
}

// Función para predecir valores usando el modelo entrenado
double predict(LinearRegression* model, Matrix* X) {
    Matrix* predictions = matrix_multiply(X, model->coefficients);
    double result = predictions->data[0][0]; 
    matrix_free(predictions);
    return result;
}

// Función para liberar la memoria utilizada por el modelo de regresión lineal
void free_linear_regression(LinearRegression* model) {
    if (model->coefficients) {
        matrix_free(model->coefficients);
    }
    free(model);
}

void exec_linear_regression(CSVData *csv_data, double learning_rate, int max_iterations, double tolerance, int use_normal_equations, double lambda, int regularization_type) {
    LinearRegression* model = (LinearRegression*)malloc(sizeof(LinearRegression));
    if (!model) {
        memory_error(__FILE__, __LINE__, "No se pudo asignar memoria para el modelo de regresión lineal");
    }

    train_linear_regression(model, csv_data->data, csv_data->labels, max_iterations, learning_rate, use_normal_equations, lambda, regularization_type, tolerance);

    Matrix* predictions = matrix_multiply(csv_data->data, model->coefficients);
    printf("Predicción para el primer ejemplo: %f\n", predictions->data[0][0]);

    export_results_to_csv(model->coefficients, predictions, csv_data->labels, csv_data->data, "resultados_lr.csv");

    free_linear_regression(model);
    matrix_free(predictions);
}

// Función para calcular la inversa de una matriz utilizando Gauss-Jordan
Matrix* matrix_inverse(Matrix* matrix) {
    int n = matrix->rows;

    Matrix* identity = matrix_create(n, n);
    for (int i = 0; i < n; i++) {
        identity->data[i][i] = 1.0;
    }

    Matrix* copy = matrix_create(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            copy->data[i][j] = matrix->data[i][j];
        }
    }

    // Algoritmo de eliminación de Gauss-Jordan
    for (int i = 0; i < n; i++) {
        double pivot = copy->data[i][i];
        if (pivot == 0) {
            matrix_free(identity);
            matrix_free(copy);
            return NULL;  // La matriz no es invertible
        }
    }
    matrix_free(copy);
    return identity;
}

// Función para calcular el MSE
double calculate_mse(Matrix* y, Matrix* predictions) {
    double mse = 0.0;
    for (int i = 0; i < y->rows; i++) {
        mse += (y->data[i][0] - predictions->data[i][0]) * (y->data[i][0] - predictions->data[i][0]);
    }
    return mse / y->rows;
}

// Función para calcular el coeficiente de determinación R²
double calculate_r2(Matrix* y, Matrix* predictions) {
    double ss_total = 0.0;
    double ss_residual = 0.0;
    double y_mean = 0.0;

    // Calcular la media de y
    for (int i = 0; i < y->rows; i++) {
        y_mean += y->data[i][0];
    }
    y_mean /= y->rows;

    // Calcular SS_total y SS_residual
    for (int i = 0; i < y->rows; i++) {
        ss_total += (y->data[i][0] - y_mean) * (y->data[i][0] - y_mean);
        ss_residual += (y->data[i][0] - predictions->data[i][0]) * (y->data[i][0] - predictions->data[i][0]);
    }

    return 1 - (ss_residual / ss_total);
}

// Función para guardar los resultados de la regresión lineal en un archivo CSV dentro de la carpeta "stats"
void export_results_to_csv(Matrix* coefficients, Matrix* predictions, Matrix* y, Matrix* X, const char* filename) {

    char filepath[256];
    snprintf(filepath, sizeof(filepath), "stats/%s", filename); 

    FILE* file = fopen(filepath, "w");
    if (file == NULL) {
        fprintf(stderr, "No se pudo abrir el archivo para escribir: %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    // Calcular MSE y R²
    double mse = calculate_mse(y, predictions);
    double r2 = calculate_r2(y, predictions);

    // Imprimir en la terminal los resultados
    printf("=== Regresión Lineal ===\n");
    printf("Coeficiente: %f\n", coefficients->data[0][0]);
    printf("Intercepto: %f\n", coefficients->data[1][0]);
    printf("Error Cuadrático Medio: %f\n", mse);
    printf("Coeficiente R²: %f\n", r2);

    fprintf(file, "X, Y realista, Y predicción\n");

    for (int i = 0; i < X->rows; i++) {
        fprintf(file, "%f, %f, %f\n", X->data[i][0], y->data[i][0], predictions->data[i][0]);
    }

    fclose(file);
    printf("Resultados exportados a %s\n", filepath);
}