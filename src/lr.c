#include "lr.h"
#include "libs.h"
#include "config.h"
#include "csv.h"
#include "errors.h"

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

// Función para liberar la memoria utilizada por el modelo de regresión lineal
void free_linear_regression(LinearRegression* model) {
    if (model->weights) {
        matrix_free(model->weights);
    }
    free(model);
}

// Función para guardar los resultados de la regresión lineal en un archivo CSV
void export_results_to_csv(Matrix* weights, Matrix* predictions, Matrix* y, Matrix* X, const char* filename) {
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
    printf("Pesos:\n");
    for (int i = 0; i < weights->rows; i++) {
        printf("%f\n", weights->data[i][0]);
    }
    printf("Error Cuadrático Medio: %f\n", mse);
    printf("Coeficiente R²: %f\n", r2);

    // Escribir en el archivo CSV
    fprintf(file, "X, Y realista, Y predicción\n");
    for (int i = 0; i < X->rows; i++) {
        fprintf(file, "%f, %f, %f\n", X->data[i][0], y->data[i][0], predictions->data[i][0]);
    }

    fclose(file);
    printf("Resultados exportados a %s\n", filepath);
}

// Función para entrenar el modelo de regresión lineal usando descenso de gradiente
void train_linear_regression(LinearRegression* model, Matrix* X, Matrix* y, int max_iter, double learning_rate, double lambda, const char* regularization_type, double tolerance) {
    int m = X->rows;
    int n = X->cols;

    model->weights = matrix_create(n, 1);  // Inicializar los pesos
    model->bias = 0.0;

    // Inicializar pesos a cero
    for (int i = 0; i < n; i++) {
        model->weights->data[i][0] = 0.0;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        Matrix* predictions = matrix_multiply(X, model->weights);
        for (int i = 0; i < predictions->rows; i++) {
            predictions->data[i][0] += model->bias;
        }

        Matrix* error = matrix_subtract(predictions, y);

        // Calcular el MSE
        double mse = calculate_mse(y, predictions);
        if (mse < tolerance) {
            printf("Convergencia alcanzada (MSE < tolerance). Iteración: %d\n", iter);
            matrix_free(predictions);
            matrix_free(error);
            break;
        }

        for (int j = 0; j < n; j++) {
            double gradient = 0.0;
            for (int i = 0; i < m; i++) {
                gradient += error->data[i][0] * X->data[i][j];
            }

            if (strcmp(regularization_type, "lasso") == 0) {
                model->weights->data[j][0] -= (learning_rate / m) * (gradient + lambda * (model->weights->data[j][0] > 0 ? 1 : -1));
            } else if (strcmp(regularization_type, "ridge") == 0) {
                model->weights->data[j][0] -= (learning_rate / m) * (gradient + lambda * model->weights->data[j][0]);
            } else {
                model->weights->data[j][0] -= (learning_rate / m) * gradient;
            }
        }

        double bias_gradient = 0.0;
        for (int i = 0; i < m; i++) {
            bias_gradient += error->data[i][0];
        }
        model->bias -= (learning_rate / m) * bias_gradient;

        matrix_free(predictions);
        matrix_free(error);
    }
}

// Función para ejecutar la regresión lineal desde un archivo CSV
void exec_linear_regression_from_csv(const char *filename, double learning_rate, int max_iterations, double tolerance, const char* regularization_type, double lambda) {
    // Cargar datos desde el archivo CSV
    CSVData *csv_data = load_csv_data(filename, 1, -1, ',');
    if (!csv_data) {
        fprintf(stderr, "Error al cargar los datos desde %s\n", filename);
        return;
    }

    // Crear modelo de regresión lineal
    LinearRegression *model = (LinearRegression *)malloc(sizeof(LinearRegression));
    if (!model) {
        memory_error(__FILE__, __LINE__, "No se pudo asignar memoria para el modelo de regresión lineal");
    }

    // Entrenar el modelo
    train_linear_regression(model, csv_data->data, csv_data->labels, max_iterations, learning_rate, lambda, regularization_type, tolerance);

    // Generar predicciones
    Matrix *predictions = matrix_multiply(csv_data->data, model->weights);
    for (int i = 0; i < predictions->rows; i++) {
        predictions->data[i][0] += model->bias;
    }

    // Calcular R²
    double r2 = calculate_r2(csv_data->labels, predictions);
    printf("Coeficiente R²: %f\n", r2);

    // Exportar resultados
    export_results_to_csv(model->weights, predictions, csv_data->labels, csv_data->data, "resultados_lr.csv");

    // Liberar memoria
    free_linear_regression(model);
    matrix_free(predictions);
    csv_free(csv_data);
}

void exec_linear_regression(CSVData *csv_data, double learning_rate, int max_iterations, double tolerance) {
    // Crear modelo de regresión lineal
    LinearRegression *model = (LinearRegression *)malloc(sizeof(LinearRegression));
    if (!model) {
        memory_error(__FILE__, __LINE__, "No se pudo asignar memoria para el modelo de regresión lineal");
    }

    // Entrenar el modelo
    train_linear_regression(model, csv_data->data, csv_data->labels, max_iterations, learning_rate, LR_LAMBDA, LR_REGULARIZATION, tolerance);

    // Generar predicciones
    Matrix *predictions = matrix_multiply(csv_data->data, model->weights);
    for (int i = 0; i < predictions->rows; i++) {
        predictions->data[i][0] += model->bias;
    }

    // Calcular R²
    double r2 = calculate_r2(csv_data->labels, predictions);
    printf("Coeficiente R²: %f\n", r2);

    // Exportar resultados
    export_results_to_csv(model->weights, predictions, csv_data->labels, csv_data->data, "resultados_lr.csv");

    // Liberar memoria
    free_linear_regression(model);
    matrix_free(predictions);
}
