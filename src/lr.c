/*
 * @file: lr.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 03/07/2025
 * @description: Regresión lineal simple: longitud de pétalo -> ancho de pétalo.
 */

#include "lr.h"
#include "libs.h"
#include "config.h"
#include "errors.h"

// Función para calcular el MSE
double calculate_mse(Matrix *y, Matrix *predictions)
{
    double mse = 0.0;
    for (int i = 0; i < y->rows; i++)
        mse += (y->data[i][0] - predictions->data[i][0]) * (y->data[i][0] - predictions->data[i][0]);
    return mse / y->rows;
}

double calculate_r2(Matrix *y, Matrix *predictions)
{
    double ss_total = 0.0, ss_residual = 0.0, y_mean = 0.0;
    for (int i = 0; i < y->rows; i++) y_mean += y->data[i][0];
    y_mean /= y->rows;
    for (int i = 0; i < y->rows; i++) {
        ss_total += (y->data[i][0] - y_mean) * (y->data[i][0] - y_mean);
        ss_residual += (y->data[i][0] - predictions->data[i][0]) * (y->data[i][0] - predictions->data[i][0]);
    }
    return 1 - (ss_residual / ss_total);
}

double calculate_mae(Matrix *y, Matrix *predictions)
{
    double mae = 0.0;
    for (int i = 0; i < y->rows; i++)
        mae += fabs(y->data[i][0] - predictions->data[i][0]);
    return mae / y->rows;
}

// Liberar memoria del modelo
void free_linear_regression(LinearRegression *model)
{
    if (model->weights)
        matrix_free(model->weights);
    free(model);
}

// Exportar resultados a CSV (x, y_realista, y_prediccion)
void export_results_to_csv(Matrix *predictions, Matrix *y, Matrix *X, const char *filename)
{
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "stats/%s", filename);
    FILE *file = fopen(filepath, "w");
    if (!file) return;
    fprintf(file, "x, y_realista, y_prediccion\n");
    for (int i = 0; i < X->rows; i++)
        fprintf(file, "%f, %f, %f\n", X->data[i][0], y->data[i][0], predictions->data[i][0]);
    fclose(file);
}

// Descenso de gradiente (con o sin regularización)
void train_linear_regression_gradient(LinearRegression *model, Matrix *X, Matrix *y, int max_iter, double learning_rate, double tolerance, const char *regularization, double lambda)
{
    int m = X->rows, n = X->cols;
    model->weights = matrix_create(n, 1);
    for (int i = 0; i < n; i++) model->weights->data[i][0] = 0.0;
    for (int iter = 0; iter < max_iter; iter++) {
        Matrix *predictions = matrix_multiply(X, model->weights);
        Matrix *error = matrix_subtract(predictions, y);
        double mse = calculate_mse(y, predictions);
        if (mse < tolerance) {
            matrix_free(predictions);
            matrix_free(error);
            break;
        }
        for (int j = 0; j < n; j++) {
            double gradient = 0.0;
            for (int i = 0; i < m; i++)
                gradient += error->data[i][0] * X->data[i][j];
            if (strcmp(regularization, "ridge") == 0)
                model->weights->data[j][0] -= (learning_rate / m) * (gradient + lambda * model->weights->data[j][0]);
            else if (strcmp(regularization, "lasso") == 0)
                model->weights->data[j][0] -= (learning_rate / m) * (gradient + lambda * (model->weights->data[j][0] > 0 ? 1 : -1));
            else
                model->weights->data[j][0] -= (learning_rate / m) * gradient;
        }
        matrix_free(predictions);
        matrix_free(error);
    }
}

// Suma de matrices (local)
static Matrix *matrix_add(Matrix *A, Matrix *B) {
    Matrix *C = matrix_create(A->rows, A->cols);
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            C->data[i][j] = A->data[i][j] + B->data[i][j];
    return C;
}

// Inversa de matriz (local, eliminación gaussiana)
static Matrix *matrix_inverse(Matrix *matrix) {
    if (matrix->rows != matrix->cols)
        return NULL;
    int n = matrix->rows;
    Matrix *augmented = matrix_create(n, 2 * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            augmented->data[i][j] = matrix->data[i][j];
        augmented->data[i][n + i] = 1.0;
    }
    for (int i = 0; i < n; i++) {
        double pivot = augmented->data[i][i];
        if (pivot == 0) {
            matrix_free(augmented);
            return NULL;
        }
        for (int j = 0; j < 2 * n; j++)
            augmented->data[i][j] /= pivot;
        for (int k = 0; k < n; k++)
            if (k != i) {
                double factor = augmented->data[k][i];
                for (int j = 0; j < 2 * n; j++)
                    augmented->data[k][j] -= factor * augmented->data[i][j];
            }
    }
    Matrix *inverse = matrix_create(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            inverse->data[i][j] = augmented->data[i][n + j];
    matrix_free(augmented);
    return inverse;
}

// Ecuaciones normales (con o sin Ridge)
void train_linear_regression_normal(LinearRegression *model, Matrix *X, Matrix *y, const char *regularization, double lambda)
{
    int n = X->cols;
    Matrix *Xt = matrix_transpose(X);
    Matrix *XtX = matrix_multiply(Xt, X);
    if (strcmp(regularization, "ridge") == 0 && lambda > 0) {
        Matrix *I = matrix_create(n, n);
        for (int i = 0; i < n; i++) I->data[i][i] = (i == 0) ? 0.0 : 1.0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) I->data[i][j] *= lambda;
        Matrix *XtX_reg = matrix_add(XtX, I);
        Matrix *XtX_inv = matrix_inverse(XtX_reg);
        Matrix *Xty = matrix_multiply(Xt, y);
        model->weights = matrix_multiply(XtX_inv, Xty);
        matrix_free(I); matrix_free(XtX_reg); matrix_free(XtX_inv); matrix_free(Xty);
    } else {
        Matrix *XtX_inv = matrix_inverse(XtX);
        Matrix *Xty = matrix_multiply(Xt, y);
        model->weights = matrix_multiply(XtX_inv, Xty);
        matrix_free(XtX_inv); matrix_free(Xty);
    }
    matrix_free(Xt); matrix_free(XtX);
}

void exec_linear_regression_from_csv(const char *filename, double learning_rate, int max_iterations, double tolerance, const char *method, const char *regularization, double lambda)
{
    CSVData *csv_data = load_csv_data_no_norm(filename, 1, ',');
    if (!csv_data) return;
    printf("Datos cargados correctamente desde: %s.\n\n", filename);
    printf("Dimensiones de los datos: %d filas x %d columnas (petal_length, petal_width)\n\n", csv_data->data->rows, 2);
    printf("Primeras 5 muestras:\n");
    int mostrar = csv_data->data->rows < 5 ? csv_data->data->rows : 5;
    for (int i = 0; i < mostrar; i++) {
        printf("Muestra %d: [%g] -> %g\n", i, csv_data->data->data[i][0], csv_data->labels->data[i][0]);
    }
    printf("\n");
    Matrix *Xb = add_bias_column(csv_data->data);
    LinearRegression *model = malloc(sizeof(LinearRegression));
    if (!model) return;
    if (strcmp(method, "normal") == 0)
        train_linear_regression_normal(model, Xb, csv_data->labels, regularization, lambda);
    else
        train_linear_regression_gradient(model, Xb, csv_data->labels, max_iterations, learning_rate, tolerance, regularization, lambda);
    Matrix *predictions = matrix_multiply(Xb, model->weights);
    double r2 = calculate_r2(csv_data->labels, predictions);
    double mse = calculate_mse(csv_data->labels, predictions);
    double mae = calculate_mae(csv_data->labels, predictions);
    printf("MSE: %f\n", mse);
    printf("MAE: %f\n", mae);
    printf("R2: %f\n", r2);
    export_results_to_csv(predictions, csv_data->labels, csv_data->data, "resultados_lr.csv");
    free_linear_regression(model);
    matrix_free(predictions);
    matrix_free(Xb);
    csv_free(csv_data);
}

// Cargar CSV sin normalizar (solo longitud y ancho de pétalo)
CSVData *load_csv_data_no_norm(const char *filename, int header, char delimiter) {
    FILE *file = fopen(filename, "r");
    if (!file) return NULL;
    int rows = 0, cols = 0;
    csv_dimensions(filename, header, delimiter, &rows, &cols);
    Matrix *data = matrix_create(rows, 1);
    Matrix *labels = matrix_create(rows, 1);
    char line[1024];
    int row = 0;
    if (header && !fgets(line, sizeof(line), file)) { fclose(file); return NULL; }
    while (fgets(line, sizeof(line), file)) {
        char *token;
        int col = 0;
        double petal_length = 0.0, petal_width = 0.0;
        token = strtok(line, ",\n");
        while (token && col < cols) {
            if (col == 2) petal_length = atof(token);
            if (col == 3) petal_width = atof(token);
            token = strtok(NULL, ",\n");
            col++;
        }
        data->data[row][0] = petal_length;
        labels->data[row][0] = petal_width;
        row++;
    }
    fclose(file);
    CSVData *csv = malloc(sizeof(CSVData));
    csv->data = data;
    csv->labels = labels;
    csv->header = NULL;
    csv->has_header = header;
    csv->label_col = 3;
    return csv;
}

// Añadir columna de unos a X para el bias
Matrix *add_bias_column(Matrix *X) {
    Matrix *Xb = matrix_create(X->rows, X->cols + 1);
    for (int i = 0; i < X->rows; i++) {
        Xb->data[i][0] = 1.0;
        for (int j = 0; j < X->cols; j++)
            Xb->data[i][j + 1] = X->data[i][j];
    }
    return Xb;
}
