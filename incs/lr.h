#ifndef REGRESSION_H
#define REGRESSION_H

#include "matrix.h"
#include "csv.h"

// Estructura para el modelo de regresión lineal
typedef struct {
    Matrix* coefficients; // Coeficientes de la regresión
    int num_features;     // Número de características
} LinearRegression;

// Funciones para la regresión lineal
void train_linear_regression(LinearRegression* model, Matrix* X, Matrix* y, int max_iter, double learning_rate, int use_normal_equations, double lambda, int regularization_type, double tolerance);
double predict(LinearRegression* model, Matrix* X);
void free_linear_regression(LinearRegression* model);

void exec_linear_regression(CSVData *csv_data, double learning_rate, int max_iterations, double tolerance, int use_normal_equations, double lambda, int regularization_type);
double calculate_mse(Matrix* y, Matrix* predictions);
Matrix* matrix_inverse(Matrix* matrix);
void export_results_to_csv(Matrix* coefficients, Matrix* predictions, Matrix* y, Matrix* X, const char* filename);

double calculate_r2(Matrix* y, Matrix* predictions);
double calculate_mse(Matrix* y, Matrix* predictions);

#endif // REGRESSION_H
