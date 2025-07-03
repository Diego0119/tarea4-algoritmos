/*
 * @file: linear-regression.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 25/06/2025
 * @description: Cabecera general de funciones auxiliares para el algoritmo de regresion lineal.
 */

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "matrix.h"
#include "csv.h"

#define LR_METHOD "gradient" // Opciones: "gradient", "normal"
// Regularizacion: "none", "ridge", "lasso"
#define LR_REGULARIZATION "ridge" // Opciones: "none", "ridge", "lasso"
#define LR_LAMBDA 0.1             // Parametro de regularizacion

typedef struct
{
    Matrix *weights;      // vector de pesos (coeficientes)
    double bias;          // Termino independiente (intercept)
    double learning_rate; // Tasa de aprendizaje
    int max_iterations;   // Número maximo de iteraciones
    double tolerance;     // Criterio de convergencia
} LinearRegression;

typedef struct
{
    double *mse_history;  // Historial del error cuadratico medio
    double *cost_history; // Historial de la funcion de costo
    int n_iterations;     // Número de iteraciones realizadas
    double final_mse;     // MSE final
    double r2_score;      // Coeficiente de determinacion r²
} RegressionMetrics;

// Funciones del algoritmo de regresion lineal
void export_results_to_csv(Matrix* weights, Matrix* predictions, Matrix* y, Matrix* X, const char* filename);
void train_linear_regression(LinearRegression* model, Matrix* X, Matrix* y, int max_iter, double learning_rate, double lambda, const char* regularization_type, double tolerance);
void exec_linear_regression_from_csv(const char *filename, double learning_rate, int max_iterations, double tolerance, const char* regularization_type, double lambda);
void exec_linear_regression(CSVData *csv_data, double learning_rate, int max_iterations, double tolerance);
double linear_regression_mse(Matrix *, Matrix *);
double linear_regression_r2_score(Matrix *, Matrix *);
void free_linear_regression(LinearRegression* model);

#endif