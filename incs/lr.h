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

#define LR_METHOD "gradient"
#define LR_REGULARIZATION "ridge"
#define LR_LAMBDA 0.01

typedef struct
{
    Matrix *weights;      // vector de pesos (coeficientes)
    double bias;          // Termino independiente (intercept)
    double learning_rate; // Tasa de aprendizaje
    int max_iterations;   // Numero maximo de iteraciones
    double tolerance;     // Criterio de convergencia
} LinearRegression;

// Funciones del algoritmo de regresión lineal
void export_results_to_csv(Matrix *, Matrix *, Matrix *, const char *);
void train_linear_regression(LinearRegression *, Matrix *, Matrix *, int, double, double);
void exec_linear_regression_from_csv(const char *, double, int, double, const char *, const char *, double);
void free_linear_regression(LinearRegression *);
CSVData *load_csv_data_no_norm(const char *, int, char);
Matrix *add_bias_column(Matrix *);

// Métricas
double calculate_mse(Matrix *, Matrix *);
double calculate_r2(Matrix *, Matrix *);
double calculate_mae(Matrix *, Matrix *);

// Entrenamiento
void train_linear_regression_gradient(LinearRegression *, Matrix *, Matrix *, int, double, double, const char *, double);
void train_linear_regression_normal(LinearRegression *, Matrix *, Matrix *, const char *, double);

#endif