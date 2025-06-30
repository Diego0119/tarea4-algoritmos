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
    double *mse_history;  // Historial del error cuadrático medio
    double *cost_history; // Historial de la función de costo
    int n_iterations;     // Número de iteraciones realizadas
    double final_mse;     // MSE final
    double r2_score;      // Coeficiente de determinación r²
} RegressionMetrics;

// Funciones del algoritmo de regresión lineal
void exec_linear_regression(CSVData *, double, int, double);
LinearRegression *linear_regression_create(int, double, int, double);
void linear_regression_fit(LinearRegression *, Matrix *, Matrix *, RegressionMetrics **);
Matrix *linear_regression_predict(LinearRegression *, Matrix *);
double linear_regression_mse(Matrix *, Matrix *);
double linear_regression_r2_score(Matrix *, Matrix *);
void linear_regression_free(LinearRegression *);

// Funciones auxiliares para la regresión lineal
void gradient_descent_step(LinearRegression *, Matrix *, Matrix *, Matrix *);
void regression_metrics_free(RegressionMetrics *);
void normalize_features_with_stats(Matrix *, double *, double *, int);

#endif
