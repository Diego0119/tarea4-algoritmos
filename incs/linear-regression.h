/*
 * @file: linear-regression.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 25/06/2025
 * @description: cabecera general de funciones auxiliares para el algoritmo de regresion lineal.
 */

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "matrix.h"
#include "csv.h"

typedef struct
{
    Matrix *weights;    // vector de pesos (coeficientes)
    double bias;        // termino independiente (intercept)
    double learning_rate; // tasa de aprendizaje
    int max_iterations; // numero maximo de iteraciones
    double tolerance;   // criterio de convergencia
} LinearRegression;

typedef struct
{
    double *mse_history;    // historial del error cuadratico medio
    double *cost_history;   // historial de la funcion de costo
    int n_iterations;       // numero de iteraciones realizadas
    double final_mse;       // mse final
    double r2_score;        // coeficiente de determinacion r²
} RegressionMetrics;

// funciones del algoritmo de regresion lineal
void exec_linear_regression(CSVData *, double, int, double);
LinearRegression *linear_regression_create(int, double, int, double);
void linear_regression_fit(LinearRegression *, Matrix *, Matrix *, RegressionMetrics **);
Matrix *linear_regression_predict(LinearRegression *, Matrix *);
double linear_regression_mse(Matrix *, Matrix *);
double linear_regression_r2_score(Matrix *, Matrix *);
void linear_regression_free(LinearRegression *);

// funciones auxiliares para la regresion lineal
void gradient_descent_step(LinearRegression *, Matrix *, Matrix *, Matrix *);
void regression_metrics_free(RegressionMetrics *);
void normalize_features_with_stats(Matrix *, double *, double *, int);

#endif
