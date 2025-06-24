/*
 * @file: matrix.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 21/06/2025
 * @description: Archivo para manejar operaciones con matrices en el proyecto de C.
 */

#include "libs.h"
#include "matrix.h"
#include "errors.h"

// Función para crear una nueva matriz
Matrix *matrix_create(int rows, int cols)
{
    Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
    if (!matrix)
        matrix_struct_error(__FILE__, __LINE__, NULL);

    matrix->rows = rows;
    matrix->cols = cols;

    matrix->data = (double **)malloc(rows * sizeof(double *)); // Asignar memoria para filas
    if (!matrix->data)
        matrix_struct_error(__FILE__, __LINE__, matrix);

    for (int i = 0; i < rows; i++) // Asignar memoria para columnas
    {
        matrix->data[i] = (double *)calloc(cols, sizeof(double));

        if (!matrix->data[i]) // Verificar si la asignación fue exitosa
        {
            for (int j = 0; j < i; j++)
                free(matrix->data[j]);

            free(matrix->data);

            matrix_struct_error(__FILE__, __LINE__, matrix);
        }
    }

    return matrix;
}

// Función para liberar la memoria de una matriz
void matrix_free(Matrix *matrix)
{
    if (!matrix)
        return;

    if (matrix->data)
    {
        for (int i = 0; i < matrix->rows; i++)
            if (matrix->data[i])
                free(matrix->data[i]);

        free(matrix->data);
    }

    free(matrix);
}

// Función para multiplicar dos matrices
Matrix *matrix_multiply(Matrix *a, Matrix *b)
{
    if (!a || !b || a->cols != b->rows) // Verificar si las matrices son válidas y compatibles
        return NULL;

    Matrix *result = matrix_create(a->rows, b->cols); // Crear la matriz resultado
    if (!result)
        return NULL;

    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < b->cols; j++)
        {
            result->data[i][j] = 0.0; // Inicializar el elemento de la matriz resultado
            for (int k = 0; k < a->cols; k++)
                result->data[i][j] += a->data[i][k] * b->data[k][j]; // Multiplicación de matrices
        }

    return result; // Devolver la matriz resultado
}

// Función para transponer una matriz
Matrix *matrix_transpose(Matrix *matrix)
{
    if (!matrix) // Verificar si la matriz es válida
        return NULL;

    Matrix *result = matrix_create(matrix->cols, matrix->rows); // Crear la matriz transpuesta
    if (!result)
        return NULL;

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            result->data[j][i] = matrix->data[i][j]; // Asignar los valores transpuestos

    return result; // Devolver la matriz transpuesta
}

// Función para restar dos matrices
Matrix *matrix_subtract(Matrix *a, Matrix *b)
{
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) // Verificar si las matrices son válidas y del mismo tamaño
        return NULL;

    Matrix *result = matrix_create(a->rows, a->cols); // Crear la matriz resultado
    if (!result)
        return NULL;

    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            result->data[i][j] = a->data[i][j] - b->data[i][j]; // Restar los elementos de las matrices

    return result; // Devolver la matriz resultado
}

// Función para calcular la distancia euclidiana entre dos vectores
double euclidean_distance(const double *x1, const double *x2, int n)
{
    double sum = 0.0; // Inicializar la suma de las diferencias al cuadrado

    for (int i = 0; i < n; i++)
    {
        double diff = x1[i] - x2[i]; // Calcular la diferencia entre los elementos
        sum += diff * diff;          // Sumar el cuadrado de la diferencia
    }

    return sqrt(sum); // Devolver la raíz cuadrada de la suma de las diferencias al cuadrado
}