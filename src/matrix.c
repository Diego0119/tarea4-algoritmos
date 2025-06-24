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