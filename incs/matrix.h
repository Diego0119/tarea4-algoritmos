/*
 * @file: matrix.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Cabecera general de las funciones de matrices.
 */

#ifndef MATRIX_H
#define MATRIX_H

typedef struct
{
    double **data; // Datos de la matriz
    int rows;      // Número de filas
    int cols;      // Número de columnas
} Matrix;

// Funciones de manejo de matrices
Matrix *matrix_create(int, int);
void matrix_free(Matrix *);

#endif
