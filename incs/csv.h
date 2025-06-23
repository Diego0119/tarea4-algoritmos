/*
 * @file: header.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Cabecera general de las funciones relacionadas a manejo de csv.
 */

#ifndef CSV_H
#define CSV_H

#include "matrix.h"

typedef struct
{
    Matrix *data;   // Matriz con los datos numéricos
    Matrix *labels; // Vector con las etiquetas (opcional)
    char **header;  // Nombres de las columnas (opcional)
    int has_header; // Indica si el CSV tenía encabezado
    int label_col;  // Índice de la columna de etiquetas (-1 si no hay)
} CSVData;

// Funciones de carga de datos
CSVData *load_csv_data(const char *, int, int, char);
int csv_dimensions(const char *, int, char, int *, int *);
char *my_strdup(const char *);
void csv_free(CSVData *);
void print_csv_data(CSVData *);

#endif
