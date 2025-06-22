/*
 * @file: header.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Cabecera general temporal para el proyecto de C.
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define RED_COLOR "\033[0;31m"
#define GREEN_COLOR "\033[0;32m"
#define CYAN_COLOR "\033[0;36m"
#define YELLOW_COLOR "\033[0;33m"
#define BRIGHT_PURPLE_COLOR "\033[1;35m"
#define RESET_COLOR "\033[0m"

#define N_OF_CHARS 256       // Número máximo de caracteres en una palabra
#define MAX_LINE_LENGTH 4096 // Longitud máxima de una línea en el CSV
#define MAX_FIELDS 256       // Número máximo de campos en una línea del CSV

typedef struct
{
    double **data; // Datos de la matriz
    int rows;      // Número de filas
    int cols;      // Número de columnas
} Matrix;

typedef struct
{
    Matrix *data;   // Matriz con los datos numéricos
    Matrix *labels; // Vector con las etiquetas (opcional)
    char **header;  // Nombres de las columnas (opcional)
    int has_header; // Indica si el CSV tenía encabezado
    int label_col;  // Índice de la columna de etiquetas (-1 si no hay)
} CSVData;

// Función par procesar los argumentos de la línea de comandos
void parse_args(char **);

// Funciones de carga de datos
CSVData *load_csv_data(const char *, int, int, char);
int csv_dimensions(const char *, int, char, int *, int *);
char *my_strdup(const char *);
void csv_free(CSVData *);

// Funciones de manejo de matrices
Matrix *matrix_create(int, int);
void matrix_free(Matrix *);

// Funciones auxiliares
void show_help(void);
void show_version(void);
void print_csv_data(CSVData *);

// Funciones de manejo de errores
void handle_error(const char *, const char *, const char *, int);
void number_arguments_error(const char *, int);
void argument_error(const char *, const char *, int);
void open_file_error(const char *, int, const char *);
void read_csv_error(const char *, int, const char *);
void dimensions_error(const char *, int, const char *);
void csv_struct_error(const char *, int, CSVData *);
void matrix_struct_error(const char *, int, Matrix *);