/*
 * @file: exeptions.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Archivo para manejar excepciones y errores en el proyecto de C.
 */

#include "libs.h"
#include "config.h"
#include "csv.h"

// Función principal para manejar errores de manera centralizada
void handle_error(const char *header, const char *detail, const char *file, int line)
{
    fprintf(stderr, RED_COLOR "\nERROR: %s\n", header);
    fprintf(stderr, "-> Descripcion : %s\n", detail);
    fprintf(stderr, "-> Ubicacion   : %s, Linea: %d\n", file, line);
    fprintf(stderr, "-> Accion      : Finalizando ejecucion.\n\n" RESET_COLOR);
    exit(EXIT_FAILURE);
}

// Error de cantidad de argumentos por terminal
void number_arguments_error(const char *file, int line)
{
    handle_error("Numero de argumentos incorrecto", "Se esperaban mas argumentos", file, line);
}

// Error de argumento proporcionado por terminal
void argument_error(const char *arg, const char *file, int line)
{
    handle_error("Argumento proporcionado es invalido", arg, file, line);
}

// Error al abrir un archivo
void open_file_error(const char *file, int line, const char *filename)
{
    handle_error("No se pudo abrir el archivo", filename, file, line);
}

// Error al leer el archivo CSV de pruebas
void read_csv_error(const char *file, int line, const char *filename)
{
    handle_error("No se puedo leer el archivo CSV de pruebas", filename, file, line);
}

// Error al determinar las dimensiones del archivo CSV
void dimensions_error(const char *file, int line, const char *filename)
{
    handle_error("Dimensiones mal determinadas en el archivo CSV", filename, file, line);
}

// Error al crear la estructura CSVData
void csv_struct_error(const char *file, int line, CSVData *csv_data)
{
    if (csv_data != NULL)
        free(csv_data);

    handle_error("No se pudo crear la estructura CSVData", "Memoria insuficiente", file, line);
}

// Error al crear la estructura Matrix
void matrix_struct_error(const char *file, int line, Matrix *matrix)
{
    if (matrix != NULL)
        free(matrix);

    handle_error("No se pudo crear la estructura Matrix", "Memoria insuficiente", file, line);
}

void csv_extension_error(const char *file, int line, const char *filename)
{
    handle_error("La extension del archivo no es .csv", filename, file, line);
}