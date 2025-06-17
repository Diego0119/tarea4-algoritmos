/*
 * @file: exeptions.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Archivo para manejar excepciones y errores en el proyecto de C.
 */

#include "header.h"

void handle_error(const char *header, const char *detail, const char *file, int line)
{
    fprintf(stderr, RED_COLOR "\nERROR: %s\n", header);
    fprintf(stderr, "-> Descripcion : %s\n", detail);
    fprintf(stderr, "-> Ubicacion   : %s, Linea: %d\n", file, line);
    fprintf(stderr, "-> Accion      : Finalizando ejecucion.\n\n" RESET_COLOR);
    exit(EXIT_FAILURE);
}

void number_arguments_error(const char *file, int line)
{
    handle_error("Numero de argumentos incorrecto", "Se esperaban mas argumentos", file, line);
}

void argument_error(const char *arg, const char *file, int line)
{
    handle_error("Argumento proporcionado es invalido", arg, file, line);
}

void open_file_error(const char *file, int line, const char *filename)
{
    handle_error("No se pudo abrir el archivo", filename, file, line);
}