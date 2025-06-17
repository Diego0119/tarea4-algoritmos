/*
 * @file: header.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Cabecera general temporal para el proyecto de C.
 */

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

#define N_OF_CHARS 256 // Número máximo de caracteres en una palabra
#define TOTAL_JOKES 10 // Número total de chistes disponibles

// Función par procesar los argumentos de la línea de comandos
void parse_args(char **);

// Funciones auxiliares
void show_help(void);
void show_version(void);
void show_joke(const char *);

// Funciones de manejo de errores
void handle_error(const char *, const char *, const char *, int);
void number_arguments_error(const char *, int);
void argument_error(const char *, const char *, int);
void open_file_error(const char *, int, const char *);