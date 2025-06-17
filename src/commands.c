/*
 * @file: commands.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Archivo para procesar los argumentos de la línea de comandos.
 */

#include "header.h"

void parse_args(char *argv[])
{
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
        show_help();
    else if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0)
        show_version();
    else if (strcmp(argv[1], "-j") == 0 || strcmp(argv[1], "--joke") == 0)
        show_joke("./input/jokes.txt");
    else
        argument_error(argv[1], __FILE__, __LINE__);
}