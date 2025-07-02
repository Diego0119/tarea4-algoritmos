/*
 * @file: main.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Archivo principal del proyecto de C.
 */

#include "libs.h"
#include "errors.h"
#include "utils.h"

// Función principal
int main(int argc, char *argv[])
{
    srand(3);

    parse_args(argc, argv);

    return EXIT_SUCCESS;
}