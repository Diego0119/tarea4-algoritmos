/*
 * @file: main.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Archivo principal del proyecto de C.
 */

#include "header.h"

int main(int argc, char *argv[])
{
    if (argc < 2)
        number_arguments_error(__FILE__, __LINE__);

    parse_args(argv);

    return EXIT_SUCCESS;
}