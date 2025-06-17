/*
 * @file: aux.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Archivo con funciones auxiliares para el proyecto de C.
 */

#include "header.h"

void show_help(void)
{
    fprintf(stdout, CYAN_COLOR "\nAyuda del programa:\n\n");
    fprintf(stdout, "Uso:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t./build/program.out [opcion]\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "Opciones:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t-h, --help\tMuestra esta ayuda.\n");
    fprintf(stdout, "\t-v, --version\tMuestra la version del programa.\n");
    fprintf(stdout, "\t-j, --joke\tMuestra un chiste aleatorio.\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "Ejemplos:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t./build/program.out -h\n\n" RESET_COLOR);
    exit(EXIT_SUCCESS);
}

void show_version(void)
{
    fprintf(stdout, CYAN_COLOR "\nVersion del programa: 1.0.0\n\n" RESET_COLOR);
    exit(EXIT_SUCCESS);
}

void show_joke(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
        open_file_error(__FILE__, __LINE__, filename);

    char line[N_OF_CHARS];

    srand(time(NULL));

    int random_line = rand() % TOTAL_JOKES;
    int current_line = 0;

    while (fgets(line, sizeof(line), file) != NULL)
    {
        if (current_line == random_line)
        {
            fprintf(stdout, BRIGHT_PURPLE_COLOR "\nChiste: %s\n" RESET_COLOR, line);
            break;
        }

        current_line++;
    }

    fclose(file);

    exit(EXIT_SUCCESS);
}