/*
 * @file: aux.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Archivo con funciones auxiliares para el proyecto de C.
 */

#include "libs.h"
#include "config.h"
#include "csv.h"

// Muestra un mensaje de ayuda del programa
void show_help(void)
{
    fprintf(stdout, CYAN_COLOR "\nAyuda del programa:\n\n");
    fprintf(stdout, "Uso:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t./build/program.out [opcion]\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "Opciones:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t-h, --help\tMuestra esta ayuda.\n");
    fprintf(stdout, "\t-v, --version\tMuestra la version del programa.\n");
    fprintf(stdout, "\t-knn\t\tAplicar algoritmo K-Nearest Neighboors.\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "Ejemplos:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t./build/program.out -h\n");
    fprintf(stdout, "\t./build/program.out -knn ./input/iris.csv\n\n" RESET_COLOR);
    exit(EXIT_SUCCESS);
}

// Muestra la versión del programa
void show_version(void)
{
    fprintf(stdout, CYAN_COLOR "\nVersion del programa: 1.3.3\n\n" RESET_COLOR);
    exit(EXIT_SUCCESS);
}

// Imprime los datos al cargar el DF a la estructura CSVData
void print_csv_data(CSVData *csv_data)
{
    fprintf(stdout, "\nDimensiones de los datos: %d filas x %d columnas\n", csv_data->data->rows, csv_data->data->cols);

    if (csv_data->has_header && csv_data->header)
    {
        fprintf(stdout, "\nEncabezados: ");

        for (int i = 0; i < csv_data->data->cols; i++)
            fprintf(stdout, "%s ", csv_data->header[i]);

        fprintf(stdout, "%s (etiqueta)\n", csv_data->header[csv_data->label_col]);
    }

    fprintf(stdout, "\nPrimeras 5 muestras:\n");
    for (int i = 0; i < 5 && i < csv_data->data->rows; i++)
    {
        fprintf(stdout, "Muestra %d: [", i);

        for (int j = 0; j < csv_data->data->cols; j++)
        {
            fprintf(stdout, "%.1f", csv_data->data->data[i][j]);

            if (j < csv_data->data->cols - 1)
                fprintf(stdout, ", ");
        }

        fprintf(stdout, "] -> Clase: %.0f\n", csv_data->labels->data[i][0]);
    }

    fprintf(stdout, "\n");
}
