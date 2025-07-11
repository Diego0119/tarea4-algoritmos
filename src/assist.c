/*
 * @file: aux.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Archivo con funciones auxiliares para el proyecto de C.
 */

#include "libs.h"
#include "config.h"
#include "csv.h"

// muestra un mensaje de ayuda del programa
void show_help(void)
{
    fprintf(stdout, CYAN_COLOR "\nAYUDA DEL PROGRAMA:\n\n");
    fprintf(stdout, "uso:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t./build/program.out [opcion]\n");
    fprintf(stdout, "\t./build/program.out [opcion] [archivo] [parametros necesarios] . . .\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "opciones:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t-h\t\tmuestra esta ayuda.\n");
    fprintf(stdout, "\t-v\t\tmuestra la version del programa.\n");
    fprintf(stdout, "\t-k\t\taplicar algoritmo k-nearest neighbors.\n");
    fprintf(stdout, "\t-l\t\taplicar regresion lineal.\n");
    fprintf(stdout, "\t-m\t\taplicar clustering k-means.\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "parametros de k-nearest neighbors:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t[k]\t\t= numero de vecinos (debe ser impar y mayor a 0).\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "parametros de regresion lineal:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t[learning rate]\t= tamano de paso de cada iteracion.\n");
    fprintf(stdout, "\t[iteraciones]\t= maximo de iteraciones.\n");
    fprintf(stdout, "\t[tolerancia]\t= tolerancia de convergencia.\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "parametros de k-means:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t[k]\t\t= numero de clusters (debe ser mayor a 0).\n");
    fprintf(stdout, "\t[iteraciones]\t= maximo de iteraciones.\n");
    fprintf(stdout, "\t[tolerancia]\t= tolerancia de convergencia.\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "ejemplos:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t./build/program.out -h\n");
    fprintf(stdout, "\t./build/program.out -v\n");
    fprintf(stdout, "\t./build/program.out -k ./data/iris.csv 1\n");
    fprintf(stdout, "\t./build/program.out -l ./data/iris.csv 0.001 2000 1e-8\n");
    fprintf(stdout, "\t./build/program.out -m ./data/iris.csv 3 100 1e-4\n");
    fprintf(stdout, "\tmake run (ejecuta todos los comandos disponibles)\n\n" RESET_COLOR);
    exit(EXIT_SUCCESS);
}

// Muestra la versión del programa
void show_version(void)
{
    fprintf(stdout, CYAN_COLOR "\nVersion del programa: 5.0.2\n\n" RESET_COLOR);
    exit(EXIT_SUCCESS);
}

// Imprime los datos al cargar el DF a la estructura CSVData
void print_csv_data(CSVData *csv_data)
{
    fprintf(stdout, GREEN_COLOR "\nDimensiones de los datos: %d filas x %d columnas\n", csv_data->data->rows, csv_data->data->cols);

    if (csv_data->has_header && csv_data->header)
    {
        fprintf(stdout, "\nEncabezados: ");

        for (int i = 0; i < csv_data->data->cols; i++)
            fprintf(stdout, "%s ", csv_data->header[i]);

        fprintf(stdout, "%s (etiqueta)\n" RESET_COLOR, csv_data->header[csv_data->label_col]);
    }

    fprintf(stdout, YELLOW_COLOR "\nPrimeras 5 muestras:\n\n" RESET_COLOR);
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
