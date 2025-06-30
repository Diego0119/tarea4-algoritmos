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
    fprintf(stdout, "\t./build/program.out [opcion] [parametros] [archivo]\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "opciones:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t-h, --help\t\tmuestra esta ayuda.\n");
    fprintf(stdout, "\t-v, --version\t\tmuestra la version del programa.\n");
    fprintf(stdout, "\t-knn, --neighboor\taplicar algoritmo k-nearest neighbors.\n");
    fprintf(stdout, "\t-lr, --linear\t\taplicar regresion lineal.\n");
    fprintf(stdout, "\t-km, --kmeans\t\taplicar clustering k-means.\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "parametros de k-nearest neighbors:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t[k]       = numero de vecinos (debe ser impar y mayor a 0)\n");
    fprintf(stdout, "\t[archivo] = archivo CSV o Excel con los datos\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "parametros de regresion lineal:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t[lr]   = learning rate (por defecto: 0.01)\n");
    fprintf(stdout, "\t[iter] = maximo de iteraciones (por defecto: 1000)\n");
    fprintf(stdout, "\t[tol]  = tolerancia de convergencia (por defecto: 1e-6)\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "parametros de k-means:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t[k]         = numero de clusters (mayor a 0)\n");
    fprintf(stdout, "\t[max_iters] = maximo de iteraciones (por defecto: 100)\n");
    fprintf(stdout, "\t[tolerance] = tolerancia de convergencia (por defecto: 1e-4)\n\n" RESET_COLOR);
    fprintf(stdout, CYAN_COLOR "ejemplos:\n" RESET_COLOR);
    fprintf(stdout, YELLOW_COLOR "\t./build/program.out -h\n");
    fprintf(stdout, "\t./build/program.out -knn 3 ./data/iris.csv\n");
    fprintf(stdout, "\t./build/program.out -lr ./data/iris.csv\n");
    fprintf(stdout, "\t./build/program.out -lr ./data/iris.csv 0.001 2000 1e-8\n");
    fprintf(stdout, "\t./build/program.out -km ./data/iris.csv 3 100 1e-4\n\n" RESET_COLOR);
    exit(EXIT_SUCCESS);
}

// Muestra la versión del programa
void show_version(void)
{
    fprintf(stdout, CYAN_COLOR "\nVersion del programa: 3.0.2\n\n" RESET_COLOR);
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
