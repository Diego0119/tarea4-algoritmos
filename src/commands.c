/*
 * @file: commands.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Archivo para procesar los argumentos de la línea de comandos.
 */

#include "libs.h"
#include "config.h"
#include "csv.h"
#include "utils.h"
#include "errors.h"
#include "lr.h"
#include "k-means.h"

// Se encarga de procesar los argumentos de la línea de comandos (EN EL FUTURO CAMBIAR POR GETOPT)
void parse_args(char *argv[])
{
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
        show_help();
    else if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0)
        show_version();
    else if (strcmp(argv[1], "-knn") == 0 || strcmp(argv[1], "--neighboor") == 0)
    {
        if (argv[2] != NULL && argv[3] != NULL)
        {
            const char *filename = argv[2];

            const char *extension = strrchr(filename, '.');
            if (extension == NULL || (strcmp(extension, ".csv") != 0 && strcmp(extension, ".xlsx") != 0))
                csv_extension_error(__FILE__, __LINE__, filename);

            int k = atoi(argv[3]);
            if (k <= 0 || k % 2 == 0)
                k_parameter_error(__FILE__, __LINE__);

            CSVData *csv_data = load_csv_data(filename, 1, 4, ',');
            if (!csv_data)
                read_csv_error(__FILE__, __LINE__, filename);

            fprintf(stdout, GREEN_COLOR "\nDatos cargados correctamente desde: %s.\n" RESET_COLOR, filename);

            print_csv_data(csv_data);

            exec_knn(csv_data, k);

            csv_free(csv_data);
        }
        else
            argument_error(argv[2], __FILE__, __LINE__);
    }
    else if (strcmp(argv[1], "-lr") == 0 || strcmp(argv[1], "--linear") == 0)
    {
        if (argv[2] != NULL && argv[3] != NULL && argv[4] != NULL && argv[5] != NULL)
        {
            const char *filename = argv[2];

            double learning_rate = atof(argv[3]);
            if (learning_rate <= 0.0)
                learning_rate_parameter_error(__FILE__, __LINE__);

            int max_iterations = atoi(argv[4]);
            if (max_iterations <= 0)
                iterations_parameter_error(__FILE__, __LINE__);

            double tolerance = atof(argv[5]);
            if (tolerance <= 0.0)
                tolerance = 1e-6;

            const char *extension = strrchr(filename, '.');
            if (extension == NULL || (strcmp(extension, ".csv") != 0 && strcmp(extension, ".xlsx") != 0))
                csv_extension_error(__FILE__, __LINE__, filename);

            CSVData *csv_data = load_csv_data(filename, 1, 0, ',');
            if (!csv_data)
                read_csv_error(__FILE__, __LINE__, filename);

            fprintf(stdout, GREEN_COLOR "\nDatos cargados correctamente desde: %s.\n" RESET_COLOR, filename);

            print_csv_data(csv_data);

            exec_linear_regression(csv_data, learning_rate, max_iterations, tolerance);

            csv_free(csv_data);
        }
        else
            argument_error(argv[2], __FILE__, __LINE__);
    }
    else if (strcmp(argv[1], "-km") == 0 || strcmp(argv[1], "--kmeans") == 0)
    {
        if (argv[2] != NULL && argv[3] != NULL && argv[4] != NULL && argv[5] != NULL)
        {
            const char *filename = argv[2];
            int k = atoi(argv[3]);
            if (k <= 0 || k % 2 == 0)
                k_parameter_error(__FILE__, __LINE__);

            int max_iterations = atoi(argv[4]);
            if (max_iterations <= 0)
                iterations_parameter_error(__FILE__, __LINE__);

            double tolerance = atof(argv[5]);
            if (tolerance <= 0.0)
                tolerance = 1e-6;

            const char *extension = strrchr(filename, '.');
            if (!extension || (strcmp(extension, ".csv") != 0 && strcmp(extension, ".xlsx") != 0))
                csv_extension_error(__FILE__, __LINE__, filename);

            CSVData *csv_data = load_csv_data(filename, 1, 4, ',');
            if (!csv_data)
                read_csv_error(__FILE__, __LINE__, filename);

            fprintf(stdout, GREEN_COLOR "\nDatos cargados correctamente desde: %s.\n" RESET_COLOR, filename);

            exec_kmeans(csv_data, k, max_iterations, tolerance);

            csv_free(csv_data);
        }
        else
            argument_error(argv[2], __FILE__, __LINE__);
    }
    else
    {
        argument_error(argv[1], __FILE__, __LINE__);
    }
}