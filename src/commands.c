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
        if (argv[2] != NULL)
        {
            int k = atoi(argv[2]);
            if (k <= 0 || k % 2 == 0)
                k_parameter_error(__FILE__, __LINE__);

            if (argv[3] != NULL)
            {
                const char *filename = argv[3];
                const char *extension = strrchr(filename, '.');
                if (extension == NULL || (strcmp(extension, ".csv") != 0 && strcmp(extension, ".xlsx") != 0))
                    csv_extension_error(__FILE__, __LINE__, filename);

                CSVData *csv_data = load_csv_data(filename, 1, 4, ',');
                if (!csv_data)
                    read_csv_error(__FILE__, __LINE__, filename);

                fprintf(stdout, GREEN_COLOR "\nDatos cargados correctamente desde: %s.\n" RESET_COLOR, filename);

                print_csv_data(csv_data);

                exec_knn(csv_data, k);

                csv_free(csv_data);
            }
            else
                argument_error(argv[3], __FILE__, __LINE__);
        }
        else
            argument_error(argv[2], __FILE__, __LINE__);
    }
    // regrecion lineal
    else if (strcmp(argv[1], "-lr") == 0 || strcmp(argv[1], "--linear") == 0)
    {
        if (argv[2] != NULL && argv[3] != NULL && argv[4] != NULL && argv[5] != NULL)
        {
            const char *filename = argv[2];
            double learning_rate = atof(argv[3]);
            int max_iterations = atoi(argv[4]);
            double tolerance = atof(argv[5]);
            if (learning_rate <= 0.0)
                learning_rate_parameter_error(__FILE__, __LINE__);
            if (max_iterations <= 0)
                iterations_parameter_error(__FILE__, __LINE__);
            if (tolerance <= 0.0)
                tolerance = 1e-6;
            const char *extension = strrchr(filename, '.');
            if (extension == NULL || (strcmp(extension, ".csv") != 0 && strcmp(extension, ".xlsx") != 0))
                csv_extension_error(__FILE__, __LINE__, filename);
            CSVData *csv_data = load_csv_data(filename, 1, 0, ',');
            if (!csv_data)
                read_csv_error(__FILE__, __LINE__, filename);
            printf(CYAN_COLOR "\n╔══════════════════════════════════════════════════════════════╗\n");
            printf("║                    " BRIGHT_PURPLE_COLOR "REGRESION LINEAL" CYAN_COLOR "                    ║\n");
            printf("╚══════════════════════════════════════════════════════════════╝\n" RESET_COLOR);
            printf(GREEN_COLOR "\nDatos cargados correctamente desde: %s.\n" RESET_COLOR, filename);
            printf(YELLOW_COLOR "\nEntrenando modelo...\n" RESET_COLOR);
            exec_linear_regression(csv_data, learning_rate, max_iterations, tolerance);
            printf(YELLOW_COLOR "\nFin del analisis de regresion lineal.\n" RESET_COLOR);
            csv_free(csv_data);
        }
        else
            argument_error(argv[1], __FILE__, __LINE__);
    }
    else if (strcmp(argv[1], "-km") == 0 || strcmp(argv[1], "--kmeans") == 0)
    {
        int k = 3;
        int max_iters = 100;
        double tolerance = 1e-4;
        const char *filename = NULL;

        int arg_index = 2;

        if (argv[arg_index] != NULL)
        {
            filename = argv[arg_index];
            arg_index++;

            const char *extension = strrchr(filename, '.');
            if (extension == NULL || (strcmp(extension, ".csv") != 0 && strcmp(extension, ".xlsx") != 0))
                csv_extension_error(__FILE__, __LINE__, filename);
        }
        else
        {
            printf(CYAN_COLOR "\n╔══════════════════════════════════════════════════════════════╗\n");
            printf("║                         " BRIGHT_PURPLE_COLOR "K-MEANS" CYAN_COLOR "                            ║\n");
            printf("╚══════════════════════════════════════════════════════════════╝\n" RESET_COLOR);
            printf(YELLOW_COLOR "\nuso:\n" RESET_COLOR);
            printf("   %s -km " GREEN_COLOR "<archivo.csv>" RESET_COLOR " [k] [max_iters] [tolerance]\n\n", argv[0]);
            printf(YELLOW_COLOR "ejemplo:\n" RESET_COLOR);
            printf("   %s -km " GREEN_COLOR "./data/iris.csv" RESET_COLOR " 3 100 1e-4\n\n", argv[0]);
            exit(EXIT_SUCCESS);
        }

        if (argv[arg_index] != NULL)
        {
            k = atoi(argv[arg_index]);
            if (k <= 0)
            {
                fprintf(stderr, RED_COLOR "⚠ Error: k debe ser un entero positivo.\n" RESET_COLOR);
                exit(EXIT_FAILURE);
            }
            arg_index++;
        }

        if (argv[arg_index] != NULL)
        {
            max_iters = atoi(argv[arg_index]);
            if (max_iters <= 0)
            {
                fprintf(stderr, RED_COLOR "⚠ Error: max_iters debe ser un entero positivo.\n" RESET_COLOR);
                exit(EXIT_FAILURE);
            }
            arg_index++;
        }

        if (argv[arg_index] != NULL)
        {
            tolerance = atof(argv[arg_index]);
            if (tolerance <= 0.0)
            {
                fprintf(stderr, RED_COLOR "⚠ Error: tolerance debe ser un número positivo.\n" RESET_COLOR);
                exit(EXIT_FAILURE);
            }
        }

        CSVData *csv_data = load_csv_data(filename, 1, 4, ',');
        if (!csv_data)
            read_csv_error(__FILE__, __LINE__, filename);

        fprintf(stdout, GREEN_COLOR "\nDatos cargados correctamente desde: %s.\n" RESET_COLOR, filename);

        exec_kmeans(csv_data, k, max_iters, tolerance);

        csv_free(csv_data);
    }
    else
        argument_error(argv[1], __FILE__, __LINE__);
}