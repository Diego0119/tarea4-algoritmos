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

// Se encarga de procesar los argumentos de la línea de comandos usando getopt (sin struct option)
void parse_args(int argc, char *argv[])
{
    int opt;

    if (argc < 2)
    {
        show_help();
        return;
    }

    while ((opt = getopt(argc, argv, "hvk:l:m:")) != -1)
    {
        switch (opt)
        {
        case 'h':
            show_help();
            break;
        case 'v':
            show_version();
            break;
        case 'k':
            if (optarg && optind < argc)
            {
                const char *filename = optarg;

                const char *extension = strrchr(filename, '.');
                if (extension == NULL || (strcmp(extension, ".csv") != 0 && strcmp(extension, ".xlsx")))
                    csv_extension_error(__FILE__, __LINE__, filename);

                int k = atoi(argv[optind]);
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
                argument_error("Faltan argumentos para KNN", __FILE__, __LINE__);
            break;
        case 'l':
            if (optarg && optind + 1 < argc)
            {
                const char *filename = optarg;

                const char *extension = strrchr(filename, '.');
                if (extension == NULL || (strcmp(extension, ".csv") != 0 && strcmp(extension, ".xlsx")))
                    csv_extension_error(__FILE__, __LINE__, filename);

                double learning_rate = atof(argv[optind]);
                if (learning_rate <= 0.0)
                    learning_rate_parameter_error(__FILE__, __LINE__);

                int max_iterations = atoi(argv[optind + 1]);
                if (max_iterations <= 0)
                    iterations_parameter_error(__FILE__, __LINE__);

                double tolerance = (optind + 2 < argc) ? atof(argv[optind + 2]) : 1e-6;
                if (tolerance <= 0.0)
                    tolerance = 1e-6;

                exec_linear_regression_from_csv(filename, learning_rate, max_iterations, tolerance, LR_METHOD, LR_REGULARIZATION, LR_LAMBDA);
            }
            else
                argument_error("Faltan argumentos para regresión lineal", __FILE__, __LINE__);
            break;
        case 'm':
            if (optarg && optind + 1 < argc)
            {
                const char *filename = optarg;

                const char *extension = strrchr(filename, '.');
                if (extension == NULL || (strcmp(extension, ".csv") != 0 && strcmp(extension, ".xlsx")))
                    csv_extension_error(__FILE__, __LINE__, filename);

                int k = atoi(argv[optind]);
                if (k <= 0 || k % 2 == 0)
                    k_parameter_error(__FILE__, __LINE__);

                int max_iterations = atoi(argv[optind + 1]);
                if (max_iterations <= 0)
                    iterations_parameter_error(__FILE__, __LINE__);

                double tolerance = (optind + 2 < argc) ? atof(argv[optind + 2]) : 1e-6;
                if (tolerance <= 0.0)
                    tolerance = 1e-6;

                CSVData *csv_data = load_csv_data(filename, 1, 4, ',');
                if (!csv_data)
                    read_csv_error(__FILE__, __LINE__, filename);

                fprintf(stdout, GREEN_COLOR "\nDatos cargados correctamente desde: %s.\n" RESET_COLOR, filename);

                print_csv_data(csv_data);

                exec_kmeans(csv_data, k, max_iterations, tolerance);

                csv_free(csv_data);
            }
            else
                argument_error("Faltan argumentos para KMeans", __FILE__, __LINE__);
            break;
        default:
            argument_error(argv[1], __FILE__, __LINE__);
        }
    }
}