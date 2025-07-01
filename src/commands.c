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
            if (extension == NULL || (strcmp(extension, ".csv") != 0 && strcmp(extension, ".xlsx") != 0))
                csv_extension_error(__FILE__, __LINE__, filename);

            CSVData *csv_data = load_csv_data(filename, 1, 0, ',');
            if (!csv_data)
                read_csv_error(__FILE__, __LINE__, filename);

            fprintf(stdout, GREEN_COLOR "\nDatos cargados correctamente desde: %s.\n" RESET_COLOR, filename);

            print_csv_data(csv_data);

            /* TOOOOODO ESTO DEBE IR EN UNA FUNCIÓN LLAMADA EXEC_KMEANS(), COMO SE HACE CON KNN Y CON LR

            PD: FIJATE SI AL ENTRENAR TU ALGORITMO K-MEANS CON LA FUNCIÓN TRAIN_VALID_TEST_SPLIT, NECESITAS LA PARTE DEL VALIDADO, YO CREO QUE SI... PORQUE AHORA SE ENTRENA UN 60%, SE VALIDA UN 20% Y SE PRUEBA UN 20% DEL CONJUNTO DE DATOS

            Matrix *data = csv_data->data;

            KMeansResult *result = kmeans_fit(data, k, max_iters, tolerance);

            printf(CYAN_COLOR "\nCentroides finales:\n" RESET_COLOR);
            for (int i = 0; i < k; i++)
            {
                printf("Centroide %d: (", i);
                for (int j = 0; j < result->centroids->cols; j++)
                {
                    printf("%.4f", result->centroids->data[i][j]);
                    if (j < result->centroids->cols - 1)
                        printf(", ");
                }
                printf(")\n");
            }

            printf(BRIGHT_PURPLE_COLOR "\nAsignaciones de cluster:\n" RESET_COLOR);
            for (int i = 0; i < data->rows; i++)
            {
                printf("Punto %d → Cluster %d\n", i, result->labels[i]);
            }

            kmeans_free(result);

            */

            csv_free(csv_data);
        }
    }
    else
        argument_error(argv[1], __FILE__, __LINE__);
}