/*
 * @file: commands.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Archivo para procesar los argumentos de la línea de comandos.
 */

#include "header.h"

void parse_args(char *argv[])
{
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
        show_help();
    else if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0)
        show_version();
    else if (strcmp(argv[1], "-t") == 0 || strcmp(argv[1], "-test") == 0)
    {
        if (argv[2] != NULL)
        {
            fprintf(stdout, CYAN_COLOR "\nArchivo de datos de prueba desde: %s\n" RESET_COLOR, argv[2]);

            CSVData *csv_data = load_csv_data(argv[2], 1, 4, ',');
            if (!csv_data)
                read_csv_error(__FILE__, __LINE__, argv[2]);

            fprintf(stdout, GREEN_COLOR "\nDatos cargados correctamente.\n" RESET_COLOR);

            print_csv_data(csv_data);

            csv_free(csv_data);
        }
        else
            number_arguments_error(__FILE__, __LINE__);
    }
    else
        argument_error(argv[1], __FILE__, __LINE__);
}