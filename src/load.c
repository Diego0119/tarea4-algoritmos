/*
 * @file: load.c
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 21/06/2025
 * @description: Archivo para cargar y procesar datos de un archivo CSV.
 */

#include "libs.h"
#include "csv.h"
#include "config.h"
#include "errors.h"

CSVData *load_csv_data(const char *filename, int has_header, int label_col, char delimiter)
{
    int rows, cols;

    if (!csv_dimensions(filename, has_header, delimiter, &rows, &cols))
        dimensions_error(__FILE__, __LINE__, filename);

    CSVData *csv_data = (CSVData *)malloc(sizeof(CSVData)); // Crear estruuctura para almacenar los datos
    if (!csv_data)
        csv_struct_error(__FILE__, __LINE__, NULL);

    csv_data->has_header = has_header;
    csv_data->label_col = label_col;
    csv_data->header = NULL;

    int data_cols = (label_col >= 0) ? cols - 1 : cols; // Determinar dimensiones de las matrices de datos y etiquetas

    csv_data->data = matrix_create(rows, data_cols); // Crear matrices para datos
    if (!csv_data->data)
        csv_struct_error(__FILE__, __LINE__, csv_data);

    if (label_col >= 0) // Crear matriz para etiquetas
    {
        csv_data->labels = matrix_create(rows, 1);

        if (!csv_data->labels)
        {
            matrix_free(csv_data->data);
            csv_struct_error(__FILE__, __LINE__, csv_data);
        }
    }
    else
        csv_data->labels = NULL;

    if (has_header) // Si hay encabezado, reservar memoria para los nombres de columnas
    {
        csv_data->header = (char **)malloc(cols * sizeof(char *));

        if (!csv_data->header)
        {
            if (csv_data->labels)
                matrix_free(csv_data->labels);

            matrix_free(csv_data->data);

            csv_struct_error(__FILE__, __LINE__, csv_data);
        }

        for (int i = 0; i < cols; i++)
            csv_data->header[i] = NULL;
    }

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        if (csv_data->header)
            free(csv_data->header);

        if (csv_data->labels)
            matrix_free(csv_data->labels);

        matrix_free(csv_data->data);

        free(csv_data);

        open_file_error(__FILE__, __LINE__, filename);
    }

    char line[MAX_LINE_LENGTH];
    char *fields[MAX_FIELDS];
    int row = 0;

    if (has_header && fgets(line, MAX_LINE_LENGTH, file)) // Leer encabezado si existe
    {
        line[strcspn(line, "\r\n")] = 0; // Eliminar salto de línea

        char delim_str[2] = {delimiter, '\0'}; // Crear una cadena con el delimitador para strtok

        char *token = strtok(line, delim_str);
        int col = 0;

        while (token && col < cols)
        {
            while (*token == ' ' || *token == '\t') // Eliminar espacios en blanco al inicio
                token++;

            char *end = token + strlen(token) - 1;

            while (end > token && (*end == ' ' || *end == '\t')) // Eliminar espacios en blanco al final
                end--;

            *(end + 1) = 0;

            csv_data->header[col] = my_strdup(token); // Guardar el nombre de la columna
            token = strtok(NULL, delim_str);
            col++;
        }
    }

    while (fgets(line, MAX_LINE_LENGTH, file) && row < rows) // Leer datos
    {
        line[strcspn(line, "\r\n")] = 0; // Eliminar salto de línea

        char delim_str[2] = {delimiter, '\0'}; // Crear una cadena con el delimitador para strtok

        char *token = strtok(line, delim_str);
        int field = 0;

        while (token && field < MAX_FIELDS)
        {
            while (*token == ' ' || *token == '\t') // Eliminar espacios en blanco al inicio
                token++;

            char *end = token + strlen(token) - 1;

            while (end > token && (*end == ' ' || *end == '\t')) // Eliminar espacios en blanco al final
                end--;

            *(end + 1) = 0;

            fields[field] = token; // Guardar el campo
            token = strtok(NULL, delim_str);
            field++;
        }

        int data_col = 0;

        for (int col = 0; col < cols && col < field; col++) // Llenar matrices de datos y etiquetas
        {
            if (col == label_col) // Esta columna contiene etiquetas
            {
                if (csv_data->labels)
                    csv_data->labels->data[row][0] = atof(fields[col]);
            }
            else // Esta columna contiene datos
            {
                csv_data->data->data[row][data_col] = atof(fields[col]);
                data_col++;
            }
        }

        row++;
    }

    fclose(file);

    return csv_data;
}

int csv_dimensions(const char *filename, int has_header, char delimiter, int *rows, int *cols)
{
    FILE *file = fopen(filename, "r");
    if (!file)
        open_file_error(__FILE__, __LINE__, filename);

    char line[MAX_LINE_LENGTH];

    *rows = 0;
    *cols = 0;

    if (fgets(line, MAX_LINE_LENGTH, file)) // Lee la primera línea para determinar el número de columnas
    {
        char delim_str[2] = {delimiter, '\0'}; // Crea una cadena con el delimitador para strtok

        char *token = strtok(line, delim_str);

        while (token)
        {
            (*cols)++;
            token = strtok(NULL, delim_str);
        }

        *rows = 1;
        while (fgets(line, MAX_LINE_LENGTH, file)) // Cuenta el número de filas
            (*rows)++;

        if (has_header) // Si hay encabezado, ajusta el número de filas
            (*rows)--;
    }

    fclose(file);

    return 1;
}

char *my_strdup(const char *s)
{
    if (s == NULL)
        return NULL;

    size_t len = strlen(s) + 1;

    char *new_str = (char *)malloc(len);
    if (new_str == NULL)
        return NULL;

    return (char *)memcpy(new_str, s, len);
}

void csv_free(CSVData *csv_data)
{
    if (!csv_data)
        return;

    int total_cols = 0;
    if (csv_data->header && csv_data->data) // Si hay encabezado y datos, calcular el número total de columnas
    {
        total_cols = csv_data->data->cols;
        if (csv_data->label_col >= 0) // Calcular el número total de columnas (datos + etiqueta si existe)
            total_cols++;
    }

    if (csv_data->data) // Liberar matriz de datos
        matrix_free(csv_data->data);

    if (csv_data->labels) // Liberar matriz de etiquetas
        matrix_free(csv_data->labels);

    if (csv_data->header) // Liberar nombres de columnas
    {
        for (int i = 0; i < total_cols; i++) // Liberar cada cadena de encabezado
            if (csv_data->header[i])
            {
                free(csv_data->header[i]);
                csv_data->header[i] = NULL;
            }

        free(csv_data->header);
        csv_data->header = NULL;
    }

    free(csv_data);
}