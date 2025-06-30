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

// Función para cargar datos desde un archivo CSV
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
                if (csv_data->labels) // Si hay etiquetas, almacenar el valor
                {
                    if (fields[col][0] == '\0')
                        csv_data->labels->data[row][0] = NAN; // Valor faltante
                    else
                        csv_data->labels->data[row][0] = atof(fields[col]); // Convertir a número
                }
            }
            else // Esta columna contiene datos
            {
                if (fields[col][0] == '\0')
                    csv_data->data->data[row][data_col] = NAN; // Valor faltante
                else
                    csv_data->data->data[row][data_col] = atof(fields[col]); // Convertir a número
                data_col++;
            }
        }

        row++;
    }

    fclose(file);

    normalize_csv_data(csv_data->data);

    return csv_data;
}

// Función para determinar las dimensiones de un archivo CSV
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

// Función personalizada de duplicación de cadenas
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

// Liberar memoria de la estructura CSVData
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

// Función para dividir un conjunto de datos en conjuntos de entrenamiento y prueba
int train_test_split(Matrix *data, Matrix *labels, double test_ratio, Matrix **X_train, Matrix **y_train, Matrix **X_test, Matrix **y_test)
{
    if (!data || test_ratio < 0.0 || test_ratio > 1.0)
        return 0;

    int n_samples = data->rows;
    int n_features = data->cols;

    // Calcular tamaños de conjuntos de entrenamiento y prueba
    int test_size = (int)(test_ratio * n_samples);
    int train_size = n_samples - test_size;
    if (train_size <= 0 || test_size <= 0)
        return 0;

    // Crear matrices para los conjuntos de entrenamiento y prueba
    *X_train = matrix_create(train_size, n_features);
    *X_test = matrix_create(test_size, n_features);

    if (!*X_train || !*X_test)
    {
        if (*X_train)
            matrix_free(*X_train);
        if (*X_test)
            matrix_free(*X_test);
        return 0;
    }

    // Si hay etiquetas, crear matrices para ellas también
    if (labels)
    {
        *y_train = matrix_create(train_size, 1);
        *y_test = matrix_create(test_size, 1);

        if (!*y_train || !*y_test)
        {
            matrix_free(*X_train);
            matrix_free(*X_test);
            if (*y_train)
                matrix_free(*y_train);
            if (*y_test)
                matrix_free(*y_test);
            return 0;
        }
    }
    else
    {
        *y_train = NULL;
        *y_test = NULL;
    }

    // Crear un arreglo de índices y mezclarlo aleatoriamente
    int *index = (int *)malloc(n_samples * sizeof(int));
    if (!index)
    {
        matrix_free(*X_train);
        matrix_free(*X_test);
        if (*y_train)
            matrix_free(*y_train);
        if (*y_test)
            matrix_free(*y_test);
        return 0;
    }

    for (int i = 0; i < n_samples; i++)
        index[i] = i;

    // Mezclar los índices (Fisher-Yates shuffle)
    for (int i = n_samples - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int temp = index[i];
        index[i] = index[j];
        index[j] = temp;
    }

    // Llenar conjuntos de entrenamiento y prueba
    for (int i = 0; i < train_size; i++)
    {
        int idx = index[i];
        for (int j = 0; j < n_features; j++)
            (*X_train)->data[i][j] = data->data[idx][j];

        if (labels && *y_train)
            (*y_train)->data[i][0] = labels->data[idx][0];
    }

    for (int i = 0; i < test_size; i++)
    {
        int idx = index[train_size + i];
        for (int j = 0; j < n_features; j++)
            (*X_test)->data[i][j] = data->data[idx][j];

        if (labels && *y_test)
            (*y_test)->data[i][0] = labels->data[idx][0];
    }

    free(index);

    return 1;
}

// Función para normalizar los datos de un CSV (detecta espacios y completa datos faltantes) utilizando Min-Max Scaling
void normalize_csv_data(Matrix *data)
{
    if (!data)
        return;

    int rows = data->rows;
    int cols = data->cols;

    double *min = (double *)malloc(cols * sizeof(double));
    double *max = (double *)malloc(cols * sizeof(double));
    double *mean = (double *)malloc(cols * sizeof(double));
    int *count = (int *)malloc(cols * sizeof(int));
    if (!min || !max || !mean || !count)
        return;

    for (int j = 0; j < cols; j++) // Inicializar
    {
        min[j] = DBL_MAX;
        max[j] = -DBL_MAX;
        mean[j] = 0.0;
        count[j] = 0;
    }

    for (int j = 0; j < cols; j++) // Calcular media ignorando NAN
    {
        for (int i = 0; i < rows; i++)
        {
            double v = data->data[i][j];

            if (!isnan(v)) // Verificar si el valor no es NAN
            {
                mean[j] += v;
                count[j]++;

                if (v < min[j]) // Actualizar mínimo
                    min[j] = v;

                if (v > max[j]) // Actualizar máximo
                    max[j] = v;
            }
        }

        if (count[j] > 0) // Calcular la media
            mean[j] /= count[j];
        else
            mean[j] = 0.0;
    }

    for (int j = 0; j < cols; j++) // Imputar NAN con la media
        for (int i = 0; i < rows; i++)
            if (isnan(data->data[i][j]))
                data->data[i][j] = mean[j];

    for (int j = 0; j < cols; j++) // Recalcular min y max tras imputación
    {
        min[j] = DBL_MAX;
        max[j] = -DBL_MAX;
        for (int i = 0; i < rows; i++)
        {
            double v = data->data[i][j];

            if (v < min[j]) // Actualizar mínimo
                min[j] = v;

            if (v > max[j]) // Actualizar máximo
                max[j] = v;
        }
    }

    for (int j = 0; j < cols; j++) // Normalizar Min-Max
    {
        double range = max[j] - min[j];

        if (range == 0)
            range = 1.0;

        for (int i = 0; i < rows; i++)
            data->data[i][j] = (data->data[i][j] - min[j]) / range;
    }

    free(min);
    free(max);
    free(mean);
    free(count);
}