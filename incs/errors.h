/*
 * @file: errors.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Cabecera general de errores.
 */

#ifndef ERRORS_H
#define ERRORS_H

#include "csv.h"
#include "matrix.h"
#include "knn.h"

// Funciones de manejo de errores
void handle_error(const char *, const char *, const char *, int);
void number_arguments_error(const char *, int);
void argument_error(const char *, const char *, int);
void open_file_error(const char *, int, const char *);
void read_csv_error(const char *, int, const char *);
void dimensions_error(const char *, int, const char *);
void csv_struct_error(const char *, int, CSVData *);
void matrix_struct_error(const char *, int, Matrix *);
void csv_extension_error(const char *, int, const char *);
void train_test_split_error(const char *, int);
void create_knn_classifier_error(const char *, int, Matrix *, Matrix *, Matrix *, Matrix *, KNNClassifier *);
void predict_knn_error(const char *, int, Matrix *, Matrix *, Matrix *, Matrix *, KNNClassifier *);
void k_parameter_error(const char *, int);

#endif
