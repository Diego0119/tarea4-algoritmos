/*
 * @file: config.h
 * @authors: Miguel Loaiza, Diego Sanhueza y Duvan Figueroa
 * @date: 17/06/2025
 * @description: Cabecera general de configuración.
 */

#ifndef CONFIG_H
#define CONFIG_H

#define RED_COLOR "\033[0;31m"
#define GREEN_COLOR "\033[0;32m"
#define CYAN_COLOR "\033[0;36m"
#define YELLOW_COLOR "\033[0;33m"
#define BRIGHT_PURPLE_COLOR "\033[1;35m"
#define RESET_COLOR "\033[0m"

#define N_OF_CHARS 256       // Número máximo de caracteres en una palabra
#define MAX_LINE_LENGTH 4096 // Longitud máxima de una línea en el CSV
#define MAX_FIELDS 256       // Número máximo de campos en una línea del CSV
#define MAX_CLASSES 10       // Número máximo de clases en el KNN

#endif
