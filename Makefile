CC = gcc
CFLAGS = -Wall -Wextra -g -Iincs

SRC_DIR = src
OBJ_DIR = obj
BUILD_DIR = build

SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC))

TARGET = $(BUILD_DIR)/program

all: folders $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

folders:
	mkdir -p $(OBJ_DIR) $(BUILD_DIR)

clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

run:
	cd build && ./program --help

