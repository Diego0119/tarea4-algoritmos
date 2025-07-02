
CC=gcc
EXEC=program.out
GRUPO=G4
NTAR=3

SRC_DIR=src
OBJ_DIR=obj
BUILD_DIR=build
SRC_FILES=$(wildcard $(SRC_DIR)/*.c)
OBJ_FILES=$(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))
INCLUDE=-I./incs/
LIBS= -lm

CFLAGS=-Wall -Wextra -Wpedantic -O3 -g -Wno-stringop-truncation
LDFLAGS= -Wall -lm -g

all: folders $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/$(EXEC) $(OBJ_FILES) $(INCLUDE) $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^ $(INCLUDE)

.PHONY: clean folders send run run-val

clean:
	rm -f $(OBJ_FILES)
	rm -f build/$(EXEC)

folders:
	mkdir -p src obj incs build docs data

send:
	tar czf $(GRUPO)-$(NTAR).tgz --transform 's,^,$(GRUPO)-$(NTAR)/,' Makefile src incs docs

run-all: all
	./$(BUILD_DIR)/$(EXEC) -h
	./$(BUILD_DIR)/$(EXEC) -v
	./$(BUILD_DIR)/$(EXEC) -k ./data/iris.csv 1 
	./$(BUILD_DIR)/$(EXEC) -l ./data/iris.csv 0.01 2000 1e-8
	./$(BUILD_DIR)/$(EXEC) -m ./data/iris.csv 3 100 1e-4

run-all-val: all
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./$(BUILD_DIR)/$(EXEC) -h
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./$(BUILD_DIR)/$(EXEC) -v
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./$(BUILD_DIR)/$(EXEC) -k ./data/iris.csv 1
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./$(BUILD_DIR)/$(EXEC) -l ./data/iris.csv 0.01 2000 1e-8
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./$(BUILD_DIR)/$(EXEC) -m ./data/iris.csv 3 100 1e-4

run-knn:
	./$(BUILD_DIR)/$(EXEC) -k ./data/iris.csv 1 
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./$(BUILD_DIR)/$(EXEC) -k ./data/iris.csv 1

run-lr:
	./$(BUILD_DIR)/$(EXEC) -l ./data/iris.csv 0.01 2000 1e-8
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./$(BUILD_DIR)/$(EXEC) -l ./data/iris.csv 0.01 2000 1e-8

run-km:
	./$(BUILD_DIR)/$(EXEC) -k ./data/iris.csv 3 100 1e-4
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./$(BUILD_DIR)/$(EXEC) -m ./data/iris.csv 3 100 1e-4