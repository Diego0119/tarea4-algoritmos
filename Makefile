
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

run: all
	./$(BUILD_DIR)/$(EXEC) 

clean:
	rm -f $(OBJ_FILES)
	rm -f build/$(EXEC)

folders:
	mkdir -p src obj incs build docs

send:
	tar czf $(GRUPO)-$(NTAR).tgz --transform 's,^,$(GRUPO)-$(NTAR)/,' Makefile src incs docs

run-val: all
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./$(BUILD_DIR)/$(EXEC)