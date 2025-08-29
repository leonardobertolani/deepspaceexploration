# Compiler
CC := /usr/bin/gcc

# Compiler flags
CFLAGS := -std=gnu11 -Wno-incompatible-pointer-types -Wno-unused-variable -Wno-unused-but-set-variable -I /usr/include/xrt/ -O0
LDFLAGS := -lxrt_core -lxrt_coreutil

# Directories
SRC_DIR := src
BUILD_DIR := build

# Source files
SRC := $(wildcard $(SRC_DIR)/*.c)
OBJ := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRC))

# Target executable
TARGET := main

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) -c $< -o $@ $(CFLAGS)

$(BUILD_DIR):
	mkdir -p $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
