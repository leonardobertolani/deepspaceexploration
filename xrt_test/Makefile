CC := /usr/bin/gcc

CFLAGS := -std=gnu11 -Wno-incompatible-pointer-types -I /usr/include/xrt/ 
LDFLAGS := -lxrt_core -lxrt_coreutil

BUILD_DIR := build

.PHONY: all clean 

all: time_bench 

time_bench: $(BUILD_DIR)
	$(CC) src/main.c src/header.h src/header.c -o build/time_bench $(CFLAGS) $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
