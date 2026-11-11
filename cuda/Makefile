NVCC ?= nvcc
BUILD_DIR ?= ../target/cuda
ARCH ?= native

.PHONY: check ptx clean

check: $(BUILD_DIR)/fold $(BUILD_DIR)/fold.ptx
	$(BUILD_DIR)/fold

ptx: $(BUILD_DIR)/fold.ptx

$(BUILD_DIR)/fold: fold.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) --std=c++17 -O3 -arch=$(ARCH) $< -o $@

$(BUILD_DIR)/fold.ptx: fold.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) --std=c++17 -O3 -arch=$(ARCH) --ptx $< -o $@

clean:
	rm -f $(BUILD_DIR)/fold $(BUILD_DIR)/fold.ptx
