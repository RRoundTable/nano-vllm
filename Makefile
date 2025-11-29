CC = gcc
NVCC = nvcc
CUDA_PATH ?= /usr/local/cuda

CFLAGS = -O3 -Wall -Iinclude -I$(CUDA_PATH)/include
NVCC_FLAGS = -O3 -Iinclude -arch=native

LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lm

# Source files
SRCS_C = src/main.c src/model.c
SRCS_CU = kernels/layers.cu kernels/attention.cu

# Object files
OBJS_C = $(SRCS_C:.c=.o)
OBJS_CU = $(SRCS_CU:.cu=.o)

TARGET = nano_vllm

all: $(TARGET)

$(TARGET): $(OBJS_C) $(OBJS_CU)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS_C) $(OBJS_CU) $(TARGET)

.PHONY: all clean

