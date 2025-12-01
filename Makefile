TARGET_CPU = nano_vllm_cpu
TARGET_GPU = nano_vllm_gpu
CC = gcc
NVCC = nvcc

# Flags for Mac (CPU)
CFLAGS = -O3 -Wall -Iinclude
LDFLAGS_CPU = -lm

# Conditional OpenMP for CPU
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # Mac Default Clang doesn't support -fopenmp easily
else
    CFLAGS += -fopenmp
    LDFLAGS_CPU += -fopenmp
endif

# Flags for CUDA
NVCC_FLAGS = -O3 -arch=sm_70 -Iinclude

# Sources
SRCS_C = src/main.c src/model.c src/tokenizer.c src/log.c src/sampler.c src/memory.c
SRCS_CU = kernels/gpu/layers.cu kernels/gpu/attention.cu
SRCS_CPU_KERNELS = kernels/cpu/kernels.c kernels/cpu/visualizer.c

# Objects
OBJS_C = $(SRCS_C:.c=.o)
OBJS_CPU_KERNELS = $(SRCS_CPU_KERNELS:.c=.o)

all: $(TARGET_CPU)

# CPU Build
.PHONY: cpu
cpu:
	$(MAKE) clean
	$(MAKE) $(TARGET_CPU)

$(TARGET_CPU): $(OBJS_C) $(OBJS_CPU_KERNELS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS_CPU)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# GPU Build (requires nvcc)
gpu: $(TARGET_GPU)

$(TARGET_GPU): $(SRCS_C) $(SRCS_CU)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lcublas

setup:
	./setup_models.sh

clean:
	rm -f $(TARGET_CPU) $(TARGET_GPU) src/*.o kernels/cpu/*.o *.log
