CC = gcc
NVCC = nvcc
CUDA_PATH ?= /usr/local/cuda

# CPU Config
# Check if we are on Mac (Darwin) to handle OpenMP differently
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # Mac (Clang) usually needs libomp installed (brew install libomp)
    # For simplicity in standard envs without libomp, we can disable OpenMP or try to detect it.
    # Let's just remove -fopenmp for now to ensure it compiles out-of-the-box on standard Mac.
    # If user has libomp, they can add it back.
    CFLAGS = -O3 -Wall -Iinclude
    LDFLAGS_CPU = -lm
else
    # Linux (GCC)
    CFLAGS = -O3 -Wall -Iinclude -fopenmp
    LDFLAGS_CPU = -lm -lgomp
endif

# GPU Config
NVCC_FLAGS = -O3 -Iinclude -arch=native
LDFLAGS_GPU = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lm

# Common Sources
SRCS_C = src/main.c src/model.c src/tokenizer.c src/log.c src/sampler.c
OBJS_C = $(SRCS_C:.c=.o)

# CPU Sources
SRCS_CPU = kernels/cpu/kernels.c kernels/cpu/visualizer.c
OBJS_CPU = $(SRCS_CPU:.c=.o)

# GPU Sources
SRCS_GPU = kernels/gpu/layers.cu kernels/gpu/attention.cu
OBJS_GPU = $(SRCS_GPU:.cu=.o)

TARGET_CPU = nano_vllm_cpu
TARGET_GPU = nano_vllm_gpu

# Default target
cpu: $(TARGET_CPU)

gpu: $(TARGET_GPU)

$(TARGET_CPU): $(OBJS_C) $(OBJS_CPU)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS_CPU)

$(TARGET_GPU): $(OBJS_C) $(OBJS_GPU)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS_GPU)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f src/*.o kernels/cpu/*.o kernels/gpu/*.o $(TARGET_CPU) $(TARGET_GPU)

.PHONY: cpu gpu clean
