TARGET = nano_vllm
CC = gcc

# Flags for CPU
CFLAGS = -O3 -Wall -Iinclude
LDFLAGS = -lm

# Conditional OpenMP for CPU
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # Use Homebrew libomp
    # Check if brew exists and find prefix, otherwise fallback or assume standard paths
    BREW_PREFIX := $(shell command -v brew >/dev/null 2>&1 && brew --prefix)
    ifneq ($(BREW_PREFIX),)
        CFLAGS += -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
        LDFLAGS += -L$(BREW_PREFIX)/opt/libomp/lib -lomp
    else
        # Fallback for standard Mac without brew or if libomp is elsewhere
        # User might need to adjust this if using Apple Clang without OpenMP
    endif
else
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
endif

# Sources
SRCS_C = src/main.c src/model.c src/tokenizer.c src/log.c src/sampler.c src/memory.c src/visualizer.c
SRCS_KERNELS = src/kernels/layers.c src/kernels/attention.c

# Objects
OBJS_C = $(SRCS_C:.c=.o)
OBJS_KERNELS = $(SRCS_KERNELS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS_C) $(OBJS_KERNELS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

setup:
	./setup_models.sh

clean:
	rm -f $(TARGET) src/*.o src/kernels/*.o *.log
