# C++ Compiler
CC = g++
CFLAGS = -g -Wall
OPENCV = opencv
LDFLAGS = `pkg-config --libs --cflags $(OPENCV)`

# CUDA Compiler
CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_61,code=compute_61
        
# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname),Darwin)
	LDFLAGS       := $(LDFLAGS) -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft
	CCFLAGS   	  := -arch $(OS_ARCH)
else
	ifeq ($(OS_SIZE),32)
		LDFLAGS   := $(LDFLAGS) -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS   := -m32
	else
		CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
		LDFLAGS       := $(LDFLAGS) -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS       := -m64
	endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCCFLAGS := -m32
else
	NVCCFLAGS := -m64
endif

# Folders 
BINDIR = bin
OBJDIR = obj
SRCDIR = src

# Files
SRC_CPP = $(wildcard $(SRCDIR)/*.cpp)
SRC_CUDA = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(SRC_CPP:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o) $(SRC_CUDA:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/wfc


.PHONY: all
all: build

.PHONY: clean
clean:
	@echo "Cleaning ..."
	@rm -rf $(OBJDIR)
	@rm -rf $(BINDIR)
	@rm -rf results

.PHONY: dirs
dirs:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(BINDIR)
	@mkdir -p results

.PHONY: build
build: dirs $(TARGET)

.PHONY: test
test:
	bin/wfc tiles/red/ 2 1 1 4 4 0 0 red
	bin/wfc tiles/spirals/ 3 1 1 4 4 0 0 spirals
	bin/wfc tiles/bricks/ 3 0 1 4 4 0 0 bricks
	bin/wfc tiles/dungeons/ 3 0 1 4 4 0 0 dungeons
	bin/wfc tiles/paths/ 3 0 1 4 4 0 0 paths

$(OBJDIR)/%.o: $(SRCDIR)/%.cu 
	@echo "Compiling cuda objects: $@"
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $< $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp 
	@echo "Compiling cpp objects: $@"
	$(CC) $(CFLAGS) -MP -MMD -c $< -o $@ $(LDFLAGS)

$(TARGET): $(OBJECTS)
	@echo "Linking: $@"
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
