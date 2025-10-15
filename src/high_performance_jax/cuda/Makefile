NVCCFLAGS = -g -G

all: build

build: vector_add_simple.out vector_add.out

vector_add_simple.out: vector_add_simple.cu
	nvcc $(NVCCFLAGS) -o vector_add_simple.out vector_add_simple.cu

vector_add.out: vector_add.cu
	nvcc $(NVCCFLAGS) -o vector_add.out vector_add.cu

clean:
	rm -f vector_add.out
	rm -f vector_add_simple.out

run: build
	./vector_add.out
	./vector_add_simple.out