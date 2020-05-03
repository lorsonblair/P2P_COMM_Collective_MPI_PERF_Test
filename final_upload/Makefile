all:common-cuda-lib \
	io-main \
	parallelio

common-cuda-lib: io-cuda.cu
	nvcc -g -G -c -arch=sm_70 io-cuda.cu \
		-o io-cuda.o

io-main: io-main.c
	mpicc -g -c io-main.c \
		-o io-main.o 
	mpicc -g io-cuda.o io-main.o \
		-o io-main-exe \
		-L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++

parallelio: parallelio.c
	mpicc -g parallelio.c \
		-o parallelio-exe \
		-L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++

clean:
	rm -rf *.o *-exe