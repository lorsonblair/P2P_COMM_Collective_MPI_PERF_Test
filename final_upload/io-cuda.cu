/************************************************************/
// Cuda function to allocate space for the file using 
// CudaMallocManaged. This file is used with io-main.c 
// 05/01/2020
/***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern long long *buffer;
extern long long block_size; 

/* Initialize the buffer with all 1s */
extern "C" void initialize(long long blk_sz)
{
    cudaMallocManaged(&buffer, block_size * sizeof(long long));
    
    for (long long int i = 0; i < blk_sz; i++) 
    {
        buffer[i] = '1';
    }
}

/* Frees memory allocated */
extern "C" void freeMemory()
{
    cudaFree(buffer);
}