#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define K 1000
#define M 1000000
#define NUM_BLOCKS 64

#define BLOCK_SIZE 12 * K

int main(int argc, char **argv)
{
    int myrank, nprocs, block_size;
    char *buffer = malloc(BLOCK_SIZE);

    MPI_File file;
    MPI_Offset offset;
    MPI_Status status;
    
    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);     // my rank. current rank
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);     // number of ranks (processes)
    
    //offset = myrank * BLOCK_SIZE;               // start of the view for each processor 
    //count = BLOCK_SIZE;             // count is basically the block size
    
    // fill the buffer with all 1s
    for (int i = 0; i < BLOCK_SIZE; i++) buffer[i] = '1';
    
    // perform NUM_BLOCKS writes 
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        offset = i * BLOCK_SIZE;
        MPI_File_open(MPI_COMM_WORLD, "datafile.out", MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        MPI_File_write_at(file, offset, buffer, BLOCK_SIZE, MPI_CHAR, &status);
        MPI_File_close(&file);
    }

    // perform NUM_BLOCKS reads
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        MPI_File_open(MPI_COMM_WORLD, "datafile.out", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        MPI_File_read_at(file, offset, buffer, BLOCK_SIZE, MPI_CHAR, &status);    
        MPI_File_close(&file);
    }
    
    MPI_Finalize();
    free(buffer);
    return 0;
    
}
