#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>


#define NUM_BLOCKS 64
#define FREQUENCY 512000000

/****************************************************************/
// FOR POWER9 SYSTEMS ONLY - x86 SYSTEMS HAVE A DIFFERENT CODE  //
/****************************************************************/

typedef unsigned long long ticks;

static __inline__ ticks getticks(void)
{
  unsigned int tbl, tbu0, tbu1;

  do {
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  return (((unsigned long long)tbu0) << 32) | tbl;
}

int main(int argc, char **argv)
{
    int myrank, nprocs, block_size;
    
    char file_name[64];

    unsigned long long start = 0;
    unsigned long long finish = 0;
    unsigned long long result = 0;
    float time = 0.0;
    
    if (argc != 2)
    {
      printf("Block size required\n");
      return (EXIT_FAILURE);
    }

    block_size = atoi(argv[1]);
    char *buffer = malloc(block_size);

    MPI_File file;
    MPI_Offset offset;
    MPI_Status status;
    
    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);     // my rank. current rank
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);     // number of ranks (processes)
    
    // fill the buffer with all 1s
    for (int i = 0; i < block_size; i++) buffer[i] = '1';

    sprintf(file_name, "%d_ranks_%d_blksz", nprocs, block_size);
    
    if (myrank == 0) 
    {
        printf("***********************************\n");
        printf("Parallel IO Tests for %s\n", file_name);
	printf("Write Test\n");
        start = getticks();
    }
    
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    // perform NUM_BLOCKS writes 
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        offset = (myrank * block_size) + (i * nprocs * block_size);
        MPI_File_write_at(file, offset, buffer, block_size, MPI_CHAR, &status);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&file);
    
    // stop timer and print result
    if (myrank == 0)
    {
        finish = getticks();
        result = finish - start;
	time = (float)result / FREQUENCY;
	printf("Start Ticks: %llu\n", start);
	printf("Finish Ticks: %llu\n", finish); 
        printf("Result: %llu\n", result);
	  
        printf("Write Time (s): %.3f\n", time);
    } 

    // start timer for reads
    if (myrank == 0) 
    {
        printf("\nRead Test\n");
        start = getticks();
    }

    // perform NUM_BLOCKS reads
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        offset = (myrank * block_size) + (i * nprocs * block_size);
        MPI_File_read_at(file, offset, buffer, block_size, MPI_CHAR, &status);    
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&file);

    // stop timer and print result
    if (myrank == 0)
    {
        finish = getticks();
        result = finish - start;
	time = (float)result / FREQUENCY;
	printf("Start Ticks: %llu\n", start);
	printf("Finish Ticks: %llu\n", finish); 
        printf("Result: %llu\n", result);
	printf("Read Time (s): %.3f\n", time);

        printf("***********************************\n");
    }
    
    // clean up
    MPI_Finalize();
    free(buffer);
    return 0;
}
