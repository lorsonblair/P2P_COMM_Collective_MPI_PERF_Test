/****************************************************************
 * Lorson Blair					
 * 03/29/2020				
 * Assignment3			
 * 			
 * This is the Main file. it performs all the MPI tasks and
 * calls the gol_kernelLaunch function to compute each new
 * iteration of the worlds.
 *			
 ***************************************************************/  

#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <stdio.h>
#include <mpi.h>

/* Definition and initialization of global (extern) variables */
unsigned char *g_data = NULL;
unsigned char *g_resultData = NULL;
unsigned char *top_ghost_row = NULL;
unsigned char *bottom_ghost_row = NULL;

/* Definition of extern functions. These functions are declared in gol-with-cuda.cu */
// Calculates the number of Cuda blocks
extern ushort calculateBlocks(size_t worldWidth, size_t worldHeight, ushort threadsCount);

// Computes the world using the CUDA kernel
extern void gol_kernelLaunch(unsigned char ** d_data, unsigned char ** d_resultData, unsigned char ** top_ghost, 
                             unsigned char ** bottom_ghost, size_t worldWidth, size_t worldHeight, ushort blocks, 
                             ushort threadsCount);

// Initializes the worlds
extern void gol_initMaster(unsigned int pattern, int my_rank, int num_ranks, size_t worldWidth, size_t worldHeight);

extern void gol_printWorld(int my_rank);	// Prints each rank's world (chunk) to output files
extern void freeMemory();			        // Frees memory allocated for myrank's world and the ghost rows

/* Main Function */
int main(int argc, char *argv[])
{
    unsigned int pattern = 0, 
                 worldSize = 0,			 
		         iterations = 0,	// Total number of iterations 
		         printFlag;         // Used to check whether or not to print the output         
    ushort threads;         // Number of threads to use

    int myrank, numranks;           		// Rank's Id and the number of ranks
    double startTime, endTime, totalTime;	// Used to calculate program execution time

    // Check for the correct number of arguments 
    if( argc != 6 )
    {
        printf("GOL requires 5 arguments: pattern number, sq size of the world, # of itterations, # of threads-per-block, ");
        printf("and output on/off (0 is output off, 1 is on): e.g. ./gol 0 32 2 256 0\n");
	    exit(-1);
    }

    // Assign arguments 
    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    threads = atoi(argv[4]);
    printFlag = atoi(argv[5]);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get current rank and number of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    // Start timer and print information
    if (myrank == 0)
    {
        startTime = MPI_Wtime();
        printf("\n###################################################################################################\n\n");
        printf("This is the Game of Life running in parallel using CUDA and MPI.\n");
    }

    // Initialize world
    gol_initMaster(pattern, myrank, numranks, worldSize, worldSize);
    
    // Checks for receives and sends
    MPI_Request recvReq[2], sendReq[2];
    MPI_Status status;
    
    // number of CUDA blocks to us
    ushort numBlocks = calculateBlocks(worldSize, worldSize, threads);
    //printf("Main: # of blocks: %d\n", numBlocks);
    
    // MPI iterations. Data is sent and received between ranks
    int iter;  
    for (iter = 0; iter < iterations; iter++)
    {
        // fill myrank's top_ghost_row with the last row received from the previous rank
        MPI_Irecv(top_ghost_row, worldSize, MPI_UNSIGNED_CHAR, ((myrank + numranks - 1) % numranks), 0, MPI_COMM_WORLD, &recvReq[0]);
            
        // fill my_rank's bottom_ghost_row with the first row received from the next rank
        MPI_Irecv(bottom_ghost_row, worldSize, MPI_UNSIGNED_CHAR, ((myrank + 1) % numranks), 1, MPI_COMM_WORLD, &recvReq[1]);

        // send myrank's first row to the bottom_ghost_row of the previous rank 
        MPI_Isend(&g_data[0], worldSize, MPI_UNSIGNED_CHAR, ((myrank + numranks - 1) % numranks), 1, MPI_COMM_WORLD, &sendReq[0]);

        // send myrank's last row to the top_ghost_row of the next rank
        MPI_Isend(&g_data[worldSize * (worldSize - 1)], worldSize, MPI_UNSIGNED_CHAR, ((myrank + 1) % numranks), 0, MPI_COMM_WORLD, &sendReq[1]);

        // wait for information to be received before computing the next generation.
        MPI_Wait(&recvReq[0], &status);
        MPI_Wait(&recvReq[1], &status);
        MPI_Wait(&sendReq[0], &status);             
        MPI_Wait(&recvReq[1], &status);     

        // compute next generation 
        gol_kernelLaunch(&g_data, &g_resultData, &top_ghost_row, &bottom_ghost_row, worldSize, worldSize, numBlocks, threads); 
    }
    
    /* After iterations. Stops timer and calculate and print total execution time. */ 
    if (myrank == 0) 
    {
        endTime = MPI_Wtime();
        totalTime = endTime - startTime;
	    printf("Total Execution Time: %.3fs\n", totalTime);
	    printf("If output is turned on, the worlds for each rank are printed to their corressponding .txt files.\n\n");
        printf("###################################################################################################\n\n");     
    }

    // If print flag is set, print rank's world to file
    if (printFlag)
    {
        gol_printWorld(myrank);
        //printGhostRow();
    }

    /* End MPI and free allocated memory. */ 
    MPI_Finalize();
    freeMemory();

    return 0;
}
