/************************************************************************
 * Lorson Blair                                                 	   
 * 03/29/2020                                                   	   
 * Assignment3                                                  	   
 *                                                             	   
 * In this assignment, I implemented Conways Game of Life using CUDA 
 * and MPI. This is the CUDA file. It performs all the CUDA tasks. I
 * reused the template from Assignment 2 and made the following adjustments:
 * 1. The gol_kernel function was modified to account for the lack of	
 *    top and bottom world wrapping. 					
 * 2. The initialization functions of patterns 2, 3, 4, and the		
 *    init_master function  were modified to account for the 		
 *    different ranks. 							
 * 3. The printWorld function was modified to print the rank's worlds 	
 *    to files.								
 * 4. A function to calculate the number of CUDA blocks was added.  	
 * 5. No iterations were needed in the gol_KernelLaunch function. MPI 
 *    handled the iterations.
 *			
 * The initial template was provided by Dr. Carothers

************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
 
#define MAX_BLOCKS 65535    // maximum number of blocks supported by CUDA
#define min(num1, num2) (((num1) < (num2)) ? (num1) : (num2))   // used to cap the number of blocks used to the MAX_BLOCKS

/* Declaration of global (extern) variables. */ 
// Current state of world. 
extern unsigned char *g_data;
 
// Result from last compute of world.
extern unsigned char *g_resultData;
 
// ghost row above
extern unsigned char *top_ghost_row;
 
// ghost row below
extern unsigned char *bottom_ghost_row;
 
// Current width of world.
size_t g_worldWidth = 0;
 
// Current height of world.
size_t g_worldHeight = 0;
 
// Current data length (product of width and height)
// g_worldWidth * g_worldHeight
size_t g_dataLength = 0; 
 
/* Initialization function for Pattern 0 */
static inline void gol_initAllZeros( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;
 
    // Allocating memory for g_data and g_resultData, top_ghost_row, and bottom_ghost_row  
    cudaMallocManaged( &g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged( &g_resultData, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged( &top_ghost_row, g_worldWidth * sizeof(unsigned char));
    cudaMallocManaged( &bottom_ghost_row, g_worldWidth * sizeof(unsigned char));
     
    // Initialize all elements to 0
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(top_ghost_row, 0, g_worldWidth * sizeof(unsigned char));
    cudaMemset(bottom_ghost_row, 0, g_worldWidth * sizeof(unsigned char)); 
}

/* Initialization function for Pattern 0 */
static inline void gol_initAllOnes( size_t worldWidth, size_t worldHeight )
{
    int i;
     
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;
 
    // Allocating memory for g_data and g_resultData, top_ghost_row, and bottom_ghost_row
    cudaMallocManaged( &g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged( &g_resultData, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged( &top_ghost_row, g_worldWidth * sizeof(unsigned char));
    cudaMallocManaged( &bottom_ghost_row, g_worldWidth * sizeof(unsigned char));
 
    // Initialize all elements to 0
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(top_ghost_row, 0, g_worldWidth * sizeof(unsigned char));
    cudaMemset(bottom_ghost_row, 0, g_worldWidth * sizeof(unsigned char));
         
    // Set all rows of world to true
    for( i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 1;
    }
}

/* Initialization function for Pattern 2. A streak of 10 ones in the last row 
   of each MPI rank, starting at column 128. */ 
static inline void gol_initOnesInMiddle( size_t worldWidth, size_t worldHeight )
{
    int i;
     
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;
 
    // Allocating memory for g_data and g_resultData, top_ghost_row, and bottom_ghost_row
    cudaMallocManaged( &g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged( &g_resultData, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged( &top_ghost_row, g_worldWidth * sizeof(unsigned char));
    cudaMallocManaged( &bottom_ghost_row, g_worldWidth * sizeof(unsigned char));
     
    // initialize all elements to 0
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(top_ghost_row, 0, g_worldWidth * sizeof(unsigned char));
    cudaMemset(bottom_ghost_row, 0, g_worldWidth * sizeof(unsigned char)); 
        
    // Last row 
    for( i = (g_worldHeight - 1) * g_worldWidth; i < g_worldHeight * g_worldWidth; i++)
    {
        // Column 128. Row and Column numbers start at 0.
        if( (i >= ( (g_worldHeight - 1) * g_worldWidth + 128)) && (i < ((g_worldHeight - 1) * g_worldWidth + 138)))
        {
           g_data[i] = 1;
        }
    }
 
   /*// Used to test the initialization on small world sizes
    for( i = (g_worldHeight-1)*g_worldHeight; i < g_worldHeight*g_worldHeight; i++)
    {
        if( (i >= ( ((g_worldHeight-1)*g_worldWidth) + 10)) && (i < (((g_worldHeight-1)*g_worldWidth) + 20)))
        {
            g_data[i] = 1;
        }
    }*/
}
 
/* Initialization function for Pattern 3. The corners are the upper left and upper right cells of  
   of rank 0, and the lower left and lower right cells of the last rank. */
static inline void gol_initOnesAtCorners( int my_rank, int num_ranks, size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Allocating memory for g_data and g_resultData, top_ghost_row, and bottom_ghost_row
    cudaMallocManaged( &g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged( &g_resultData, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged( &top_ghost_row, g_worldWidth * sizeof(unsigned char));
    cudaMallocManaged( &bottom_ghost_row, g_worldWidth * sizeof(unsigned char));
     
    // Initialize all elements to 0
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(top_ghost_row, 0, g_worldWidth * sizeof(unsigned char));
    cudaMemset(bottom_ghost_row, 0, g_worldWidth * sizeof(unsigned char));
     
    // Set the top corners of the first rank
    if (my_rank == 0)
    {
        g_data[0] = 1; // upper left
        g_data[worldWidth - 1]=1; // upper right
    }
    // Set the bottom corners of the last rank
    if (my_rank == num_ranks - 1)
    {
        g_data[((worldHeight-1) * (worldWidth))]=1; // lower left
        g_data[((worldHeight-1) * worldWidth) + (worldWidth-1)]=1; // lower rigiht
    }
}

/* Initialization function for Patter 4. Only the first rank does the initialization. */ 
static inline void gol_initSpinnerAtCorner( int my_rank, size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;
 
    // Allocating memory for g_data and g_resultData, top_ghost_row, and bottom_ghost_row
    cudaMallocManaged( &g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged( &g_resultData, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged( &top_ghost_row, g_worldWidth * sizeof(unsigned char));
    cudaMallocManaged( &bottom_ghost_row, g_worldWidth * sizeof(unsigned char));
 
    // initialize all elements to 0
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(top_ghost_row, 0, g_worldWidth * sizeof(unsigned char));
    cudaMemset(bottom_ghost_row, 0, g_worldWidth * sizeof(unsigned char));
         
    // set the spinners to true
    if (my_rank == 0)
    {
        g_data[0] = 1; // upper left
        g_data[1] = 1; // upper left +1
        g_data[worldWidth - 1] = 1; // upper right
    }
}
 
/* Master initialization function. Contains the CUDA device setup and initialization. */
extern "C" void gol_initMaster( unsigned int pattern, int my_rank, int num_ranks, size_t worldWidth, size_t worldHeight )
{
    // CUDA device setup and initialization 
    int cE, cudaDeviceCount;	
         
    if ((cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess)
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n",
                cE, cudaDeviceCount );
        exit(-1);
    }
    if ((cE = cudaSetDevice(my_rank % cudaDeviceCount)) != cudaSuccess)
    {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
                my_rank, (my_rank % cudaDeviceCount), cE);
        exit(-1);
    }
 
    switch(pattern)
    {
        case 0:
        gol_initAllZeros( worldWidth, worldHeight );
        break;
     
        case 1:
        gol_initAllOnes( worldWidth, worldHeight );
        break;
     
        case 2:
        gol_initOnesInMiddle( worldWidth, worldHeight );
        break;
     
        case 3:
        gol_initOnesAtCorners( my_rank, num_ranks, worldWidth, worldHeight );
        break;
 
        case 4:
        gol_initSpinnerAtCorner( my_rank, worldWidth, worldHeight );
        break;
 
        default:
        printf("Pattern %u has not been implemented \n", pattern);
        exit(-1);
    }
}
 
/* Function to swap the pointers of pA and pB */
void swap( unsigned char **pA, unsigned char **pB)
{
    unsigned char * temp = *pA;
    *pA = *pB;
    *pB = temp;
}
 
/* CUDA kernel function. Parallel version of the Game-of-Life. */
__global__ void gol_kernel(const unsigned char* d_data, unsigned char* d_resultData, 
                                     const unsigned char *top_ghost, const unsigned char *bottom_ghost,     
                                     unsigned int worldWidth, unsigned int worldHeight)
{
    unsigned int index; 
    unsigned int l_worldSize = worldWidth * worldHeight;
    unsigned int stride = blockDim.x * gridDim.x;

    unsigned int column, row, column0, column1, column2, row0, row1, row2, live_nbors;
    
    for (index = blockIdx.x * blockDim.x + threadIdx.x; index < l_worldSize; index += stride)
    {
        //printf("index: %d\n", index);

        // calculate the current row and column number 
        column = index % worldWidth;            
        row = (index - column) / worldWidth;     
        
        /* Calculate the locators of the current cell and its neighbors. Rows do not wrap around. 
           Locators are in the ghost rows if our current row is the first or the last row. */
        row0 = (row - 1) * worldWidth;  
        row1 = row * worldWidth;        
	    row2 = (row + 1) * worldWidth;  
                
        // Calculate the column locators 
        column1 = column;                                     
        column0 = (column1 + worldWidth - 1) % worldWidth;  
        column2 = (column1 + 1) % worldWidth;    

        /* Calculate the number of live cells (neighbors). Since there is no top bottom 
           wrapping, we have to check top and bottom ghosts rows for live neighbors if 
           we are at the first or last row, respectively. */
        live_nbors = 0;
	
        /* If current row is the first row, check top ghost row for live neighbors and calculate
           total. */
        if (index < worldWidth) 
        {
            live_nbors += top_ghost[column0];	// upper left neigbor
            live_nbors += top_ghost[column1];	// upper middle neighbor
            live_nbors += top_ghost[column2];	// upper right neighbor
            live_nbors += d_data[row1 + column0];
            live_nbors += d_data[row2 + column0];
            live_nbors += d_data[row2 + column1];
            live_nbors += d_data[row1 + column2];
            live_nbors += d_data[row2 + column2];
        }
        /* If current row is the last row, check bottom ghost row for live neighbors and calculate
           total. */
        else if (index >= (worldWidth - 1) * worldWidth)
        {
            live_nbors += d_data[row0 + column0];
            live_nbors += d_data[row1 + column0];
            live_nbors += d_data[row0 + column1];
            live_nbors += d_data[row0 + column2];
            live_nbors += d_data[row1 + column2];
            live_nbors += bottom_ghost[column0];	// bottom left neighbor
            live_nbors += bottom_ghost[column1];	// bottom middle neighbor
            live_nbors += bottom_ghost[column2];	// bottom right neighbor
        }
        // If current row is a middle row, check for live neighbors and calculate as normal 
        else
        {
            live_nbors += d_data[row0 + column0];
            live_nbors += d_data[row1 + column0];
            live_nbors += d_data[row2 + column0];
            live_nbors += d_data[row0 + column1];
            live_nbors += d_data[row2 + column1];
            live_nbors += d_data[row0 + column2];
            live_nbors += d_data[row1 + column2];
            live_nbors += d_data[row2 + column2];
        }
         
        /* Rules of life. Tells what should happen on the next iteration.
           Update the cell based on the rules of life. */
        d_resultData[row1 + column1] = (live_nbors == 3 || (live_nbors == 2 && d_data[row1 + column1]) ? 1 : 0);
    }
}
 
/* Global function to calculate the number of blocks to use. If the number of blocks calculated is greater than than MAX_BLOCKS, 
   the maximum number of blocks allowed by CUDA, we use MAX_NUM_BLOCKS. */
extern "C" ushort calculateBlocks(size_t worldWidth, size_t worldHeight, ushort threadsCount)
{
    size_t l_worldSize = worldWidth * worldHeight;   
    size_t numBlocksReq = (l_worldSize / threadsCount) + (l_worldSize % threadsCount != 0);  
    ushort num_blocks = (ushort)min((size_t)MAX_BLOCKS, numBlocksReq);
    //printf("numBlocksReq: %ld\n", numBlocksReq);
    //printf("Blocks: %d\n", numBlocks);
    return num_blocks;        // return the number of blocks
}

/* Global function to compute the worlds via the CUDA kernel. This function also  swaps the new world with the previous world to be 
   to be ready for the next iteration. */
extern "C" void gol_kernelLaunch(unsigned char ** d_data, unsigned char ** d_resultData, unsigned char **top_ghost, 
                                 unsigned char **bottom_ghost, size_t worldWidth, size_t worldHeight, 
                                 ushort blocks, ushort threadsCount)
{
    // Perform the parallel compution.
    gol_kernel<<<blocks, threadsCount>>>(*d_data, *d_resultData, *top_ghost, *bottom_ghost, worldWidth, worldHeight);
    cudaDeviceSynchronize();	// CPU to waits for the kernel to finish 
    swap(d_data, d_resultData);	// Swap the world for the next iteration     
}

/* Global function to print the worlds for each rank to file */
extern "C" void gol_printWorld(int my_rank)
{
    char file[10];
    sprintf(file, "Rank%d.txt", my_rank);
    FILE *output = fopen(file, "wb");
 
    int i, j;
 
    fprintf(output, "        This is the Game of Life running in parallel using CUDA/MPI.\n");
    fprintf(output, "######################### FINAL WORLD FOR RANK %d ###############################\n\n", my_rank);
     
    for( i = 0; i < g_worldHeight; i++)
    {
        fprintf(output, "Row %2d: ", i);
        for( j = 0; j < g_worldWidth; j++)
        {
            fprintf(output, "%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
        }
        fprintf(output, "\n");
    }
    
    fprintf(output, "\n\n");
    fflush(output);		// Flush buffer
    fclose(output);		// close buffer
}

/* Global function to free memory allocated for g_data, g_resultData, top_ghost_row, and 
   bottom_ghost_row */
extern "C" void freeMemory()
{
    cudaFree(g_data);
    cudaFree(g_resultData);
    cudaFree(top_ghost_row);
    cudaFree(bottom_ghost_row);
}

/* Prints the ghost rows. Used for debugging.*/
/*extern "C" void printGhostRow()
{
    int i;//, k;
 
    // print rows
    printf("Top G_Row %2d: ", 0);
    for( i = 0; i < g_worldWidth; i++)
    {
        printf("%u ", (unsigned int)top_ghost_row[i]);
    }
    printf("\n");
    printf("Bot G_Row %2d: ", 0);
    for( i = 0; i < g_worldWidth; i++)
    {
        printf("%u ", (unsigned int)bottom_ghost_row[i]);
    }
    printf("\n\n");
}*/ 