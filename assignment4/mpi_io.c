#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define FILESIZE 1000

int main(int argc, char **argv){

    int rank, nprocs; // rank as MPI rank amount, nprocs as processor blocksize
    MPI_File file;
    MPI_Status status;
    MPI_offset offset;
    int bufsize, n_ints;
    int buf[FILESIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    bufsize = FILESIZE/nprocs;
    n_ints = bufsize/sizeof(int);
    offset = bufsize * rank;

    if (rank) strcpy (buf, 1);

    MPI_File_open(MPI_COMM_WORLD, "datafile", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_read_at(file, offset, buf, n_ints, MPI_INT, &status);
    MPI_File_close(&file);

    MPI_FInalize();
    return 0
}
