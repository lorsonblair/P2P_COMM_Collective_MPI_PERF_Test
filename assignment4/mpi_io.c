#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define FILESIZE 1000

int main(int argc, char **argv){

  int rank, nprocs;
  MPI_File fh;
  MPI_Status status;
  int bufsize, nints;
  int buf[FILESIZE];
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
  bufsize = FILESIZE/nprocs;
  nints = bufsize/sizeof(int);
  
  MPI_File_open(MPI_COMM_WORLD, "datafile", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  MPI_File_read_at(fh, rank*bufsize, buf, nints, MPI_INT, &status);
  MPI_File_close(&fh);
}
