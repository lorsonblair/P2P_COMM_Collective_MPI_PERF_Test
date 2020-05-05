# (c) copyright 1999-2003 by Vrije Universiteit, Amsterdam, The Netherlands.
# For full copyright and restrictions on use see the file COPYRIGHT.

#
# Makefile for LogP/MPI, Version 1.3
#

# Change this to your C compiler
MPICC		= mpicc

# Make this the flags you normally pass to your compilers
CFLAGS		= -O3 -Wall

# Make this the flags you normally pass to your MPI link command (maybe empty)
LDOPTS		=

# Linker flags for the logp library
LDFLAGS		= -L. -llogp

# Define RANLIB as 'echo' if you do not need ranlib
RANLIB		= echo
# RANLIB		= ranlib

############################################################################
# You should not need to change anything below this line
############################################################################
LOGP_SRC	= logp_mpi.c logp_stats.c
LOGP_INC	= logp_mpi.h logp_stats.h
LOGP_OBJ	= $(LOGP_SRC:%.c=%.o)

LOGP_LIB	= liblogp.a




all: $(LOGP_LIB) gol

$(LOGP_LIB): $(LOGP_OBJ)
	ar -cr $(LOGP_LIB) $(LOGP_OBJ)
	$(RANLIB) $(LOGP_LIB)

.c.o: $(LOGP_INC)
	$(MPICC) $(CFLAGS) -c $*.c

gol: gol-main.c gol-with-cuda.cu $(LOGP_LIB) $(LOGP_INC)
		$(MPICC) -g $(LOGP_LIB) $(LOGP_SRC)  gol-main.c -c \
			-o gol-main.o
		nvcc gol-with-cuda.cu -c \
			-o gol-cuda.o
		$(MPICC) -g gol-main.o gol-cuda.o $(LOGP_LIB) \
			-o gol-cuda-mpi-exe \
			-L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++

clean:
	rm -rf *-exe *.o *.a