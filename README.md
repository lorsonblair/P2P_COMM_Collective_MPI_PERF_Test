# P2P_COMM_Collective_MPI_PERF_Test
Point to Point Communication and Collective MPI Performance Test

# Getting started

These instructions will get you compiling and running the programs in no time. The structure of the repo is separated roughly in two partitions: Assign 3 to Final contains the source code to build the benchmarking program for CSCI6360 Parallel Programming final project, while Assign4 contains the code related to Assignment 4.

# Prerequisite

Have the following libraries installed in your environment:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Open MPI](https://www.open-mpi.org/) 


Once the environemnt is properly set up, we can move on to the programs. 

# Assignmet 3 to Final

The main program `gol-main.c` runs the MPI and CUDA version of Conway's Game of Life algorithm. The algorithm is derived from Assignment 3's deliverable and it is by the courtesy of Lorson Blair. ph.D. at Department of Computer Science at RPI. 
Since we need to evaluate the performance of four parallel computing models running the algorithm, we have adopted the logp_mpi library for the assistance. This work comes from the work:

```
Thilo Kielmann, Henri E. Bal, and Kees Verstoep. 2000. Fast Measurement of LogP
Parameters for Message Passing Platforms. In Proceedings of the 15 IPDPS 2000
Workshops on Parallel and Distributed Processing (IPDPS '00), Jos� D. P. Rolim
(Ed.). Springer-Verlag, London, UK, UK, 1176-1183.
```

and curated by willtunnels (https://github.com/willtunnels) to make the library available to the public domain. 

Now, let's go through the compilation and execution of the program.

## Compiling

First thing first, get to the directory `assignment3_to_final/` and open the Makefile. In the script, make sure that the first few parameters have the preferable settings that you need. Once done checking, you should be able to simply send the `make` command. In the terminal connected to AiMOS or any sort of parallel system, enter:

```bash
make
```

And the Makefile will handle the rest from here.

## Running the test

The gol-main.c is the main source code file that contains the test. After a successful compilation, you should get an executable called `gol-cuda-mpi-exe`. To execute it, you need to imput another 5 parameters. These parameters are:

- pattern number
- square size of the world
- number of iterations
- number of threads per block
- whether to ouput or not (0: false, 1: true)

For instance, you can issue the following in the terminal:

```bash
./gol-cuda-mpi-exe 0 32 2 256 0
```

To properly run the program.

# Assignment 4

There are two binary files after the compilation: io-main and parallelio.

## Compiling

The compilation is straighforward. Simply enter in the terminal:

```bash
make
```

And voila, the two executables appear.

## Run the tests

To run the tests, simply execute either one as:

```bash
./io-main
```

or 

```bash
./parallelio
```

# Authors

- [Lorson Blair]()
- [Yitao Shen]()
- [Jinqiang Jiang]()
- [Charly Huang](huangc11@rpi.edu)

# Acknowledgements

And also a special thanks to the authors and maintainers of logp_mpi libraries to make this benchmarking possible:

- [Thilo Kielmann](kielmann@cs.vu.nl)
- [Kees Verstoep](versto@cs.vu.nl)

One last thing. A special credit and memory is given to Dr. John H. Conway, the brilliant mathematician who invented the Game of Life algorithm and passed away on April 11, 2020 due to the outbreak of COVID-19 at the age of 82. This time of hardship has been affecting all of us but we will get through it together to endure and prosper.