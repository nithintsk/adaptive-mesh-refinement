# Parallelization of Adaptive Mesh Refinement Problem
### Description
Compare using pthreads, openMP, CUDA and MPI, the performance of different parallelization approaches to the Adaptive Mesh Refinement(AMR) problem. AMR is a modeling method to simulate the dissipation of a natural phenomenon like force or temperature across a medium. The input is a grid of arbitrary resolution divided into a number of boxes, each of which contain a certain weight(Domain specific value). Based on a dissipation model specific to the natural phenomenon, we can simulate the spread of weights between adjacent elements of the grid. In this project, the natural phenomenon is heat and we model the dissipation of heat throughout the medium represented by the grid. 

### Parameters
AFFECT_RATE - Fraction of the difference between the weighted average temperature and current temperature of grid elements by which the current temperature of the element increases or decreases
EPSILON - Specifies the convergence point. If the difference between the maximum and minimum temperature of the grid is below EPSILON % of the maximum temperature in the grid, convergence is said to be achieved.

### Serial Execution
The folder Serial contains the initial serial implementation of the dissipation model. 

**Execution instructions:**
```
$ make
$ time ./serial <AFFECT_RATE> <EPSILON>  < input_file
```

### pthreads and openMP based parallelization
These folders contain the pthreads and openMP implementation of the dissipation model respectively.
+ Persistent.c - Threads remain persistent through multiple iterations. This way, thread creation and destroying overhead is decreased.
+ Disposable.c - New threads are created for each iteration to calculate dissipation at that particular instance throughout the grid.

**Execution instructions:**
```
$ make
$ time ./persistent <AFFECT_RATE> <EPSILON>  < input_file
$ time ./disposable <AFFECT_RATE> <EPSILON>  < input_file
```

### CUDA implementation
A cuda kernel was implemented to move computation to GPU threads. Due to high number of cudaMemcpy calls between device and host, this method had high overhead compared to pthreads and openMP.

**Execution instructions:**
```
$ make
$ time ./persistent <AFFECT_RATE> <EPSILON>  < input_file
```

### MPI implementation
Message passing was used to split work between processes and subsequently perform a second level of parallelization using openMP.

**Execution instructions:**
```
$ make
$ time ./mpiprog <AFFECT_RATE> <EPSILON>  < input_file
```

### Sample output
```
Dissipation converged in 794818 iterations.
     with max DSV = 0.084637 and min DSV = 0.082944
     affect rate = 0.020000;     epsilon = 0.020000
Elapsed convergence loop time (clock): 261030000
Elapsed convergence loop time (time): 261.000000
Elapsed convergence loop time (chrono): 261046235.000000
```
