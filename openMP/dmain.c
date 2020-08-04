#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "disposable.h"

int NUM_BOXES;
int NUM_ROWS;
int NUM_COLS;
int NUM_THREADS;
int WORKLOAD;
double AFFECT_RATE;
double EPSILON;
double MIN_DSV;
double MAX_DSV;

int main (int argc, char *argv[])
{
    AFFECT_RATE = strtod(argv[1],NULL);
    EPSILON = strtod(argv[2],NULL);
    NUM_THREADS = (int)strtod(argv[3],NULL);
    
    // Get the parameters describing the grid --> First line in data file
    readgridparam();
    
    // Create a Box array of size NUM_BOXES
    struct Box boxes[NUM_BOXES];
    
	// Populate the boxes datastructure
	populate(boxes);
    
    // Now we need to compute the overlaps between each node.
    calcoverlap(boxes);
	
	// Setting up time calculation reporting
	time_t start_time,end_time;
	clock_t clock_time;
	struct timespec t_start, t_end;
	double diff;

    clock_gettime(CLOCK_REALTIME,&t_start);
    time(&start_time);
    clock_time = clock();
	
    // This should occur till convergence
    int count=1;
    do {
		MAX_DSV = INT_MIN;
		MIN_DSV = INT_MAX;
		compute_commit_dsv(boxes);
        count++;
        if (MAX_DSV == 0) break;
    } while (((MAX_DSV-MIN_DSV) / MAX_DSV) > EPSILON);
    count--;
	
	time(&end_time);
    clock_time = clock() - clock_time;
    clock_gettime(CLOCK_REALTIME,&t_end);
    diff = (double)(((t_end.tv_sec - t_start.tv_sec) * CLOCKS_PER_SEC) + ((t_end.tv_nsec - t_start.tv_nsec) / 1000));
	
	printf("*************************************************************************\n");
    printf("Dissipation converged in %d iterations.\n", count);
    printf("     with max DSV = %lf and min DSV = %lf\n", MAX_DSV, MIN_DSV);
    printf("     affect rate = %lf;     epsilon = %lf\n", AFFECT_RATE, EPSILON);
    printf("Elapsed convergence loop time (clock): %lu\n",clock_time);
    printf("Elapsed convergence loop time (time): %lf\n",difftime(end_time, start_time));
    printf("Elapsed convergence loop time (chrono): %lf\n",diff);
}

