#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "disposable.h"

int NUM_BOXES;
int NUM_ROWS;
int NUM_COLS;
int WORKLOAD;
//int NUM_THREADS;
double AFFECT_RATE;
double EPSILON;
double MIN_DSV;
double MAX_DSV;

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	AFFECT_RATE = 0.02;
	EPSILON = 0.02;
    int flag = 0;

	// Declare a variable to store rank
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int *num_nbor, **nbor_ids, **overlap, *perimeter;
	double* dsv;
	int nbor_dim = 0;
	const int root = 0;
	MPI_Status status;
	int count = 0, count1 = 0;

	// Setting up time calculation reporting
	time_t start_time, end_time;
	clock_t clock_time;
	struct timespec t_start, t_end;
	double diff;
    FILE *fp;

	if (rank == 0) {
		// Get the parameters describing the grid --> First line in data file
        fp = fopen(TESTFILE,"r");
		readgridparam(fp);
		WORKLOAD = (int)ceil((float)NUM_BOXES / (float)(size - 1));
	}

	// Create a Box array of size NUM_BOXES
	struct Box* boxes;

	if (rank == 0) {
		// Allocating memory for boxes only within process 0
		boxes = (struct Box*)malloc(NUM_BOXES * (sizeof(struct Box)));

		// Populate the boxes datastructure
		populate(boxes, fp);

		// Now we need to compute the overlaps between each node.
		calcoverlap(boxes);

		// Find the max dimension of overlap
		for (int i = 0; i < NUM_BOXES; i++) {
			int temp;
			temp = boxes[i].num_top + boxes[i].num_bottom + boxes[i].num_left + boxes[i].num_right;
			if (boxes[i].num_top == 0) temp++;
			if (boxes[i].num_bottom == 0) temp++;
			if (boxes[i].num_left == 0) temp++;
			if (boxes[i].num_right == 0) temp++;
			if (temp > nbor_dim) nbor_dim = temp;
		}
	}

	// Broadcast the neighbor dimension
	MPI_Bcast(&nbor_dim, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&NUM_BOXES, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&WORKLOAD, 1, MPI_INT, root, MPI_COMM_WORLD);
	//printf("Rank: %d. NUM_BOXES: %d. WORKLOAD: %d. AFFECT_RATE: %lf. EPSILON: %lf.\n", rank,NUM_BOXES,WORKLOAD,AFFECT_RATE,EPSILON);

	// Now every process has the dimensions required to allocate memory
	num_nbor = (int*)malloc(NUM_BOXES * sizeof(int));
	dsv = (double*)malloc(NUM_BOXES * sizeof(double));
	perimeter = (int*)malloc(NUM_BOXES * sizeof(int));
	nbor_ids = alloc_2d_init(NUM_BOXES, nbor_dim);
	overlap = alloc_2d_init(NUM_BOXES, nbor_dim);

	//Serializing the array of structures
	if (rank == 0) {
		reformat_overlap(boxes, num_nbor, dsv, nbor_ids, overlap, perimeter);
		//printboxes(boxes);
		//printarrays(num_nbor,dsv,nbor_ids,overlap,perimeter);
	}

	// Now broadcast the arrays to all processes.
	MPI_Bcast(num_nbor, NUM_BOXES, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(perimeter, NUM_BOXES, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&(nbor_ids[0][0]), NUM_BOXES * nbor_dim, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&(overlap[0][0]), NUM_BOXES * nbor_dim, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(dsv, NUM_BOXES, MPI_DOUBLE, root, MPI_COMM_WORLD);
	
	double* temp_dsv = (double*)malloc(NUM_BOXES * sizeof(double));

	if (rank == 0) {

		clock_gettime(CLOCK_REALTIME, &t_start);
		time(&start_time);
		clock_time = clock();
        printf("Entering do while\n");
	}

	do {
        if (rank==0) printf("\n\nBack to top of do while\n");

		MIN_DSV = INT_MAX;
		MAX_DSV = INT_MIN;
		if (rank != 0){
			compute_commit_dsv(num_nbor, dsv, nbor_ids, overlap, perimeter);
        }

		if (rank == 0)
		{
			MAX_DSV = INT_MIN;
			MIN_DSV = INT_MAX;
            // Substitute with gather
			double* dsv1 = (double*)malloc(NUM_BOXES * sizeof(double)),
				* dsv2 = (double*)malloc(NUM_BOXES * sizeof(double)),
				* dsv3 = (double*)malloc(NUM_BOXES * sizeof(double)),
				* dsv4 = (double*)malloc(NUM_BOXES * sizeof(double)),
				* dsv5 = (double*)malloc(NUM_BOXES * sizeof(double)),
				* dsv6 = (double*)malloc(NUM_BOXES * sizeof(double)),
				* dsv7 = (double*)malloc(NUM_BOXES * sizeof(double)),
				* dsv8 = (double*)malloc(NUM_BOXES * sizeof(double));
			MPI_Recv(dsv1, NUM_BOXES, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
            printf("Received from 1\n");
			MPI_Recv(dsv2, NUM_BOXES, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &status);
            printf("Received from 2\n");
			MPI_Recv(dsv3, NUM_BOXES, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, &status);
            printf("Received from 3\n");
			MPI_Recv(dsv4, NUM_BOXES, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD, &status);
            printf("Received from 4\n");
			MPI_Recv(dsv5, NUM_BOXES, MPI_DOUBLE, 5, 0, MPI_COMM_WORLD, &status);
			printf("Received from 5\n");
			MPI_Recv(dsv6, NUM_BOXES, MPI_DOUBLE, 6, 0, MPI_COMM_WORLD, &status);
            printf("Received from 6\n");
			MPI_Recv(dsv7, NUM_BOXES, MPI_DOUBLE, 7, 0, MPI_COMM_WORLD, &status);
            printf("Received from 7\n");
			MPI_Recv(dsv8, NUM_BOXES, MPI_DOUBLE, 8, 0, MPI_COMM_WORLD, &status);
            printf("Received from 8\n");
            
			int m;
			for (m = 0; m < NUM_BOXES && m < WORKLOAD; m++)
				dsv[m] = dsv1[m];
			for (m = WORKLOAD; m < NUM_BOXES && m < 2 * WORKLOAD; m++)
				dsv[m] = dsv2[m];
			for (m = 2 * WORKLOAD; m < NUM_BOXES && m < 3 * WORKLOAD; m++)
				dsv[m] = dsv3[m];
			for (m = 3 * WORKLOAD; m < NUM_BOXES && m < 4 * WORKLOAD; m++)
				dsv[m] = dsv4[m];
			for (m = 4 * WORKLOAD; m < NUM_BOXES && m < 5 * WORKLOAD; m++)
				dsv[m] = dsv5[m];
			for (m = 5 * WORKLOAD; m < NUM_BOXES && m < 6 * WORKLOAD; m++)
				dsv[m] = dsv6[m];
			for (m = 6 * WORKLOAD; m < NUM_BOXES && m < 7 * WORKLOAD; m++)
				dsv[m] = dsv7[m];
			for (m = 7 * WORKLOAD; m < NUM_BOXES && m < 8 * WORKLOAD; m++)
				dsv[m] = dsv8[m];

			count++;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(dsv, NUM_BOXES, MPI_DOUBLE, root, MPI_COMM_WORLD);
		
		double MIN_RED, MAX_RED;
		MPI_Reduce(&MIN_DSV, &MIN_RED, 1, MPI_DOUBLE, MPI_MIN, root, MPI_COMM_WORLD);
		MPI_Reduce(&MAX_DSV, &MAX_RED, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&MIN_DSV, &MIN_RED, 1, MPI_DOUBLE, MPI_MIN, root, MPI_COMM_WORLD);
        if (rank == 0) {
			MIN_DSV = MIN_RED;
			MAX_DSV = MAX_RED;
		    printf("\nCount: %d. ", count);
		    printf("\nMAX_DSV: %lf. MIN_DSV: %lf\n",MAX_DSV,MIN_DSV);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&MIN_DSV, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
		MPI_Bcast(&MAX_DSV, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
	    if(rank==0) printf("End of do while\n");	

	} while (((MAX_DSV - MIN_DSV) / MAX_DSV) > EPSILON);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	if (rank == 0) {
		time(&end_time);
		clock_time = clock() - clock_time;
		clock_gettime(CLOCK_REALTIME, &t_end);
		diff = (double)(((t_end.tv_sec - t_start.tv_sec) * CLOCKS_PER_SEC) + ((t_end.tv_nsec - t_start.tv_nsec) / 1000));

		printf("*************************************************************************\n");
		printf("Dissipation converged in %d iterations.\n", count);
		printf("     with max DSV = %lf and min DSV = %lf\n", MAX_DSV, MIN_DSV);
		printf("     affect rate = %lf;     epsilon = %lf\n", AFFECT_RATE, EPSILON);
		printf("Elapsed convergence loop time (clock): %lu\n", clock_time);
		printf("Elapsed convergence loop time (time): %lf\n", difftime(end_time, start_time));
		printf("Elapsed convergence loop time (chrono): %lf\n", diff);
	}

}

