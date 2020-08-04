#include "persistent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "cuda.h"
#include <cuda_runtime.h>

__global__ void compute_commit_dsv(Box *box_arr, int *count, int *count_iter, int* DNUM_BOXES, double* DAFFECT_RATE, double* DEPSILON, double* CMIN_DSV, double* CMAX_DSV) {

	int index = threadIdx.x; //Thread index within block
	int stride = blockDim.x; // Num of threads in the block
    int workload = (*DNUM_BOXES + (stride-1)) / stride;  //1fp
	if(threadIdx.x==0) {
 	    *count=1;
		*count_iter=1;	
    }
	
    __shared__ double KMIN_DSV;
	__shared__ double KMAX_DSV;
	__shared__ double LMIN_DSV[NUM_THREADS];
	__shared__ double LMAX_DSV[NUM_THREADS];

    __syncthreads();

    do {
        for (int i=index*workload; (i<(index+1)*workload) && (i<*DNUM_BOXES);i++) { 
			box_arr[i].waat = 0;

			// Get weighted average of top neighbours 
			if (box_arr[i].num_top != 0) {
				for (int j = 0; j < box_arr[i].num_top; j++) {
					int cur_topid = box_arr[i].top_ids[j];
					int overlap = box_arr[i].top_ov[j];
					box_arr[i].waat = box_arr[i].waat + box_arr[cur_topid].dsv * overlap; 
					//atomicAdd(count,2);
				}
			}
			else {
				box_arr[i].waat = box_arr[i].waat + box_arr[i].width * box_arr[i].dsv;
				//atomicAdd(count,2);
			}

			// Get weighted average of bottom neighbours
			if (box_arr[i].num_bottom != 0) {
				int j;
				for (j = 0; j < box_arr[i].num_bottom; j++) {
					int cur_bottomid = box_arr[i].bottom_ids[j];
					int overlap = box_arr[i].bottom_ov[j];
					box_arr[i].waat = box_arr[i].waat + box_arr[cur_bottomid].dsv * overlap;
					//atomicAdd(count,2);
				}
			}
			else {
				box_arr[i].waat = box_arr[i].waat + box_arr[i].width * box_arr[i].dsv;
				//atomicAdd(count,2);
			}

			// Get weighted average of left neighbours
			if (box_arr[i].num_left != 0) {
				int j;
				for (j = 0; j < box_arr[i].num_left; j++) {
					int cur_leftid = box_arr[i].left_ids[j];
					int overlap = box_arr[i].left_ov[j];
					box_arr[i].waat = box_arr[i].waat + box_arr[cur_leftid].dsv * overlap;
					//atomicAdd(count,2);
				}
			}
			else {
				box_arr[i].waat = box_arr[i].waat + box_arr[i].height * box_arr[i].dsv;
				//atomicAdd(count,2);
			}

			// Get weighted average of right neighbours
			if (box_arr[i].num_right != 0) {
				int j;
				for (j = 0; j < box_arr[i].num_right; j++) {
					int cur_rightid = box_arr[i].right_ids[j];
					int overlap = box_arr[i].right_ov[j];
					box_arr[i].waat = box_arr[i].waat + box_arr[cur_rightid].dsv * overlap;
					//atomicAdd(count,2);
				}
			}
			else {
				box_arr[i].waat = box_arr[i].waat + box_arr[i].height * box_arr[i].dsv;
				//atomicAdd(count,2);
			}

			// Find the weighted average by dividing with the perimeter
			box_arr[i].waat = box_arr[i].waat / box_arr[i].perimeter;
			//atomicAdd(count,1);

		}

		LMIN_DSV[index] = INT_MAX;
		LMAX_DSV[index] = INT_MIN;

		if(threadIdx.x == 0)
		{
			KMIN_DSV=INT_MAX;
			KMAX_DSV=INT_MIN;			
			//atomicAdd(count,1);
		}

		__syncthreads();

        for (int i=index*workload; (i<(index+1)*workload) && (i<*DNUM_BOXES);i++) {
			if (box_arr[i].waat > box_arr[i].dsv) {
				box_arr[i].dsv = box_arr[i].dsv + *DAFFECT_RATE * (box_arr[i].waat - box_arr[i].dsv);
				//atomicAdd(count,3);
			}
			else {
				box_arr[i].dsv = box_arr[i].dsv - *DAFFECT_RATE * (box_arr[i].dsv - box_arr[i].waat);
				//atomicAdd(count,3);
			}
			if (box_arr[i].dsv < LMIN_DSV[threadIdx.x]) {
				LMIN_DSV[threadIdx.x] = box_arr[i].dsv;
				//atomicAdd(count,1);
			}
			if (box_arr[i].dsv > LMAX_DSV[threadIdx.x]) {
				LMAX_DSV[threadIdx.x] = box_arr[i].dsv;
				//atomicAdd(count,1);
			}
		}
		
		__syncthreads();

		if(threadIdx.x == 0)
		{
			atomicAdd(count_iter,1);
			for(int j=0;j<NUM_THREADS;j++) {
				if(LMIN_DSV[j] < KMIN_DSV) KMIN_DSV=LMIN_DSV[j];				
				if(LMAX_DSV[j] > KMAX_DSV) KMAX_DSV=LMAX_DSV[j];
				//atomicAdd(count,2);
			}
		}

        __syncthreads();

    } while (((KMAX_DSV - KMIN_DSV) / KMAX_DSV) > *DEPSILON);

	if(threadIdx.x == 0)
	{
		atomicAdd(count_iter,-1);
		*CMIN_DSV=KMIN_DSV;
		*CMAX_DSV=KMAX_DSV;
	}
}

int cuda_compute_commit_dsv(Box* box_arr, int &numboxes, double &affectrate, double &epsilon, double &hmin, double &hmax) {
	
	// Retrieving count value of number of iterations
	int *count = (int*)malloc(sizeof(int));
	*count=0;
	int *dcount;
	cudaMalloc(&dcount,sizeof(int));

	int *count_flop = (int*)malloc(sizeof(int));
	*count_flop=0;
	int *dcount_flop;
	cudaMalloc(&dcount_flop,sizeof(int));

    cudaMemcpy(dcount,count,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dcount_flop,count,sizeof(int),cudaMemcpyHostToDevice);

	int *DNUM_BOXES;
	double *DAFFECT_RATE;
	double *DEPSILON;
	double *CMIN_DSV;
	double *CMAX_DSV;

	cudaMalloc(&DAFFECT_RATE,sizeof(double));  
	cudaMemcpy(DAFFECT_RATE,&affectrate,sizeof(double),cudaMemcpyHostToDevice); 
    cudaMalloc(&DEPSILON,sizeof(double));  
	cudaMemcpy(DEPSILON,&epsilon,sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc(&CMIN_DSV,sizeof(double));
	cudaMemcpy(CMIN_DSV,&hmin,sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc(&CMAX_DSV,sizeof(double));
	cudaMemcpy(CMAX_DSV,&hmax,sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc(&DNUM_BOXES,sizeof(int));  
	cudaMemcpy(DNUM_BOXES,&numboxes,sizeof(int),cudaMemcpyHostToDevice); 
	
	struct Box *h1boxes, *dboxes;
	// Copy the array over to h1boxes
	h1boxes = (Box*)malloc(NUM_BOXES * (sizeof(Box)));
	memcpy(h1boxes,box_arr,NUM_BOXES * (sizeof(Box)));
	//printboxes(box_arr);
	for(int i=0;i<numboxes;i++)
	{
		cudaMalloc(&(h1boxes[i].top_ids),box_arr[i].num_top * sizeof(int));
		cudaMalloc(&(h1boxes[i].top_ov),box_arr[i].num_top * sizeof(int));
		cudaMalloc(&(h1boxes[i].bottom_ids),box_arr[i].num_bottom * sizeof(int));
		cudaMalloc(&(h1boxes[i].bottom_ov),box_arr[i].num_bottom * sizeof(int));
		cudaMalloc(&(h1boxes[i].left_ids),box_arr[i].num_left * sizeof(int));
		cudaMalloc(&(h1boxes[i].left_ov),box_arr[i].num_left * sizeof(int));
		cudaMalloc(&(h1boxes[i].right_ids),box_arr[i].num_right * sizeof(int));
		cudaMalloc(&(h1boxes[i].right_ov),box_arr[i].num_right * sizeof(int));
		
		cudaMemcpy(h1boxes[i].top_ids,box_arr[i].top_ids,box_arr[i].num_top * sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(h1boxes[i].top_ov,box_arr[i].top_ov,box_arr[i].num_top * sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(h1boxes[i].bottom_ids,box_arr[i].bottom_ids,box_arr[i].num_bottom * sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(h1boxes[i].bottom_ov,box_arr[i].bottom_ov,box_arr[i].num_bottom * sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(h1boxes[i].left_ids,box_arr[i].left_ids,box_arr[i].num_left * sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(h1boxes[i].left_ov,box_arr[i].left_ov,box_arr[i].num_left * sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(h1boxes[i].right_ids,box_arr[i].right_ids,box_arr[i].num_right * sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(h1boxes[i].right_ov,box_arr[i].right_ov,box_arr[i].num_right * sizeof(int),cudaMemcpyHostToDevice);

	}

	cudaMalloc(&dboxes, NUM_BOXES * sizeof(Box)); 
	cudaMemcpy(dboxes,h1boxes,NUM_BOXES * (sizeof(Box)),cudaMemcpyHostToDevice);

	printf("Placing call to kernel\n");
    compute_commit_dsv<<<1, NUM_THREADS>>>(dboxes,dcount_flop , dcount, DNUM_BOXES, DAFFECT_RATE, DEPSILON, CMIN_DSV, CMAX_DSV);
	cudaDeviceSynchronize();
	printf("Returned from call to kernel\n");

	cudaMemcpy(count,dcount,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(count_flop,dcount_flop,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(&hmin,CMIN_DSV,sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(&hmax,CMAX_DSV,sizeof(double),cudaMemcpyDeviceToHost);

	printf("Flops per iteration are: %d\n",count_flop);

	return *count;
}

void readgridparam() {

	// Assuming each line in the datafile is Max 500 characters
	char line[MAXLEN] = "";

	fflush(stdin);
	if (fgets(line, sizeof(line), stdin)) {
        // If the first line of the file contains -1, exit
		if (line[0] == '-') {
			fprintf(stderr, "First line of the file contains -1. Exiting....");
			exit(EXIT_FAILURE);
		}
		else {
			// We only expect 3 numbers in the first line
			// <number of grid boxes> <num_grid_rows> <num_grid_cols>
			int arr[3];
			parseline(arr, line, 0);
			NUM_BOXES = arr[0];
			NUM_ROWS = arr[1];
			NUM_COLS = arr[2];
		}
	}
	else {
		fprintf(stderr, "File may not exist or is empty. Exiting....");
		exit(EXIT_FAILURE);
	}

}

void populate(Box* box_arr) {

	char line1[MAXLEN] = "";
	int box_count = 0;
	// Read rest of file and populate the data structure
	fflush(stdin);
	while (fgets(line1, sizeof(line1), stdin)) {
		if (line1[0] == '-') {
			break;
		}
		else if (!strcmp(line1, "")) continue;
		else if (!(line1[0] >= '0' && line1[0] <= '9')) continue;
		else {
			// Create new Box element

			// Get Box id;
			int id[1];
			parseline(id, line1, 0);
			box_arr[box_count].id = id[0];

			// Get location, height and width
			fflush(stdin);
			fgets(line1, sizeof(line1), stdin);
			int box_loc[4];
			parseline(box_loc, line1, 0);
			box_arr[box_count].up_left_y = box_loc[0];
			box_arr[box_count].up_left_x = box_loc[1];
			box_arr[box_count].height = box_loc[2];
			box_arr[box_count].width = box_loc[3];
			box_arr[box_count].perimeter = 2 * (box_arr[box_count].height + box_arr[box_count].width);

			// Get top neighbours
			fflush(stdin);
			fgets(line1, sizeof(line1), stdin);
			int top_num;
			top_num = parsefirst(line1);
			box_arr[box_count].num_top = top_num;
			int* toparr = (int*)malloc(top_num * sizeof(int));
			int* toparrov = (int*)malloc(top_num * sizeof(int));
			parseline(toparr, line1, 1);
			box_arr[box_count].top_ids = toparr;
			box_arr[box_count].top_ov = toparrov;
			if (top_num == 0) {
				box_arr[box_count].top_ids = NULL;
			}

			// Get bottom neighbours
			fflush(stdin);
			fgets(line1, sizeof(line1), stdin);
			int bottom_num;
			bottom_num = parsefirst(line1);
			box_arr[box_count].num_bottom = bottom_num;
			int* bottomarr = (int*)malloc(bottom_num * sizeof(int));
			int* bottomarrov = (int*)malloc(bottom_num * sizeof(int));
			parseline(bottomarr, line1, 1);
			box_arr[box_count].bottom_ids = bottomarr;
			box_arr[box_count].bottom_ov = bottomarrov;
			if (bottom_num == 0) {
				box_arr[box_count].bottom_ids = NULL;
			}

			// Get left neighbours
			fflush(stdin);
			fgets(line1, sizeof(line1), stdin);
			int left_num;
			left_num = parsefirst(line1);
			box_arr[box_count].num_left = left_num;
			int* leftarr = (int*)malloc(left_num * sizeof(int));
			int* leftarrov = (int*)malloc(left_num * sizeof(int));
			parseline(leftarr, line1, 1);
			box_arr[box_count].left_ids = leftarr;
			box_arr[box_count].left_ov = leftarrov;
			if (left_num == 0) {
				box_arr[box_count].left_ids = NULL;
			}

			// Get right neighbours
			fflush(stdin);
			fgets(line1, sizeof(line1), stdin);
			int right_num;
			right_num = parsefirst(line1);
			box_arr[box_count].num_right = right_num;
			int* rightarr = (int*)malloc(right_num * sizeof(int));
			int* rightarrov = (int*)malloc(right_num * sizeof(int));
			parseline(rightarr, line1, 1);
			box_arr[box_count].right_ids = rightarr;
			box_arr[box_count].right_ov = rightarrov;
			if (right_num == 0) {
				box_arr[box_count].right_ids = NULL;
			}

			// Get dsv value
			fflush(stdin);
			fgets(line1, sizeof(line1), stdin);
			double dsv_val;
			dsv_val = parsedsv(line1);
			box_arr[box_count].dsv = dsv_val;

			// Move to next box
			box_count++;
			fflush(stdin);
		}
	}
}

void parseline(int* num, char* path, int func) {
	//char c;
	int i = 0, digit, number = 0;
	int num_count = 0;
	if (func == 0) i = 0;
	if (func == 1) {
		while (i < strlen(path) && path[i] >= '0' && path[i] <= '9') {
			i++;
		}
	}
	for (; i < strlen(path); i++)
	{
		if (path[i] >= '0' && path[i] <= '9') //to confirm it's a digit
		{
			number = 0;
			do {
				digit = path[i] - '0';
				number = number * 10 + digit;
				i++;
			} while (i < strlen(path) && path[i] >= '0' && path[i] <= '9');
			num[num_count] = number;
			num_count++;
		}
	}
}

int parsefirst(char* path) {
	int i = 0, digit, number = 0;
	do {
		digit = path[i] - '0';
		number = number * 10 + digit;
		i++;
	} while (i < strlen(path) && path[i] >= '0' && path[i] <= '9');
	return number;
}

double parsedsv(char* path) {
	double number = 0;
	number = strtod(path, NULL);
	return number;
}

void calcoverlap(struct Box* box_arr) {
	int i;
	for (i = 0; i < NUM_BOXES; i++) {
		// Calculate TOP overlap for each node.
		// If 0, skip.
		if (box_arr[i].num_top != 0) {
			int j;
			for (j = 0; j < box_arr[i].num_top; j++) {
				// find right most of x_left and xtop_left
				int cur_topid = box_arr[i].top_ids[j];
				int len2, len1;
				if (box_arr[i].up_left_x >= box_arr[cur_topid].up_left_x) len1 = box_arr[i].up_left_x;
				else len1 = box_arr[cur_topid].up_left_x;
				//printf("Box id %d. Len1 value %d.\n",i,len1);

				if ((box_arr[i].up_left_x + box_arr[i].width) <= (box_arr[cur_topid].up_left_x + box_arr[cur_topid].width)) len2 = (box_arr[i].up_left_x + box_arr[i].width);
				else len2 = (box_arr[cur_topid].up_left_x + box_arr[cur_topid].width);
				//printf("Box id %d. Len2 value %d.\n",i,len2);

				box_arr[i].top_ov[j] = abs(len2 - len1);
				//printf("Box id %d. Neighbor id %d. Value %d.\n",i,cur_topid,box_arr[i].top_ov[j]);
			}
		}

		// Calculate BOTTOM overlap for each node.
		// If 0, skip.
		if (box_arr[i].num_bottom != 0) {
			int j;
			for (j = 0; j < box_arr[i].num_bottom; j++) {
				// find right most of x_left and xbottom_left
				int cur_bottomid = box_arr[i].bottom_ids[j];
				int len2, len1;
				if (box_arr[i].up_left_x >= box_arr[cur_bottomid].up_left_x) len1 = box_arr[i].up_left_x;
				else len1 = box_arr[cur_bottomid].up_left_x;
				//printf("Box id %d. Len1 value %d.\n",i,len1);

				// find left most of x_left + width and xbottom_left + its width
				if ((box_arr[i].up_left_x + box_arr[i].width) <= (box_arr[cur_bottomid].up_left_x + box_arr[cur_bottomid].width)) len2 = (box_arr[i].up_left_x + box_arr[i].width);
				else len2 = (box_arr[cur_bottomid].up_left_x + box_arr[cur_bottomid].width);
				//printf("Box id %d. Len2 value %d.\n",i,len2);

				box_arr[i].bottom_ov[j] = abs(len2 - len1);
				//printf("Box id %d. Neighbor id %d. Value %d.\n",i,cur_bottomid,box_arr[i].bottom_ov[j]);
			}
		}

		// Calculate left overlap for each node.
		// If 0, skip.
		if (box_arr[i].num_left != 0) {
			int j;
			for (j = 0; j < box_arr[i].num_left; j++) {
				// find bottom most of y_left and yleft_left
				int cur_leftid = box_arr[i].left_ids[j];
				int len2, len1;
				if (box_arr[i].up_left_y >= box_arr[cur_leftid].up_left_y) len1 = box_arr[i].up_left_y;
				else len1 = box_arr[cur_leftid].up_left_y;
				//printf("Box id %d. Len1 value %d.\n",i,len1);

				// find top most of y_left + height and yleft_left + its height
				if ((box_arr[i].up_left_y + box_arr[i].height) <= (box_arr[cur_leftid].up_left_y + box_arr[cur_leftid].height)) len2 = (box_arr[i].up_left_y + box_arr[i].height);
				else len2 = (box_arr[cur_leftid].up_left_y + box_arr[cur_leftid].height);
				//printf("Box id %d. Len2 value %d.\n",i,len2);

				box_arr[i].left_ov[j] = abs(len2 - len1);
				//printf("Box id %d. Neighbor id %d. Value %d.\n",i,cur_leftid,box_arr[i].left_ov[j]);
			}
		}

		// Calculate right overlap for each node.
		// If 0, skip.
		if (box_arr[i].num_right != 0) {
			int j;
			for (j = 0; j < box_arr[i].num_right; j++) {
				// find bottom most of y_left and yright_left
				int cur_rightid = box_arr[i].right_ids[j];
				int len2, len1;
				if (box_arr[i].up_left_y >= box_arr[cur_rightid].up_left_y) len1 = box_arr[i].up_left_y;
				else len1 = box_arr[cur_rightid].up_left_y;
				//printf("Box id %d. Len1 value %d.\n",i,len1);

				// find top most of y_left + height and yright_left + its height
				if ((box_arr[i].up_left_y + box_arr[i].height) <= (box_arr[cur_rightid].up_left_y + box_arr[cur_rightid].height)) len2 = (box_arr[i].up_left_y + box_arr[i].height);
				else len2 = (box_arr[cur_rightid].up_left_y + box_arr[cur_rightid].height);
				//printf("Box id %d. Len2 value %d.\n",i,len2);

				box_arr[i].right_ov[j] = abs(len2 - len1);
				//printf("Box id %d. Neighbor id %d. Value %d.\n",i,cur_rightid,box_arr[i].right_ov[j]);
			}
		}
	}
}

void printboxes(struct Box* box_arr) {

	int i;
	for (i = 0; i < NUM_BOXES; i++) {
		printf("================================");
		printf("\n\nBox id: %d\n", box_arr[i].id);
		printf("Box left_X, left_y, height, width, perimiter: %d, %d, %d, %d, %d\n", box_arr[i].up_left_x, box_arr[i].up_left_y, box_arr[i].height, box_arr[i].width, box_arr[i].perimeter);

		printf("Box top neighbours and overlap: ");
		int j;
		for (j = 0; j < box_arr[i].num_top; j++) {
			printf("%d:%d, ", box_arr[i].top_ids[j], box_arr[i].top_ov[j]);
		}
		printf("\n");

		printf("Box bottom neighbours and overlap: ");
		for (j = 0; j < box_arr[i].num_bottom; j++) {
			printf("%d:%d, ", box_arr[i].bottom_ids[j], box_arr[i].bottom_ov[j]);
		}
		printf("\n");

		printf("Box left neighbours: ");
		for (j = 0; j < box_arr[i].num_left; j++) {
			printf("%d:%d, ", box_arr[i].left_ids[j], box_arr[i].left_ov[j]);
		}
		printf("\n");

		printf("Box right neighbours: ");
		for (j = 0; j < box_arr[i].num_right; j++) {
			printf("%d:%d, ", box_arr[i].right_ids[j], box_arr[i].right_ov[j]);
		}
		printf("\n");

		printf("Box dsv value: %lf", box_arr[i].dsv);
		printf("\n");

	}
}
