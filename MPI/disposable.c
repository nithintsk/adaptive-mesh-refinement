#include "disposable.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

void compute_commit_dsv(int* num_nbor, double* dsv, int** nbor_ids, int** overlap, int* perimeter) {

	// Get rank and use it to compute workload distribution
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int start_index = (rank - 1) * WORKLOAD;
	int end_index = rank * WORKLOAD;
	if (NUM_BOXES < end_index) end_index = NUM_BOXES;

	omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);
	//To be removed after debugging
    //omp_set_num_threads(1);
	
    double* waat = (double*)malloc(NUM_BOXES * sizeof(double));

#pragma omp parallel
	{
#pragma omp for
		for (int i = start_index; i < end_index; i++) {
			waat[i] = 0;
			for (int j = 0; j < num_nbor[i]; j++) {
				int cur_id = nbor_ids[i][j];
				int ov = overlap[i][j];
				waat[i] = waat[i] + dsv[cur_id] * ov;
				//printf("Neighbor id: %d. Neighbor overlap: %d.\n", nbor_ids[i][j],overlap[i][j]);
			}
			// Find the weighted average by dividing with the perimeter
			waat[i] = waat[i] / perimeter[i];
		}

#pragma omp barrier

		double LOCAL_MIN_DSV = INT_MAX;
		double LOCAL_MAX_DSV = INT_MIN;

#pragma omp for
		for (int i = start_index; i < end_index; i++) {
			if (waat[i] > dsv[i]) {
				dsv[i] = dsv[i] + AFFECT_RATE * (waat[i] - dsv[i]);
			}
			else {
				dsv[i] = dsv[i] - AFFECT_RATE * (dsv[i] - waat[i]);
			}
			if (dsv[i] < LOCAL_MIN_DSV) LOCAL_MIN_DSV = dsv[i];
			if (dsv[i] > LOCAL_MAX_DSV) LOCAL_MAX_DSV = dsv[i];

		}

#pragma omp critical
		{
			if (LOCAL_MIN_DSV < MIN_DSV)
				MIN_DSV = LOCAL_MIN_DSV;
			if (LOCAL_MAX_DSV > MAX_DSV)
				MAX_DSV = LOCAL_MAX_DSV;
		}
	}

	MPI_Send(dsv, NUM_BOXES, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

}

void readgridparam(FILE *fd) {

	// Assuming each line in the datafile is Max 500 characters
	char line[MAXLEN] = "";

	fflush(stdin);
	if (fgets(line, sizeof(line), fd)) {
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

void populate(Box* box_arr, FILE *fd) {

	char line1[MAXLEN] = "";
	int box_count = 0;
	fflush(stdin);
	char *fileName = "testgrid_1";
	while (fgets(line1, sizeof(line1), fd)) {
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

			//printf("\n\nBox id: %d. Box count: %d\n", box_arr[box_count].id, box_count);
			// Get location, height and width

			fflush(stdin);

			fgets(line1, sizeof(line1), fd);

			int box_loc[4];

			parseline(box_loc, line1, 0);

			box_arr[box_count].up_left_y = box_loc[0];

			box_arr[box_count].up_left_x = box_loc[1];

			box_arr[box_count].height = box_loc[2];

			box_arr[box_count].width = box_loc[3];

			box_arr[box_count].perimeter = 2 * (box_arr[box_count].height + box_arr[box_count].width);
			//printf("Box left_X, left_y, height, width: %d, %d, %d, %d\n", box_arr[box_count].up_left_x, box_arr[box_count].up_left_y, box_arr[box_count].height, box_arr[box_count].width);

			// Get top neighbours
			fflush(stdin);
			fgets(line1, sizeof(line1), fd);
			int top_num;
			top_num = parsefirst(line1);
			box_arr[box_count].num_top = top_num;
			int* toparr = (int*)malloc(top_num * sizeof(int));
			int* toparrov = (int*)malloc(top_num * sizeof(int));
			parseline(toparr, line1, 1);
			box_arr[box_count].top_ids = toparr;
			box_arr[box_count].top_ov = toparrov;
			//printf("Box top neightbours: %d ==> ", box_arr[box_count].num_top);
			if (top_num == 0) {
				box_arr[box_count].top_ids = NULL;
				//printf("No Top neighbours.");
			}

			// Get bottom neighbours
			fflush(stdin);
			fgets(line1, sizeof(line1), fd);
			int bottom_num;
			bottom_num = parsefirst(line1);
			box_arr[box_count].num_bottom = bottom_num;
			int* bottomarr = (int*)malloc(bottom_num * sizeof(int));
			int* bottomarrov = (int*)malloc(bottom_num * sizeof(int));
			parseline(bottomarr, line1, 1);
			box_arr[box_count].bottom_ids = bottomarr;
			box_arr[box_count].bottom_ov = bottomarrov;
			//printf("\nBox bottom neightbours: %d ==> ", box_arr[box_count].num_bottom);
			if (bottom_num == 0) {
				box_arr[box_count].bottom_ids = NULL;
				//printf("No bottom neighbours.");
			}

			// Get left neighbours
			fflush(stdin);
			fgets(line1, sizeof(line1), fd);
			int left_num;
			left_num = parsefirst(line1);
			box_arr[box_count].num_left = left_num;
			int* leftarr = (int*)malloc(left_num * sizeof(int));
			int* leftarrov = (int*)malloc(left_num * sizeof(int));

			parseline(leftarr, line1, 1);
			box_arr[box_count].left_ids = leftarr;
			box_arr[box_count].left_ov = leftarrov;
			//printf("\nBox left neightbours: %d ==> ", box_arr[box_count].num_left);
			if (left_num == 0) {
				box_arr[box_count].left_ids = NULL;
				//printf("No left neighbours.");
			}

			// Get right neighbours
			fflush(stdin);
			fgets(line1, sizeof(line1), fd);
			int right_num;
			right_num = parsefirst(line1);
			box_arr[box_count].num_right = right_num;
			int* rightarr = (int*)malloc(right_num * sizeof(int));
			int* rightarrov = (int*)malloc(right_num * sizeof(int));
			parseline(rightarr, line1, 1);
			box_arr[box_count].right_ids = rightarr;
			box_arr[box_count].right_ov = rightarrov;
			//printf("\nBox right neightbours: %d ==> ", box_arr[box_count].num_right);
			if (right_num == 0) {

				box_arr[box_count].right_ids = NULL;
				//printf("No right neighbours.");
			}
			else {
				int j;
				for (j = 0; j < box_arr[box_count].num_right; j++) {
					//printf("%d, ",box_arr[box_count].right_ids[j]);
				}
			}

			// Get dsv value
			fflush(stdin);
			fgets(line1, sizeof(line1), fd);

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
	char c;
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
			//printf("%d:",number);
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

int** alloc_2d_init(int rows, int cols) {
	int* data = (int*)malloc(rows * cols * sizeof(int));
	int** array = (int**)malloc(rows * sizeof(int*));
	for (int i = 0; i < rows; i++)
		array[i] = &(data[cols * i]);

	return array;
}

void reformat_overlap(Box* box_arr, int* num_nbor, double* dsv, int** nbor_ids, int** overlap, int* perimeter) {
	for (int i = 0; i < NUM_BOXES; i++) {

		num_nbor[i] = box_arr[i].num_top + box_arr[i].num_bottom + box_arr[i].num_left + box_arr[i].num_right;
		if (box_arr[i].num_top == 0) num_nbor[i]++;
		if (box_arr[i].num_bottom == 0) num_nbor[i]++;
		if (box_arr[i].num_left == 0) num_nbor[i]++;
		if (box_arr[i].num_right == 0) num_nbor[i]++;

		dsv[i] = box_arr[i].dsv;
		perimeter[i] = box_arr[i].perimeter;

		int k = 0;
		if (box_arr[i].num_top == 0) {
			nbor_ids[i][k] = box_arr[i].id;
			overlap[i][k] = box_arr[i].width;
			k++;
		}
		else {
			for (int j = 0; j < box_arr[i].num_top; j++) {
				nbor_ids[i][k] = box_arr[i].top_ids[j];
				overlap[i][k] = box_arr[i].top_ov[j];
				k++;
			}
		}
		if (box_arr[i].num_bottom == 0) {
			nbor_ids[i][k] = box_arr[i].id;
			overlap[i][k] = box_arr[i].width;
			k++;
		}
		else {
			for (int j = 0; j < box_arr[i].num_bottom; j++) {
				nbor_ids[i][k] = box_arr[i].bottom_ids[j];
				overlap[i][k] = box_arr[i].bottom_ov[j];
				k++;
			}
		}
		if (box_arr[i].num_left == 0) {
			nbor_ids[i][k] = box_arr[i].id;
			overlap[i][k] = box_arr[i].height;
			k++;
		}
		else {
			for (int j = 0; j < box_arr[i].num_left; j++) {
				nbor_ids[i][k] = box_arr[i].left_ids[j];
				overlap[i][k] = box_arr[i].left_ov[j];
				k++;
			}
		}
		if (box_arr[i].num_right == 0) {
			nbor_ids[i][k] = box_arr[i].id;
			overlap[i][k] = box_arr[i].height;
			k++;
		}
		else
		{
			for (int j = 0; j < box_arr[i].num_right; j++) {
				nbor_ids[i][k] = box_arr[i].right_ids[j];
				overlap[i][k] = box_arr[i].right_ov[j];
				k++;
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

void printarrays(int* num_nbor, double* dsv, int** nbor_ids, int** overlap, int* perimeter) {

	for (int i = 0; i < NUM_BOXES; i++) {
		printf("\nDescription of Box id: %d which has %d neighbors\n", i, num_nbor[i]);
		printf("Neighbors are:- ");
		for (int j = 0; j < num_nbor[i]; j++) {
			printf("Box id: %d --> Overlap: %d || ", nbor_ids[i][j], overlap[i][j]);
		}
		printf("\nPerimeter is: %d. DSV is: %lf\n", perimeter[i], dsv[i]);
	}
}
