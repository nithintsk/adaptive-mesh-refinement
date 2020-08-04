#include "persistent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <omp.h>

int compute_commit_dsv(Box* box_arr) {

	int count = 1;
    int thread_count;
	//omp_set_dynamic(0);     // Explicitly disable dynamic teams
	omp_set_num_threads(NUM_THREADS);
    
	#pragma omp parallel shared(box_arr) 
		{
            #pragma omp master
                thread_count=omp_get_num_threads();
			do {
				#pragma omp for
					for (int i = 0; i < NUM_BOXES; i++) {
						box_arr[i].waat = 0;

						// Get weighted average of top neighbours 
						if (box_arr[i].num_top != 0) {
							int j;
							for (j = 0; j < box_arr[i].num_top; j++) {
								int cur_topid = box_arr[i].top_ids[j];
								int overlap = box_arr[i].top_ov[j];
								box_arr[i].waat = box_arr[i].waat + box_arr[cur_topid].dsv * overlap;
							}
						}
						else {
							box_arr[i].waat = box_arr[i].waat + box_arr[i].width * box_arr[i].dsv;
						}

						// Get weighted average of bottom neighbours
						if (box_arr[i].num_bottom != 0) {
							int j;
							for (j = 0; j < box_arr[i].num_bottom; j++) {
								int cur_bottomid = box_arr[i].bottom_ids[j];
								int overlap = box_arr[i].bottom_ov[j];
								box_arr[i].waat = box_arr[i].waat + box_arr[cur_bottomid].dsv * overlap;
							}
						}
						else {
							box_arr[i].waat = box_arr[i].waat + box_arr[i].width * box_arr[i].dsv;
						}

						// Get weighted average of left neighbours
						if (box_arr[i].num_left != 0) {
							int j;
							for (j = 0; j < box_arr[i].num_left; j++) {
								int cur_leftid = box_arr[i].left_ids[j];
								int overlap = box_arr[i].left_ov[j];
								box_arr[i].waat = box_arr[i].waat + box_arr[cur_leftid].dsv * overlap;
							}
						}
						else {
							box_arr[i].waat = box_arr[i].waat + box_arr[i].height * box_arr[i].dsv;
						}

						// Get weighted average of right neighbours
						if (box_arr[i].num_right != 0) {
							int j;
							for (j = 0; j < box_arr[i].num_right; j++) {
								int cur_rightid = box_arr[i].right_ids[j];
								int overlap = box_arr[i].right_ov[j];
								box_arr[i].waat = box_arr[i].waat + box_arr[cur_rightid].dsv * overlap;
							}
						}
						else {
							box_arr[i].waat = box_arr[i].waat + box_arr[i].height * box_arr[i].dsv;
						}

						// Find the weighted average by dividing with the perimeter
						box_arr[i].waat = box_arr[i].waat / box_arr[i].perimeter;

					}

				#pragma omp master
				{
					MIN_DSV = INT_MAX;
					MAX_DSV = INT_MIN;
					count++;
				}

				#pragma omp barrier

				#pragma omp for reduction(max:MAX_DSV) reduction(min:MIN_DSV)
					for (int i = 0; i < NUM_BOXES; i++) {
						if (box_arr[i].waat > box_arr[i].dsv) {
							box_arr[i].dsv = box_arr[i].dsv + AFFECT_RATE * (box_arr[i].waat - box_arr[i].dsv);
						}
						else {
							box_arr[i].dsv = box_arr[i].dsv - AFFECT_RATE * (box_arr[i].dsv - box_arr[i].waat);
						}
						if (box_arr[i].dsv < MIN_DSV) MIN_DSV = box_arr[i].dsv;
						if (box_arr[i].dsv > MAX_DSV) MAX_DSV = box_arr[i].dsv;
					}

				//if (count == 2) break;
				if (MAX_DSV == 0) break;
			} while (((MAX_DSV - MIN_DSV) / MAX_DSV) > EPSILON);
			count--;
		}
    printf("A total of %d threads were created.\n", thread_count);
	return count;
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
	/*int i=0, digit;
	double number = 0;
	do {
		digit = path[i] - '0';
		number = number*10 + digit;
		i++;
	} while (i<strlen(path) && (path[i]>='0' && path[i]<='9' || ));
	return number;*/
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

				box_arr[i].top_ov[j] = abs(len2 - len1);
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

				// find left most of x_left + width and xbottom_left + its width
				if ((box_arr[i].up_left_x + box_arr[i].width) <= (box_arr[cur_bottomid].up_left_x + box_arr[cur_bottomid].width)) len2 = (box_arr[i].up_left_x + box_arr[i].width);
				else len2 = (box_arr[cur_bottomid].up_left_x + box_arr[cur_bottomid].width);

				box_arr[i].bottom_ov[j] = abs(len2 - len1);
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

				// find top most of y_left + height and yleft_left + its height
				if ((box_arr[i].up_left_y + box_arr[i].height) <= (box_arr[cur_leftid].up_left_y + box_arr[cur_leftid].height)) len2 = (box_arr[i].up_left_y + box_arr[i].height);
				else len2 = (box_arr[cur_leftid].up_left_y + box_arr[cur_leftid].height);

				box_arr[i].left_ov[j] = abs(len2 - len1);
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

				// find top most of y_left + height and yright_left + its height
				if ((box_arr[i].up_left_y + box_arr[i].height) <= (box_arr[cur_rightid].up_left_y + box_arr[cur_rightid].height)) len2 = (box_arr[i].up_left_y + box_arr[i].height);
				else len2 = (box_arr[cur_rightid].up_left_y + box_arr[cur_rightid].height);

				box_arr[i].right_ov[j] = abs(len2 - len1);
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
