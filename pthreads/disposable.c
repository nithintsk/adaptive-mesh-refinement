#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#define MAXLEN 500

// Structure holding description of each box
// PENDING - Need to create a destroy function to deallocate memory later.
struct Box {
    int id;
    int up_left_x, up_left_y;
    int height, width, perimeter;
    int num_top, num_bottom, num_left, num_right;
    int *top_ids, *bottom_ids, *left_ids, *right_ids;
    int *top_ov, *bottom_ov, *left_ov, *right_ov;
    double waat;
    double dsv;
};

struct Targs {
	struct Box *boxarr;
	int tid;
	double min_DSV;
	double max_DSV;
};

typedef struct Box Box;
typedef struct Targs Targs;
void readgridparam();
void populate(Box*);
void parseline(int *num, char* path, int func);
int parsefirst(char* path);
double parsedsv(char* path);
//void* commitdsv(void*);
void* compute_commit_dsv(void*);
void calcoverlap(Box*);
void printboxes(Box*);

int NUM_BOXES;
int NUM_ROWS;
int NUM_COLS;
int NUM_THREADS;
int WORKLOAD;
double AFFECT_RATE;
double EPSILON;
double MIN_DSV;
double MAX_DSV;
pthread_barrier_t mybarrier;

int main (int argc, char *argv[])
{
    AFFECT_RATE = strtod(argv[1],NULL);
    EPSILON = strtod(argv[2],NULL);
    NUM_THREADS = (int)strtod(argv[3],NULL);
    
    // Initialize pthread array to store pthread id
    pthread_t thread[NUM_THREADS];
    struct Targs targs[NUM_THREADS];
    
    // Get the parameters describing the grid --> First line in data file
    readgridparam();
    
    // Create a Box array of size NUM_BOXES
    struct Box boxes[NUM_BOXES];
    WORKLOAD = (int)ceil((float)NUM_BOXES/(float)NUM_THREADS);
    //printf("Created %d number of boxes. Number of Rows: %d. Number of Columns: %d.\n", NUM_BOXES, NUM_ROWS, NUM_COLS);
    
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
    	// Create NUM_THREADS number of threads using pthread_create
    	int tn;
    	pthread_barrier_init(&mybarrier, NULL, NUM_THREADS);
    	for (tn=0; tn<NUM_THREADS; tn++) {
    		targs[tn].boxarr = boxes;
    		targs[tn].tid = tn;
    		targs[tn].max_DSV = INT_MIN;
    		targs[tn].min_DSV = INT_MAX;
    		pthread_create( &thread[tn], NULL, compute_commit_dsv, (void*) &targs[tn]);
		}
		
		// Wait for threads to reach barrier
		//pthread_barrier_wait(&mybarrier);
        
        // Join and exit threads
        void *th_status;
        for (tn = 0; tn < NUM_THREADS; tn++)
		{
			pthread_join(thread[tn], &th_status);
		}
		
		// Destroy the barrier and initialize in the next iteration
		pthread_barrier_destroy(&mybarrier);
		
		// Calculate MIN_DSV and MAX_DSV
		MAX_DSV=targs[0].max_DSV;
    	MIN_DSV=targs[0].min_DSV;
    	for (tn = 1; tn < NUM_THREADS; tn++)
		{
			if (targs[tn].max_DSV > MAX_DSV) MAX_DSV = targs[tn].max_DSV;
			if (targs[tn].min_DSV < MIN_DSV) MIN_DSV = targs[tn].min_DSV;
		}
        
        printf("\nCount: %d\n", count);
        printf("MAX_DSV: %lf\n", MAX_DSV);
        printf("MIN_DSV: %lf\n", MIN_DSV);
        printf("EPSILON: %lf\n", EPSILON);
        count++;
        //if (count == 2) break;
        if (MAX_DSV == 0) break;
    } while (((MAX_DSV-MIN_DSV) / MAX_DSV) > EPSILON);
    count--;
	
	time(&end_time);
    clock_time = clock() - clock_time;
    clock_gettime(CLOCK_REALTIME,&t_end);
    diff = (double)(((t_end.tv_sec - t_start.tv_sec) * CLOCKS_PER_SEC) + ((t_end.tv_nsec - t_start.tv_nsec) / 1000));
	
	printf("\n\n*************************************************************************\n");
    printf("Dissipation converged in %d iterations.\n", count);
    printf("     with max DSV = %lf and min DSV = %lf\n", MAX_DSV, MIN_DSV);
    printf("     affect rate = %lf;     epsilon = %lf\n", AFFECT_RATE, EPSILON);
    printf("Elapsed convergence loop time (clock): %lu\n",clock_time);
    printf("Elapsed convergence loop time (time): %lf\n",difftime(end_time, start_time));
    printf("Elapsed convergence loop time (chrono): %lf\n",diff);
	
    // Print contents of the structure
    // printboxes(boxes);
}

void* compute_commit_dsv(void* arg1) {
	struct Targs* arg = (Targs*) arg1;
	Box* box_arr = arg->boxarr;
	int id = arg->tid;
    int i;
    //printf("Initiated thread id: %d. Start index: %d. End Index: %d\n", id, id*WORKLOAD, (id+1)*WORKLOAD);
    
    for(i=id*WORKLOAD; (i < (id+1)*WORKLOAD) && (i < NUM_BOXES); i++) {
        box_arr[i].waat = 0;
        
        // Get weighted average of top neighbours
        if (box_arr[i].num_top!=0) {
            int j;
            for (j=0; j<box_arr[i].num_top; j++) {
                int cur_topid = box_arr[i].top_ids[j];
                int overlap =  box_arr[i].top_ov[j];
                box_arr[i].waat = box_arr[i].waat + box_arr[cur_topid].dsv * overlap;
            }
        } else {
            box_arr[i].waat = box_arr[i].waat + box_arr[i].width * box_arr[i].dsv;
        }
        
        // Get weighted average of bottom neighbours
        if (box_arr[i].num_bottom!=0) {
            int j;
            for (j=0; j<box_arr[i].num_bottom; j++) {
                int cur_bottomid = box_arr[i].bottom_ids[j];
                int overlap =  box_arr[i].bottom_ov[j];
                box_arr[i].waat = box_arr[i].waat + box_arr[cur_bottomid].dsv * overlap;
            }
        } else {
            box_arr[i].waat = box_arr[i].waat + box_arr[i].width * box_arr[i].dsv;
        }
        
        // Get weighted average of left neighbours
        if (box_arr[i].num_left!=0) {
            int j;
            for (j=0; j<box_arr[i].num_left; j++) {
                int cur_leftid = box_arr[i].left_ids[j];
                int overlap =  box_arr[i].left_ov[j];
                box_arr[i].waat = box_arr[i].waat + box_arr[cur_leftid].dsv * overlap;
            }
        } else {
            box_arr[i].waat = box_arr[i].waat + box_arr[i].height * box_arr[i].dsv;
        }
        
        // Get weighted average of right neighbours
        if (box_arr[i].num_right!=0) {
            int j;
            for (j=0; j<box_arr[i].num_right; j++) {
                int cur_rightid = box_arr[i].right_ids[j];
                int overlap =  box_arr[i].right_ov[j];
                box_arr[i].waat = box_arr[i].waat + box_arr[cur_rightid].dsv * overlap;
            }
        } else {
            box_arr[i].waat = box_arr[i].waat + box_arr[i].height * box_arr[i].dsv;
        }
        
        // Find the weighted average by dividing with the perimeter
        box_arr[i].waat = box_arr[i].waat / box_arr[i].perimeter;
        
    }
    
    //printf("Thread %d reached barrier.\n", id);
    pthread_barrier_wait(&mybarrier);
    //printf("Thread %d crossed barrier.\n", id);
    
    //commitdsv moved here
    double LOCAL_MIN_DSV = arg->min_DSV;
    double LOCAL_MAX_DSV = arg->max_DSV;
    
    for(i=id*WORKLOAD; (i < (id+1)*WORKLOAD) && (i < NUM_BOXES); i++) {
        if (box_arr[i].waat > box_arr[i].dsv) {
            box_arr[i].dsv = box_arr[i].dsv + AFFECT_RATE * (box_arr[i].waat - box_arr[i].dsv);
        }
        else {
            box_arr[i].dsv = box_arr[i].dsv - AFFECT_RATE * (box_arr[i].dsv - box_arr[i].waat);
        }
        if (box_arr[i].dsv < LOCAL_MIN_DSV) LOCAL_MIN_DSV = box_arr[i].dsv;
        if (box_arr[i].dsv > LOCAL_MAX_DSV) LOCAL_MAX_DSV = box_arr[i].dsv;
    }
    
    arg->min_DSV = LOCAL_MIN_DSV;
    arg->max_DSV = LOCAL_MAX_DSV;
    
}

void calcoverlap(struct Box *box_arr) {
    int i;
    for(i=0; i < NUM_BOXES; i++) {
        // Calculate TOP overlap for each node.
        // If 0, skip.
        if (box_arr[i].num_top!=0) {
            int j;
            for (j=0; j<box_arr[i].num_top; j++) {
                // find right most of x_left and xtop_left
                int cur_topid = box_arr[i].top_ids[j];
                int len2,len1;
                if (box_arr[i].up_left_x >= box_arr[cur_topid].up_left_x) len1 = box_arr[i].up_left_x;
                else len1 = box_arr[cur_topid].up_left_x;
                //printf("Box id %d. Len1 value %d.\n",i,len1);
                
                if ((box_arr[i].up_left_x + box_arr[i].width) <= (box_arr[cur_topid].up_left_x + box_arr[cur_topid].width)) len2 = (box_arr[i].up_left_x + box_arr[i].width);
                else len2 = (box_arr[cur_topid].up_left_x + box_arr[cur_topid].width);
                //printf("Box id %d. Len2 value %d.\n",i,len2);
                
                box_arr[i].top_ov[j] = abs(len2-len1);
                //printf("Box id %d. Neighbor id %d. Value %d.\n",i,cur_topid,box_arr[i].top_ov[j]);
            }
        }
        
        // Calculate BOTTOM overlap for each node.
        // If 0, skip.
        if (box_arr[i].num_bottom !=0) {
            int j;
            for (j=0; j<box_arr[i].num_bottom; j++) {
                // find right most of x_left and xbottom_left
                int cur_bottomid = box_arr[i].bottom_ids[j];
                int len2,len1;
                if (box_arr[i].up_left_x >= box_arr[cur_bottomid].up_left_x) len1 = box_arr[i].up_left_x;
                else len1 = box_arr[cur_bottomid].up_left_x;
                //printf("Box id %d. Len1 value %d.\n",i,len1);
                
                // find left most of x_left + width and xbottom_left + its width
                if ((box_arr[i].up_left_x + box_arr[i].width) <= (box_arr[cur_bottomid].up_left_x + box_arr[cur_bottomid].width)) len2 = (box_arr[i].up_left_x + box_arr[i].width);
                else len2 = (box_arr[cur_bottomid].up_left_x + box_arr[cur_bottomid].width);
                //printf("Box id %d. Len2 value %d.\n",i,len2);
                
                box_arr[i].bottom_ov[j] = abs(len2-len1);
                //printf("Box id %d. Neighbor id %d. Value %d.\n",i,cur_bottomid,box_arr[i].bottom_ov[j]);
            }
        }
        
        // Calculate left overlap for each node.
        // If 0, skip.
        if (box_arr[i].num_left !=0) {
            int j;
            for (j=0; j<box_arr[i].num_left; j++) {
                // find bottom most of y_left and yleft_left
                int cur_leftid = box_arr[i].left_ids[j];
                int len2,len1;
                if (box_arr[i].up_left_y >= box_arr[cur_leftid].up_left_y) len1 = box_arr[i].up_left_y;
                else len1 = box_arr[cur_leftid].up_left_y;
                //printf("Box id %d. Len1 value %d.\n",i,len1);
                
                // find top most of y_left + height and yleft_left + its height
                if ((box_arr[i].up_left_y + box_arr[i].height) <= (box_arr[cur_leftid].up_left_y + box_arr[cur_leftid].height)) len2 = (box_arr[i].up_left_y + box_arr[i].height);
                else len2 = (box_arr[cur_leftid].up_left_y + box_arr[cur_leftid].height);
                //printf("Box id %d. Len2 value %d.\n",i,len2);
                
                box_arr[i].left_ov[j] = abs(len2-len1);
                //printf("Box id %d. Neighbor id %d. Value %d.\n",i,cur_leftid,box_arr[i].left_ov[j]);
            }
        }
        
        // Calculate right overlap for each node.
        // If 0, skip.
        if (box_arr[i].num_right !=0) {
            int j;
            for (j=0; j<box_arr[i].num_right; j++) {
                // find bottom most of y_left and yright_left
                int cur_rightid = box_arr[i].right_ids[j];
                int len2,len1;
                if (box_arr[i].up_left_y >= box_arr[cur_rightid].up_left_y) len1 = box_arr[i].up_left_y;
                else len1 = box_arr[cur_rightid].up_left_y;
                //printf("Box id %d. Len1 value %d.\n",i,len1);
                
                // find top most of y_left + height and yright_left + its height
                if ((box_arr[i].up_left_y + box_arr[i].height) <= (box_arr[cur_rightid].up_left_y + box_arr[cur_rightid].height)) len2 = (box_arr[i].up_left_y + box_arr[i].height);
                else len2 = (box_arr[cur_rightid].up_left_y + box_arr[cur_rightid].height);
                //printf("Box id %d. Len2 value %d.\n",i,len2);
                
                box_arr[i].right_ov[j] = abs(len2-len1);
                //printf("Box id %d. Neighbor id %d. Value %d.\n",i,cur_rightid,box_arr[i].right_ov[j]);
            }
        }
    }
}

void printboxes(struct Box *box_arr) {
    
    int i;
    for(i = 0; i < NUM_BOXES; i++) {
        printf("================================");
        printf("\n\nBox id: %d\n", box_arr[i].id);
        printf("Box left_X, left_y, height, width, perimiter: %d, %d, %d, %d, %d\n", box_arr[i].up_left_x, box_arr[i].up_left_y, box_arr[i].height, box_arr[i].width, box_arr[i].perimeter);
        
        printf("Box top neighbours and overlap: ");
        int j;
        for(j=0; j<box_arr[i].num_top; j++) {
            printf("%d:%d, ",box_arr[i].top_ids[j], box_arr[i].top_ov[j]);
        }
        printf("\n");
        
        printf("Box bottom neighbours and overlap: ");
        for(j=0; j<box_arr[i].num_bottom; j++) {
            printf("%d:%d, ",box_arr[i].bottom_ids[j], box_arr[i].bottom_ov[j]);
        }
        printf("\n");
        
        printf("Box left neighbours: ");
        for(j=0; j<box_arr[i].num_left; j++) {
            printf("%d:%d, ",box_arr[i].left_ids[j], box_arr[i].left_ov[j]);
        }
        printf("\n");
        
        printf("Box right neighbours: ");
        for(j=0; j<box_arr[i].num_right; j++) {
            printf("%d:%d, ",box_arr[i].right_ids[j], box_arr[i].right_ov[j]);
        }
        printf("\n");
        
        printf("Box dsv value: %lf", box_arr[i].dsv);
        printf("\n");
        
    }
}

double parsedsv(char* path) {
    return number;*/
    double number = 0;
    number = strtod(path,NULL);
    return number;
}

int parsefirst(char* path) {
    int i=0, digit, number = 0;
    do {
		digit = path[i] - '0';
		number = number*10 + digit;
		i++;
    } while (i<strlen(path) && path[i]>='0' && path[i]<='9');
    return number;
}

void parseline(int *num, char* path, int func) {
    char c;
    int i=0,digit,number=0;
    int num_count=0;
    if (func == 0) i=0;
    if (func == 1) {
        while (i<strlen(path) && path[i]>='0' && path[i]<='9') {
            i++;
        }
    }
    for(; i<strlen(path); i++)
    {
    	if(path[i]>='0' && path[i]<='9') //to confirm it's a digit
    	{
    	    number = 0;
    	    do {
        		digit = path[i] - '0';
        		number = number*10 + digit;
        		i++;
    	    } while (i<strlen(path) && path[i]>='0' && path[i]<='9');
    	    //printf("%d:",number);
    	    num[num_count] = number;
    	    num_count++;
    	}
    }
}

void populate(Box* box_arr) {
    
	char line1[MAXLEN] = "";
	int box_count = 0;
    // Read rest of file and populate the data structure
    fflush(stdin);
    while (fgets(line1,sizeof(line1),stdin)) {
        if (line1[0] == '-') {
            break;
        } 
        else if (!strcmp(line1,"")) continue;
        else if (!(line1[0]>='0' && line1[0]<='9')) continue;
        else {
            // Get Box id;
            int id[1];
            parseline(id,line1,0);
            box_arr[box_count].id = id[0];
            //printf("\n\nBox id: %d. Box count: %d\n", box_arr[box_count].id, box_count);
            
            // Get location, height and width
            fflush(stdin);
            fgets(line1,sizeof(line1),stdin);
            int box_loc[4];
            parseline(box_loc,line1,0);
            box_arr[box_count].up_left_y = box_loc[0];
            box_arr[box_count].up_left_x = box_loc[1];
            box_arr[box_count].height = box_loc[2];
            box_arr[box_count].width = box_loc[3];
            box_arr[box_count].perimeter = 2*(box_arr[box_count].height + box_arr[box_count].width);
            //printf("Box left_X, left_y, height, width: %d, %d, %d, %d\n", box_arr[box_count].up_left_x, box_arr[box_count].up_left_y, box_arr[box_count].height, box_arr[box_count].width);
            
            // Get top neighbours
            fflush(stdin);
            fgets(line1,sizeof(line1),stdin);
            int top_num;
            top_num = parsefirst(line1);
            box_arr[box_count].num_top = top_num;
            int *toparr = (int *)malloc(top_num * sizeof(int));
            int *toparrov = (int *)malloc(top_num * sizeof(int));
            parseline(toparr, line1,1);
            box_arr[box_count].top_ids = toparr;
            box_arr[box_count].top_ov = toparrov;
            //printf("Box top neightbours: %d ==> ", box_arr[box_count].num_top);
            if (top_num == 0) {
                box_arr[box_count].top_ids = NULL;
                //printf("No Top neighbours.");
            }
            else {
                int j;
                for(j=0; j<box_arr[box_count].num_top; j++) {
                        //printf("%d, ",box_arr[box_count].top_ids[j]);
                }
            }
            
            // Get bottom neighbours
            fflush(stdin);
            fgets(line1,sizeof(line1),stdin);
            int bottom_num;
            bottom_num = parsefirst(line1);
            box_arr[box_count].num_bottom = bottom_num;
            int *bottomarr = (int *)malloc(bottom_num * sizeof(int));
            int *bottomarrov = (int *)malloc(bottom_num * sizeof(int));
            parseline(bottomarr, line1,1);
            box_arr[box_count].bottom_ids = bottomarr;
            box_arr[box_count].bottom_ov = bottomarrov;
            //printf("\nBox bottom neightbours: %d ==> ", box_arr[box_count].num_bottom);
            if (bottom_num == 0) {
                box_arr[box_count].bottom_ids = NULL;
                //printf("No bottom neighbours.");
            }
            else {
                int j;
                for(j=0; j<box_arr[box_count].num_bottom; j++) {
                        //printf("%d, ",box_arr[box_count].bottom_ids[j]);
                }
            }
            
            // Get left neighbours
            fflush(stdin);
            fgets(line1,sizeof(line1),stdin);
            int left_num;
            left_num = parsefirst(line1);
            box_arr[box_count].num_left = left_num;
            int *leftarr = (int *)malloc(left_num * sizeof(int));
            int *leftarrov = (int *)malloc(left_num * sizeof(int));
            parseline(leftarr, line1,1);
            box_arr[box_count].left_ids = leftarr;
            box_arr[box_count].left_ov = leftarrov;
            //printf("\nBox left neightbours: %d ==> ", box_arr[box_count].num_left);
            if (left_num == 0) {
                box_arr[box_count].left_ids = NULL;
                //printf("No left neighbours.");
            }
            else {
                int j;
                for(j=0; j<box_arr[box_count].num_left; j++) {
                        //printf("%d, ",box_arr[box_count].left_ids[j]);
                }
            }
            
            // Get right neighbours
            fflush(stdin);
            fgets(line1,sizeof(line1),stdin);
            int right_num;
            right_num = parsefirst(line1);
            box_arr[box_count].num_right = right_num;
            int *rightarr = (int *)malloc(right_num * sizeof(int));
            int *rightarrov = (int *)malloc(right_num * sizeof(int));
            parseline(rightarr, line1,1);
            box_arr[box_count].right_ids = rightarr;
            box_arr[box_count].right_ov = rightarrov;
            //printf("\nBox right neightbours: %d ==> ", box_arr[box_count].num_right);
            if (right_num == 0) {
                box_arr[box_count].right_ids = NULL;
                //printf("No right neighbours.");
            }
            else {
                int j;
                for(j=0; j<box_arr[box_count].num_right; j++) {
                        //printf("%d, ",box_arr[box_count].right_ids[j]);
                }
            }
            
            // Get dsv value
            fflush(stdin);
            fgets(line1,sizeof(line1),stdin);
            double dsv_val;
            dsv_val = parsedsv(line1);
            box_arr[box_count].dsv = dsv_val;
            
            // Move to next box
            box_count++;
            fflush(stdin);
        }
    }
}

void readgridparam() {
	
	// Assuming each line in the datafile is Max 500 characters
    char line[MAXLEN] = "";
	
	fflush(stdin);
    if (fgets(line,sizeof(line),stdin)) {
        // If the first line of the file contains -1, exit
        if (line[0] == '-') {
            fprintf(stderr, "First line of the file contains -1. Exiting....");
            exit(EXIT_FAILURE); 
        } else {
            // We only expect 3 numbers in the first line
            // <number of grid boxes> <num_grid_rows> <num_grid_cols>
            int arr[3];
            parseline(arr,line,0);
            NUM_BOXES = arr[0];
            NUM_ROWS = arr[1];
            NUM_COLS = arr[2];
        }
    } else {
        fprintf(stderr, "File may not exist or is empty. Exiting....");
        exit(EXIT_FAILURE); 
    }
    
}
