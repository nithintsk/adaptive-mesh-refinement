#ifndef PERSISTENT
#define PERSISTENT

#define MAXLEN 500
#define NUM_THREADS 8
//#define NUM_THREADS 500

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ int atomicAdd(int* a, int b) { return b; }
#endif

struct Box {
	int id;
	int up_left_x, up_left_y;
	int height, width, perimeter;
	int num_top, num_bottom, num_left, num_right;
	int* top_ids, * bottom_ids, * left_ids, * right_ids;
	int* top_ov, * bottom_ov, * left_ov, * right_ov;
	double waat;
	double dsv;
};

typedef struct Box Box;

extern int NUM_BOXES;
extern int NUM_ROWS;
extern int NUM_COLS;
extern double AFFECT_RATE;
extern double EPSILON;
extern double MIN_DSV;
extern double MAX_DSV;

void readgridparam();
void populate(Box*);
void parseline(int* num, char* path, int func);
int parsefirst(char* path);
double parsedsv(char* path);
void calcoverlap(Box*);
void printboxes(Box*);
int cuda_compute_commit_dsv(Box* box_arr, int &numboxes, double &affectrate, double &epsilon, double &cmin, double &cmax);

#endif
