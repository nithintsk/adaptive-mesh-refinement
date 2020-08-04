CC=nvcc
CFLAGS= -rdc=true -O -arch=sm_60

all: persistent

persistent: pmain.cu persistent.cu persistent.h
	$(CC) $(CFLAGS) -o persistent persistent.cu pmain.cu
