CC=icc
CFLAGS= -fopenmp -O3 -lrt

#all: persistent disposable
all: disposable persistent

persistent: pmain.c persistent.c persistent.h
	$(CC) $(CFLAGS) -o persistent persistent.c pmain.c
disposable: dmain.c disposable.c disposable.h
	$(CC) $(CFLAGS) -o disposable disposable.c dmain.c
