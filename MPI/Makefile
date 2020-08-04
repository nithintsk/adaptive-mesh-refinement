CC=mpicc
CFLAGS= -qopenmp -O3 -lrt

#all: persistent disposable
all: disposable

disposable: dmain.c disposable.c disposable.h
	$(CC) $(CFLAGS) -o mpiprog disposable.c dmain.c
