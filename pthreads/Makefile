CC=icc
CFLAGS= -lpthread -O3 -lrt

all: persistent disposable

persistent: persistent.c
	$(CC) $(CFLAGS) -o persistent persistent.c
disposable: disposable.c
	$(CC) $(CFLAGS) -o disposable disposable.c
