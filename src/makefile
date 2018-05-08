CC = gcc
#Using -Ofast instead of -O2 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O2 -march=native -Wall -funroll-loops -Wno-unused-result

all: lmm-a lmm-s lmm-m

lmm-a : lmm-a.c
	$(CC) lmm-a.c -o lmm-a $(CFLAGS)

lmm-s : lmm-s.c
	$(CC) lmm-s.c -o lmm-s $(CFLAGS)
	
lmm-m : lmm-m.c
	$(CC) lmm-m.c -o lmm-m $(CFLAGS)
clean:
	rm -rf lmm-a lmm-s lmm-m
