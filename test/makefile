# CFLAGS for CC
CFLAGS=-lm -std=c++17 -Wextra -O3 -fopenmp

#includeflags
IFLAGS=-I../include

# Compiler
CCo=g++ -fPIC -c $(CFLAGS) $(IFLAGS)
CC=g++ -fPIC $(CFLAGS) $(IFLAGS)
CCWo=x86_64-w64-mingw32-g++-win32 -c $(CFLAGS) $(IFLAGS)
CCW=x86_64-w64-mingw32-g++-win32 $(CFLAGS) $(IFLAGS)

OS=$(shell uname -s)
ifeq ($(OS),Linux) 
	LDFLAGS=-Wl,-rpath='$$ORIGIN/../lib' -L ../lib -lann
else 
	LDFLAGS=-L ../lib -lann
endif

all:
	$(CC) matdot.cc -o matdot	$(LDFLAGS) 
	./matdot

clean:
	-rm -v matdot
