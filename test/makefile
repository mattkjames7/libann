#includeflags
IFLAGS=-I../include

# Compiler

CCWo=x86_64-w64-mingw32-g++-win32 -c $(CFLAGS) $(IFLAGS)
CCW=x86_64-w64-mingw32-g++-win32 $(CFLAGS) $(IFLAGS)

OS=$(shell uname -s)
ifeq ($(OS),Linux)
	LDFLAGS=-Wl,-rpath='$$ORIGIN/../lib' -L ../lib -lann
	CFLAGS=-lm -std=c++17 -Wextra -O3 -fopenmp
	CCo=g++ -fPIC -c $(CFLAGS) $(IFLAGS)
	CC=g++ -fPIC $(CFLAGS) $(IFLAGS)
else
	LDFLAGS=-L ../lib -lann -L/usr/local/opt/libomp/lib -lomp
	CFLAGS=-lm -std=c++17 -Wextra -O3 "-I/usr/local/opt/libomp/include"
	CCo=g++ -Xpreprocessor -fopenmp -c $(CFLAGS) $(IFLAGS)
	CC=g++ -Xpreprocessor -fopenmp $(CFLAGS) $(IFLAGS)
endif

all:
	$(CC) matdot.cc -o matdot	$(LDFLAGS)
	./matdot

clean:
	-rm -v matdot
