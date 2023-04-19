# CFLAGS for CC
CFLAGS=-lm -fPIC -std=c++17 -Wextra -O3 -lomp

# Compiler

CCWo=x86_64-w64-mingw32-g++-win32 -c $(CFLAGS)
CCW=x86_64-w64-mingw32-g++-win32 $(CFLAGS)

ifndef BUILDDIR
	BUILDDIR=../build
endif

OS=$(shell uname -s)
ifeq ($(OS),Linux)
	LIBFILE=libann.so
	CCo=g++ -c $(CFLAGS)
	CC=g++ $(CFLAGS)
else
	LIBFILE=libann.dylib
	export LDFLAGS="-L/usr/local/opt/libomp/lib"
  export CPPFLAGS="-I/usr/local/opt/libomp/include"
	CCo=g++ -Xpreprocessor -fopenmp -c $(CFLAGS) $(CPPFLAGS)
	CC=g++ -Xpreprocessor -fopenmp $(CFLAGS) $(CPPFLAGS)
endif

all: obj lib

obj:
	$(CCo) activationfunctions.cc -o $(BUILDDIR)/activationfunctions.o
	$(CCo) backpropagate.cc -o $(BUILDDIR)/backpropagate.o
	$(CCo) boxcox.cc -o $(BUILDDIR)/boxcox.o
	$(CCo) cliplog.cc -o $(BUILDDIR)/cliplog.o
	$(CCo) costfunction.cc -o $(BUILDDIR)/costfunction.o
	$(CCo) matrix.cc -o $(BUILDDIR)/matrix.o
	$(CCo) matrixarray.cc -o $(BUILDDIR)/matrixarray.o
	$(CCo) matrixmath.cc -o $(BUILDDIR)/matrixmath.o
	$(CCo) networkfunc.cc -o $(BUILDDIR)/networkfunc.o
	$(CCo) randomnumber.cc -o $(BUILDDIR)/randomnumber.o
	$(CCo) readarray.cc -o $(BUILDDIR)/readarray.o
	$(CCo) regularization.cc -o $(BUILDDIR)/regularization.o
	$(CCo) softmax.cc -o $(BUILDDIR)/softmax.o
lib:
	$(CC) $(BUILDDIR)/*.o -shared -o ../lib/$(LIBFILE) $(LDFLAGS)

winobj:
	$(CCWo) activationfunctions.cc -o $(BUILDDIR)/activationfunctions.o
	$(CCWo) backpropagate.cc -o $(BUILDDIR)/backpropagate.o
	$(CCWo) boxcox.cc -o $(BUILDDIR)/boxcox.o
	$(CCWo) cliplog.cc -o $(BUILDDIR)/cliplog.o
	$(CCWo) costfunction.cc -o $(BUILDDIR)/costfunction.o
	$(CCWo) matrix.cc -o $(BUILDDIR)/matrix.o
	$(CCWo) matrixarray.cc -o $(BUILDDIR)/matrixarray.o
	$(CCWo) matrixmath.cc -o $(BUILDDIR)/matrixmath.o
	$(CCWo) networkfunc.cc -o $(BUILDDIR)/networkfunc.o
	$(CCWo) randomnumber.cc -o $(BUILDDIR)/randomnumber.o
	$(CCWo) readarray.cc -o $(BUILDDIR)/readarray.o
	$(CCWo) regularization.cc -o $(BUILDDIR)/regularization.o
	$(CCWo) softmax.cc -o $(BUILDDIR)/softmax.o

winlib:
	$(CCW) $(BUILDDIR)/*.o -shared -o ../lib/libann.dll
