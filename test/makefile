# include flags
IFLAGS=-I../include

# Compiler
CCWo=x86_64-w64-mingw32-g++-win32 -c $(CFLAGS) $(IFLAGS)
CCW=x86_64-w64-mingw32-g++-win32 $(CFLAGS) $(IFLAGS)

OS=$(shell uname -s)
ifeq ($(OS),Linux)
	LDFLAGS=-Wl,-rpath='$$ORIGIN/../lib' -L ../lib -lann
	CFLAGS=-lm -std=c++17 -Wextra -O3 -fopenmp
	CC=g++ -fPIC $(CFLAGS) $(IFLAGS)
else
	LDFLAGS=-L ../lib -lann -L/usr/local/opt/libomp/lib -lomp
	CFLAGS=-lm -std=c++17 -Wextra -O3 "-I/usr/local/opt/libomp/include"
	CC=g++ -Xpreprocessor -fopenmp $(CFLAGS) $(IFLAGS)
endif

GTEST_CFLAGS=$(shell pkg-config --cflags gtest 2>/dev/null)
GTEST_LIBS=$(shell pkg-config --libs gtest gtest_main 2>/dev/null)
ifeq ($(strip $(GTEST_LIBS)),)
	GTEST_LIBS=-lgtest_main -lgtest -pthread
endif

TEST_BIN=ann_tests
TEST_SRCS=$(wildcard *.cc)

all: $(TEST_BIN)
	OMP_NUM_THREADS=1 ./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRCS)
	$(CC) $(TEST_SRCS) -o $(TEST_BIN) $(LDFLAGS) $(GTEST_CFLAGS) $(GTEST_LIBS)

clean:
	-rm -v $(TEST_BIN)
