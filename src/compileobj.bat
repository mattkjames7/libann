mkdir ..\build
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp activationfunctions.cc -o ..\build\activationfunctions.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp backpropagate.cc -o ..\build\backpropagate.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp boxcox.cc -o ..\build\boxcox.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp cliplog.cc -o ..\build\cliplog.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp costfunction.cc -o ..\build\costfunction.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp matrix.cc -o ..\build\matrix.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp matrixarray.cc -o ..\build\matrixarray.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp matrixmath.cc -o ..\build\matrixmath.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp networkfunc.cc -o ..\build\networkfunc.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp randomnumber.cc -o ..\build\randomnumber.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp readarray.cc -o ..\build\readarray.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp regularization.cc -o ..\build\regularization.o
g++ -c -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp softmax.cc -o ..\build\softmax.o	