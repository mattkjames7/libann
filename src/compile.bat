call compileobj.bat

mkdir ..\lib
g++ -lm -fPIC -std=c++17 -Wextra -O3 -fopenmp ..\build\*.o -shared -o ..\lib\libann.dll
