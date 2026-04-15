# libann
[![Regression Tests](https://github.com/mattkjames7/libann/actions/workflows/regression-tests.yml/badge.svg)](https://github.com/mattkjames7/libann/actions/workflows/regression-tests.yml)

Simple C++ neural network code used by some other libraries.

## Install

In Linux and MacOS:
```bash
make
sudo make install
```

In Windows (install TDM-GCC first):
```cmd
compile.bat
```

## Usage

This code can be included in other C++ projects:
```cpp
#include <ann.h>
```
and compile with the ```-lann -lfopenmp -lm``` flags.

The main purpose of this library is to run neural networks which are already trained. It can open trained neural networks created by [NNFunction](https://github.com/mattkjames7/NNFunction) (see below for file format description). It may even be able to train new ones, if not now, then when the code is finished...

Currently, the `NetworkFunc` class is an untrainable object. It can either read in the pre-trained weights from a file, or an embedded memory location, e.g.

```cpp
/* in the case of a file */
std::string fileName = "path/to/file.bin";
std::string hiddenFunc = "softplus";
std::string outFunc = "linear";
std::string costFunc = "mean_squared";
NetworkFunc *ann = new NetworkFunc(	fileName.c_str(),
									hiddenFunc.c_str(),
									outFunc.c_str(),
									costFunc.c_str());

/* or memory location */
unsigned char *ptr = _binary_mem_location_bin;
NetworkFunc *ann = new NetworkFunc(	ptr,
									hiddenFunc.c_str(),
									outFunc.c_str(),
									costFunc.c_str());
```

The `NetworkFunc` object can then be used to make predictions when given input (`X`) and output (`y`) matrices, e.g.:
```cpp
ann->Predict(n,X,y);
```
where `X` and `y` are 2D arrays (`float**`) of shape `(n,m)` and `(n,k)`, respectively. In this case, `n` is the number of samples, `m` is the number of features (input nodes) and `k` is the number of output nodes.

## Regression Tests

This repo now includes a GoogleTest-based baseline suite for regression checks before and after upgrades.

Run from the repo root:

```bash
make
make test
```

`make test` builds and runs `test/ann_tests` with deterministic single-thread execution (`OMP_NUM_THREADS=1`).

Current test modules in `test/`:

- `matrix_ops_tests.cc`
- `activation_tests.cc`
- `loss_regularization_tests.cc`
- `networkfunc_integration_tests.cc`

The suite intentionally locks current behavior, including legacy numerical quirks, so behavior changes during upgrades are visible in test output.