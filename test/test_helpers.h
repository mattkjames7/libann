#ifndef LIBANN_TEST_HELPERS_H
#define LIBANN_TEST_HELPERS_H

#include <ann.h>

inline void FillMatrixFromArray(ann::Matrix &m, const double *vals, int nRows, int nCols) {
	int i;
	int j;
	for (i = 0; i < nRows; i++) {
		for (j = 0; j < nCols; j++) {
			m.data[i][j] = vals[i*nCols + j];
		}
	}
}

#endif
