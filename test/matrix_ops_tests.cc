#include <gtest/gtest.h>
#include <ann.h>

#include "test_helpers.h"

using namespace ann;

TEST(MatrixOps, MatrixDotLegacyFixture) {
	Matrix ma(2,3);
	Matrix mb(3,4);
	Matrix mc(2,4);

	const double aVals[] = {
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0
	};
	const double bVals[] = {
		0.1, 0.1, 0.4, 0.2,
		0.9, 0.7, 0.1, 0.2,
		0.3, 0.5, 0.6, 0.7
	};

	FillMatrixFromArray(ma, aVals, 2, 3);
	FillMatrixFromArray(mb, bVals, 3, 4);

	MatrixDot(ma, mb, false, false, mc);

	EXPECT_NEAR(mc.data[0][0], 2.8, 1.0e-12);
	EXPECT_NEAR(mc.data[0][1], 3.0, 1.0e-12);
	EXPECT_NEAR(mc.data[0][2], 2.4, 1.0e-12);
	EXPECT_NEAR(mc.data[0][3], 2.7, 1.0e-12);
	EXPECT_NEAR(mc.data[1][0], 6.7, 1.0e-12);
	EXPECT_NEAR(mc.data[1][1], 6.9, 1.0e-12);
	EXPECT_NEAR(mc.data[1][2], 5.7, 1.0e-12);
	EXPECT_NEAR(mc.data[1][3], 6.0, 1.0e-12);
}

TEST(MatrixOps, MatrixAddAndSubtract) {
	Matrix a(2,2);
	Matrix b(2,2);
	Matrix sum(2,2);
	Matrix diff(2,2);

	const double aVals[] = {1.0, 2.0, 3.0, 4.0};
	const double bVals[] = {0.5, 1.0, 1.5, 2.0};
	FillMatrixFromArray(a, aVals, 2, 2);
	FillMatrixFromArray(b, bVals, 2, 2);

	MatrixAdd(a, b, false, false, sum);
	MatrixSubtract(a, b, false, false, diff);

	EXPECT_NEAR(sum.data[0][0], 1.5, 1.0e-12);
	EXPECT_NEAR(sum.data[1][1], 6.0, 1.0e-12);
	EXPECT_NEAR(diff.data[0][0], 0.5, 1.0e-12);
	EXPECT_NEAR(diff.data[1][1], 2.0, 1.0e-12);
}

TEST(MatrixOps, MatrixMultiplyElementWise) {
	Matrix a(2,2);
	Matrix b(2,2);
	Matrix out(2,2);

	a.data[0][0] = 1.0;
	a.data[0][1] = 2.0;
	a.data[1][0] = -3.0;
	a.data[1][1] = 4.0;

	b.data[0][0] = 0.5;
	b.data[0][1] = 1.5;
	b.data[1][0] = -2.0;
	b.data[1][1] = 0.25;

	MatrixMultiply(a, b, false, false, out);

	EXPECT_NEAR(out.data[0][0], 0.5, 1.0e-12);
	EXPECT_NEAR(out.data[0][1], 3.0, 1.0e-12);
	EXPECT_NEAR(out.data[1][0], 6.0, 1.0e-12);
	EXPECT_NEAR(out.data[1][1], 1.0, 1.0e-12);
}

TEST(MatrixOps, ApplyFunctionToMatrixInPlaceSigmoid) {
	Matrix m(2,2);
	m.data[0][0] = 0.0;
	m.data[0][1] = 10.0;
	m.data[1][0] = -10.0;
	m.data[1][1] = 0.5;

	ApplyFunctionToMatrix(m, AF_Sigmoid);

	EXPECT_NEAR(m.data[0][0], 0.5, 1.0e-12);
	EXPECT_GT(m.data[0][1], 0.9999);
	EXPECT_LT(m.data[1][0], 0.0001);
	EXPECT_NEAR(m.data[1][1], AF_Sigmoid(0.5), 1.0e-12);
}
