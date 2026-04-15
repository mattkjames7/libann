#include <gtest/gtest.h>
#include <ann.h>

#include <cmath>

#include "test_helpers.h"

TEST(SoftmaxAndCliplog, ProbabilityAndFiniteChecks) {
	Matrix z(2,3);
	Matrix sm(2,3);

	const double zVals[] = {
		1.0, 2.0, 3.0,
		0.0, 0.0, 0.0
	};
	FillMatrixFromArray(z, zVals, 2, 3);

	softmax(z, sm);

	const double row0 = sm.data[0][0] + sm.data[0][1] + sm.data[0][2];
	const double row1 = sm.data[1][0] + sm.data[1][1] + sm.data[1][2];
	EXPECT_NEAR(row0, 1.0, 1.0e-6);
	EXPECT_NEAR(row1, 1.0, 1.0e-6);

	EXPECT_NEAR(sm.data[1][0], 1.0/3.0, 1.0e-6);
	EXPECT_NEAR(sm.data[1][1], 1.0/3.0, 1.0e-6);
	EXPECT_NEAR(sm.data[1][2], 1.0/3.0, 1.0e-6);

	EXPECT_TRUE(std::isfinite(cliplog(1.0e-50, 1.0e-40)));
	EXPECT_NEAR(cliplog(1.0e-50, 1.0e-40), std::log(1.0e-40), 1.0e-12);
}

TEST(Regularization, L1AndL2Values) {
	int shape[] = {1, 4};
	MatrixArray w(1, shape);

	w.matrix[0]->data[0][0] = 2.0;
	w.matrix[0]->data[0][1] = -1.0;
	w.matrix[0]->data[0][2] = 0.5;
	w.matrix[0]->data[0][3] = -1.5;

	const float l1 = L1Regularization(w, 0.01f, 10);
	const float l2 = L2Regularization(w, 0.02f, 10);

	EXPECT_NEAR(l1, 0.0025, 1.0e-8);
	EXPECT_NEAR(l2, 0.0075, 1.0e-8);
}

TEST(CostFunctions, CurrentCrossEntropyBehavior) {
	Matrix h(2,2);
	Matrix yA(2,2);
	Matrix yB(2,2);
	int shape[] = {1, 1};
	MatrixArray w(1, shape);

	const double hVals[] = {
		0.5, 0.25,
		0.8, 0.2
	};
	FillMatrixFromArray(h, hVals, 2, 2);

	const double yAVals[] = {
		1.0, 0.0,
		0.0, 1.0
	};
	const double yBVals[] = {
		0.0, 1.0,
		1.0, 0.0
	};
	FillMatrixFromArray(yA, yAVals, 2, 2);
	FillMatrixFromArray(yB, yBVals, 2, 2);

	const double jA = crossEntropyCost(h, yA, w, 0.0, 0.0);
	const double jB = crossEntropyCost(h, yB, w, 0.0, 0.0);

	const double expected =
		-(std::log(0.5) + std::log(0.8) + std::log(0.25) + std::log(0.2)) / 2.0;

	EXPECT_NEAR(jA, expected, 1.0e-12);
	EXPECT_NEAR(jB, expected, 1.0e-12);
	EXPECT_NEAR(jA, jB, 1.0e-12);
}

TEST(CostFunctions, MeanSquaredL2PathMatchesCurrentCode) {
	Matrix h(1,2);
	Matrix y(1,2);
	int shape[] = {1, 2};
	MatrixArray w(1, shape);

	h.data[0][0] = 1.0;
	h.data[0][1] = -1.0;
	y.data[0][0] = 1.0;
	y.data[0][1] = -1.0;

	w.matrix[0]->data[0][0] = 2.0;
	w.matrix[0]->data[0][1] = -1.0;

	const double j = meanSquaredCost(h, y, w, 0.0, 0.5);

	/* Current implementation uses L1Regularization in the L2 branch. */
	const double expected = 0.5 * (std::fabs(2.0) + std::fabs(-1.0)) / (1.0 * 2.0);
	EXPECT_NEAR(j, expected, 1.0e-12);
}

TEST(CostFunctions, DeltaFunctionsMatchCurrentImplementation) {
	Matrix h(2,2);
	Matrix y(2,2);
	Matrix ceDelta(2,2);
	Matrix msDelta(2,2);

	h.data[0][0] = 0.9;
	h.data[0][1] = 0.1;
	h.data[1][0] = 0.3;
	h.data[1][1] = 0.7;

	y.data[0][0] = 1.0;
	y.data[0][1] = 0.0;
	y.data[1][0] = 0.0;
	y.data[1][1] = 1.0;

	crossEntropyDelta(h, y, AF_LinearGradient, ceDelta);
	meanSquaredDelta(h, y, AF_LinearGradient, msDelta);

	EXPECT_NEAR(ceDelta.data[0][0], -0.1, 1.0e-12);
	EXPECT_NEAR(ceDelta.data[0][1], 0.1, 1.0e-12);
	EXPECT_NEAR(ceDelta.data[1][0], 0.3, 1.0e-12);
	EXPECT_NEAR(ceDelta.data[1][1], -0.3, 1.0e-12);

	/* Current implementation applies InvAFGrad to each delta element. */
	EXPECT_NEAR(msDelta.data[0][0], 1.0, 1.0e-12);
	EXPECT_NEAR(msDelta.data[0][1], 1.0, 1.0e-12);
	EXPECT_NEAR(msDelta.data[1][0], 1.0, 1.0e-12);
	EXPECT_NEAR(msDelta.data[1][1], 1.0, 1.0e-12);
}
