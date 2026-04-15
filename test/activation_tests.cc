#include <gtest/gtest.h>
#include <ann.h>

TEST(ActivationFunctions, BasicValues) {
	EXPECT_DOUBLE_EQ(AF_ReLU(-2.5), 0.0);
	EXPECT_DOUBLE_EQ(AF_ReLU(2.5), 2.5);
	EXPECT_DOUBLE_EQ(AF_LeakyReLU(-2.0), -0.02);
	EXPECT_DOUBLE_EQ(AF_LeakyReLUGradient(-2.0), 0.01);
	EXPECT_NEAR(AF_Sigmoid(0.0), 0.5, 1.0e-12);
	EXPECT_NEAR(AF_SigmoidGradient(0.0), 0.25, 1.0e-12);
	EXPECT_NEAR(AF_Tanh(0.0), 0.0, 1.0e-12);
	EXPECT_DOUBLE_EQ(AF_Softplus(60.0), 60.0);
	EXPECT_DOUBLE_EQ(AF_Linear(3.0), 3.0);
}

TEST(ActivationFunctions, InverseRoundTrip) {
	const double xSig = 0.23;
	const double xTanh = -0.35;

	EXPECT_NEAR(AF_InverseSigmoid(AF_Sigmoid(xSig)), xSig, 1.0e-12);
	EXPECT_NEAR(AF_InverseTanh(AF_Tanh(xTanh)), xTanh/2.0, 1.0e-12);
}

TEST(ActivationFunctions, BoundaryAndLookupCases) {
	EXPECT_DOUBLE_EQ(AF_ReLUGradient(0.0), 0.0);
	EXPECT_DOUBLE_EQ(AF_LeakyReLUGradient(0.0), 0.01);
	EXPECT_DOUBLE_EQ(AF_SoftplusGradient(60.0), 1.0);

	EXPECT_EQ(AFFromString("relu"), AF_ReLU);
	EXPECT_EQ(AFFromString("linear"), AF_Linear);
	EXPECT_EQ(AFFromString("unknown_function"), AF_Sigmoid);
}
