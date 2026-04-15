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

	const ActFunc relu = AFFromString("relu");
	const ActFunc linear = AFFromString("linear");
	const ActFunc fallback = AFFromString("unknown_function");

	ASSERT_NE(relu, nullptr);
	ASSERT_NE(linear, nullptr);
	ASSERT_NE(fallback, nullptr);

	// Compare behavior instead of raw function-pointer identity across modules.
	EXPECT_DOUBLE_EQ(relu(-2.0), AF_ReLU(-2.0));
	EXPECT_DOUBLE_EQ(relu(3.5), AF_ReLU(3.5));
	EXPECT_DOUBLE_EQ(linear(-2.0), AF_Linear(-2.0));
	EXPECT_DOUBLE_EQ(linear(3.5), AF_Linear(3.5));
	EXPECT_DOUBLE_EQ(fallback(-2.0), AF_Sigmoid(-2.0));
	EXPECT_DOUBLE_EQ(fallback(3.5), AF_Sigmoid(3.5));
}
