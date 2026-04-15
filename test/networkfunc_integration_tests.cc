#include <gtest/gtest.h>
#include <ann.h>

TEST(NetworkFuncIntegration, DeterministicForwardPass) {
	int s[] = {2, 2, 1};
	NetworkFunc ann(3, s, "linear", "linear", "mean_squared");

	ann.W_->matrix[0]->data[0][0] = 1.0;
	ann.W_->matrix[0]->data[0][1] = 0.0;
	ann.W_->matrix[0]->data[1][0] = 0.0;
	ann.W_->matrix[0]->data[1][1] = 1.0;
	ann.B_->matrix[0]->data[0][0] = 0.5;
	ann.B_->matrix[0]->data[0][1] = -0.5;

	ann.W_->matrix[1]->data[0][0] = 2.0;
	ann.W_->matrix[1]->data[1][0] = 3.0;
	ann.B_->matrix[1]->data[0][0] = 0.25;

	float inRow[2] = {1.0f, 2.0f};
	float *in[1] = {inRow};

	float outRow[1] = {0.0f};
	float *out[1] = {outRow};

	ann.Predict(1, in, out);

	EXPECT_NEAR(out[0][0], 7.75, 1.0e-6);
}
