#include <gtest/gtest.h>
#include <omp.h>

int main(int argc, char **argv) {
	omp_set_num_threads(1);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
