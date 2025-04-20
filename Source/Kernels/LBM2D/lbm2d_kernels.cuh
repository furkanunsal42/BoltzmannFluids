#pragma once

//#include <cuda.h>
//#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "LBMConstants/FloatingPointAccuracy.h"
#include "LBMConstants/VelocitySet.h"
#include "Kernels/handle_error.h"

__host__ void selectDevice(int device);

__host__ void add_random_population(
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int threads_per_block,
	int blocks_per_grid
);

__global__ void add_random_population_kernel(
	FloatingPointAccuracy floating_point_accuracy,
	int volume_dimentionality,
	int velocity_count
);