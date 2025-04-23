#pragma once

#include "LBMConstants/FloatingPointAccuracy.h"
#include "LBMConstants/VelocitySet.h"
#include "Kernels/handle_error.h"

//#include <cuda.h>
//#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Stream
__global__ void _cuda_stream_kernel(
	int velocity_count,
	int2 d_lattice_resolution,
	const float* lattice_velocity_set,
	const float* lattice_source,
	float* lattice_target
);

__host__ void _cuda_stream(
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int2 lattice_resolution,
	const float* d_lattice_velocity_set,
	const float* d_lattice_source,
	float* d_lattice_target
);