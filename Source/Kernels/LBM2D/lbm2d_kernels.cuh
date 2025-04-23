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

__host__ void cuda_stream(
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int2 lattice_resolution,
	const float* d_lattice_velocity_set,
	const float* d_lattice_source,
	float* d_lattice_target
);

// Collide
__global__ void _cuda_collide_kernel(
	int2 lattice_resolution,
	int velocity_count,
	const float* lattice_velocity_set,
	const float* lattice_source,
	float* lattice_target,
	float lattice_speed_of_sound,
	float relaxation_time
);

__host__ void cuda_collide(
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int2 lattice_resolution,
	const float* d_lattice_velocity_set,
	const float* d_lattice_source,
	float* d_lattice_target,
	float lattice_speed_of_sound,
	float relaxation_time
);


// Helper function
__host__ void _get_and_validate_info(
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int& volume_dimensionality,
	int& velocity_count
);

__device__ inline int _calculate_lattice_index(
	int2 pixel_coord,
	int velocity_index,
	int velocity_count,
	int2 lattice_resolution
);

__device__ inline float _get_lattice_source(
	int2 pixel_coord,
	int velocity_index,
	int velocity_count,
	int2 lattice_resolution,
	const float* lattice_source
);

__device__ inline void _set_lattice_source(
	int2 pixel_coord,
	int velocity_index,
	int velocity_count,
	int2 lattice_resolution,
	float* lattice_source,
	float value
);