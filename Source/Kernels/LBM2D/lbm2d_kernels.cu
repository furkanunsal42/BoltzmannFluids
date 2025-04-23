#include "lbm2d_kernels.cuh"

// Stream
__global__ void _cuda_stream_kernel(
	int velocity_count,
	int2 d_lattice_resolution,
	const float* lattice_velocity_set,
	const float* lattice_source,
	float* lattice_target
) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= d_lattice_resolution.x * d_lattice_resolution.y * velocity_count) {
		return;
	}

	int pixel_id = id / velocity_count;
	int velocity_index = id % velocity_count;

	int2 pixel_coord = make_int2(
		pixel_id % d_lattice_resolution.x, 
		pixel_id / d_lattice_resolution.x
	);

	int2 velocity_offset = make_int2(
		(int)lattice_velocity_set[velocity_index * 4 + 0],
		(int)lattice_velocity_set[velocity_index * 4 + 1]
	);

	int2 source_pixel_coord = make_int2(
		(pixel_coord.x - velocity_offset.x + d_lattice_resolution.x) % d_lattice_resolution.x,
		(pixel_coord.y - velocity_offset.y + d_lattice_resolution.y) % d_lattice_resolution.y
	);

	int source_pixel_index = calculate_lattice_index(source_pixel_coord, velocity_index, velocity_count, d_lattice_resolution);

	lattice_target[id] = lattice_source[source_pixel_index];
}

__host__ void cuda_stream(
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int2 lattice_resolution,
	const float* d_lattice_velocity_set,
	const float* d_lattice_source,
	float* d_lattice_target
) {
	int volume_dimentionality;
	int velocity_count;
	get_and_validate_info(floating_point_accuracy, velocity_set, volume_dimentionality, velocity_count);

	int2 d_lattice_resolution = make_int2(lattice_resolution.x, lattice_resolution.y);
	int total_threads = d_lattice_resolution.x * d_lattice_resolution.y * velocity_count;

	dim3 threads_per_block(64);																							// Move to
	dim3 blocks_per_grid((lattice_resolution.x * lattice_resolution.y + threads_per_block.x - 1) / threads_per_block.x);// host's kernel call
	// Call the kernel
	_cuda_stream_kernel<<<blocks_per_grid, threads_per_block>>>(
		velocity_count,
		d_lattice_resolution,
		d_lattice_velocity_set,
		d_lattice_source,
		d_lattice_target
		);
	cudaDeviceSynchronize(); // May not be needed
}

// Collide
__global__ void _cuda_collide_kernel(
	int2 lattice_resolution,
	int velocity_count,
	const float* lattice_velocity_set, // 4 floats per velocity: x, y, z, weight
	const float* lattice_source,
	float* lattice_target,
	float lattice_speed_of_sound,
	float relaxation_time
) {

// TODO

}

__host__ void cuda_collide(
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int2 lattice_resolution,
	const float* d_lattice_velocity_set,
	const float* d_lattice_source,
	float* d_lattice_target,
	float lattice_speed_of_sound,
	float relaxation_time
) {
	int volume_dimentionality;
	int velocity_count;
	get_and_validate_info(floating_point_accuracy, velocity_set, volume_dimentionality, velocity_count);

	dim3 threads_per_block(64);																							// Move to
	dim3 blocks_per_grid((lattice_resolution.x * lattice_resolution.y + threads_per_block.x - 1) / threads_per_block.x);// host's kernel call
	// Call the kernel
	_cuda_collide_kernel << <blocks_per_grid, threads_per_block >> > (
		make_int2(lattice_resolution.x, lattice_resolution.y),
		velocity_count,
		d_lattice_velocity_set,
		d_lattice_source,
		d_lattice_target,
		lattice_speed_of_sound,
		relaxation_time
		);
}


// Helper functions
__host__ void _get_and_validate_info(
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int& volume_dimensionality,
	int& velocity_count
) {
	// Check FloatingPointAccuracy
	if (floating_point_accuracy != FloatingPointAccuracy::fp32) {
		HANDLE_ERROR("Error: other floating point systems than fp32 aren't supported.\n");
	}

	// Check Velocity Set
	if (velocity_set == D2Q9) {
		volume_dimensionality = 2;
		velocity_count = 9;
	}
	else if (velocity_set == D3Q15) {
		volume_dimensionality = 3;
		velocity_count = 15;
	}
	else if (velocity_set == D3Q19) {
		volume_dimensionality = 3;
		velocity_count = 19;
	}
	else if (velocity_set == D3Q27) {
		volume_dimensionality = 3;
		velocity_count = 27;
	}
	else {
		HANDLE_ERROR("Velocity set is not supported.\n");
	}
}

__device__ inline int _calculate_lattice_index(
	int2 pixel_coord,
	int velocity_index,
	int velocity_count,
	int2 lattice_resolution
) {
	return (pixel_coord.y * lattice_resolution.x + pixel_coord.x) * velocity_count + velocity_index;
}


__device__ inline float _get_lattice_source(
	int2 pixel_coord,
	int velocity_index,
	int velocity_count,
	int2 lattice_resolution,
	const float* lattice_source
) {
	int index = calculate_lattice_index(pixel_coord, velocity_index, velocity_count, lattice_resolution);
	return lattice_source[index];
}


__device__ inline void _set_lattice_source(
	int2 pixel_coord,
	int velocity_index,
	int velocity_count,
	int2 lattice_resolution,
	float* lattice_source,
	float value
) {
	int index = calculate_lattice_index(pixel_coord, velocity_index, velocity_count, lattice_resolution);
	lattice_source[index] = value;
}