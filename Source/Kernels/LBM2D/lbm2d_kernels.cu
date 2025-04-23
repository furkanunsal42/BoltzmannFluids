#include "lbm2d_kernels.cuh"


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
	int velocity_id = id % velocity_count;

	int2 pixel_coord = make_int2(
		pixel_id % d_lattice_resolution.x, 
		pixel_id / d_lattice_resolution.y
	);

	int2 velocity_offset = make_int2(
		(int)lattice_velocity_set[velocity_id * 4 + 0],
		(int)lattice_velocity_set[velocity_id * 4 + 1]
	);

	int2 source_pixel_coord = make_int2(
		(pixel_coord.x - velocity_offset.x + d_lattice_resolution.x) % d_lattice_resolution.x,
		(pixel_coord.y - velocity_offset.y + d_lattice_resolution.y) % d_lattice_resolution.y
	);

	int source_pixel_id = source_pixel_coord.x + source_pixel_coord.y * d_lattice_resolution.x;
	lattice_target[id] = lattice_source[source_pixel_id * velocity_count + velocity_id];
}


__host__ void _cuda_stream(
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int2 lattice_resolution,
	const float* d_lattice_velocity_set,
	const float* d_lattice_source,
	float* d_lattice_target
) {
	// Check FloatingPointAccuracy
	if (floating_point_accuracy != FloatingPointAccuracy::fp32) {
		HANDLE_ERROR("Error: other floating point systems than fp32 aren't supported.\n");
	}

	int volume_dimentionality;
	int velocity_count;
	// Check Velocity Set
	if (velocity_set == D2Q9) {
		 volume_dimentionality = 2;
		velocity_count = 9;
	}
	else if (velocity_set == D3Q15) {
		volume_dimentionality = 3;
		velocity_count = 15;
	}
	else if (velocity_set == D3Q19) {
		volume_dimentionality = 3;
		velocity_count = 19;
	}
	else if (velocity_set == D3Q27) {
		volume_dimentionality = 3;
		velocity_count = 27;
	}
	else {
		HANDLE_ERROR("Velocity set is not supported.\n");
	}

	int2 d_lattice_resolution = make_int2(lattice_resolution.x, lattice_resolution.y);
	int total_threads = d_lattice_resolution.x * d_lattice_resolution.y * velocity_count;

	const int threads_per_block = 64;
	int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

	// Call the kernel
	_cuda_stream_kernel<<<blocks_per_grid, threads_per_block>>>(
		velocity_count,
		d_lattice_resolution,
		d_lattice_velocity_set,
		d_lattice_source,
		d_lattice_target
		);
	cudaDeviceSynchronize(); // May be not needed
}
