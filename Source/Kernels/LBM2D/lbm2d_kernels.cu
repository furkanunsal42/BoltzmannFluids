#include "lbm2d_kernels.cuh"


// Stream 
__global__ void _cuda_stream_kernel(
	const float* lattice_velocity_set,
	const float* lattice_source,
	float* lattice_target
) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= d_lbm_params.lattice_resolution.x * d_lbm_params.lattice_resolution.y * d_lbm_params.velocity_count) {
		return;
	}

	int pixel_id = id / d_lbm_params.velocity_count;
	int velocity_index = id % d_lbm_params.velocity_count;

	int2 pixel_coord = make_int2(
		pixel_id % d_lbm_params.lattice_resolution.x,
		pixel_id / d_lbm_params.lattice_resolution.x
	);

	int2 velocity_offset = make_int2(
		(int)lattice_velocity_set[velocity_index * 4 + 0],
		(int)lattice_velocity_set[velocity_index * 4 + 1]
	);

	int2 source_pixel_coord = make_int2(
		(pixel_coord.x - velocity_offset.x + d_lbm_params.lattice_resolution.x) % d_lbm_params.lattice_resolution.x,
		(pixel_coord.y - velocity_offset.y + d_lbm_params.lattice_resolution.y) % d_lbm_params.lattice_resolution.y
	);

	lattice_target[id] = _get_lattice_source(source_pixel_coord, velocity_index, lattice_source);
}

__host__ void cuda_stream(
	const float* lattice_velocity_set,
	const float* lattice_source,
	float* lattice_target
) {
	// Move below part to host's kernel call
	const int total_threads = d_lbm_params.lattice_resolution.x * d_lbm_params.lattice_resolution.y * d_lbm_params.velocity_count;
	dim3 threads_per_block(64);
	const dim3 blocks_per_grid((total_threads + threads_per_block.x - 1) / threads_per_block.x);

	// Call the kernel
	_cuda_stream_kernel<<<blocks_per_grid, threads_per_block>>>(
		lattice_velocity_set,
		lattice_source,
		lattice_target
		);
	cudaDeviceSynchronize(); // TODO: May not be needed. Check this
}

// Collide
__global__ void _cuda_collide_kernel(
	const float* lattice_velocity_set,
	const float* lattice_source,
	float* lattice_target,
	int* boundries
) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= d_lbm_params.lattice_resolution.x * d_lbm_params.lattice_resolution.y * d_lbm_params.velocity_count) {
		return;
	}

	int2 pixel_coord = make_int2(
		id % d_lbm_params.lattice_resolution.x,
		id / d_lbm_params.lattice_resolution.x
	);

	float density = _compute_density(pixel_coord, lattice_source);
	if (density <= 0) 
		return;
	
	float3 velocity = _compute_velocity(pixel_coord, density, lattice_source, lattice_velocity_set, boundries);

	for (int population_id = 0; population_id < d_lbm_params.velocity_count; population_id++) {
		float weight				= _get_lattice_weight(population_id, lattice_velocity_set);
		float equilibrium_velocity	= _compute_equilibrium(density, velocity, population_id, lattice_velocity_set);
		lattice_target[id * d_lbm_params.velocity_count + population_id] = _apply_bgk_collision(_get_lattice_source(pixel_coord, population_id, lattice_source), equilibrium_velocity);
	}
}

__host__ void cuda_collide(
	const float* lattice_velocity_set,
	const float* lattice_source,
	float* lattice_target,
	int* boundries
) {
	const int total_cells = d_lbm_params.lattice_resolution.x * d_lbm_params.lattice_resolution.y;
	dim3 threads_per_block(64);																							
	const dim3 blocks_per_grid((total_cells + threads_per_block.x - 1) / threads_per_block.x);
	
	// Call the kernel
	_cuda_collide_kernel<<<blocks_per_grid, threads_per_block>>>(
		lattice_velocity_set, 
		lattice_source, 
		lattice_target,
		boundries
	);
}


// Helper functions
__device__ inline int _calculate_lattice_index(	int2 pixel_coord, int velocity_index) {
	return (pixel_coord.y * d_lbm_params.lattice_resolution.x + pixel_coord.x) * d_lbm_params.velocity_count + velocity_index;
}

__device__ inline float _get_lattice_source(int2 pixel_coord, int velocity_index, const float* lattice_source) {
	return lattice_source[_calculate_lattice_index(pixel_coord, velocity_index)];
}

__device__ inline void _set_lattice_source(int2 pixel_coord, int velocity_index, float* lattice_source,	float value) {
	lattice_source[_calculate_lattice_index(pixel_coord, velocity_index)] = value;
}

__device__ float _compute_density(int2 pixel_coord, const float* lattice_source) {
	//if(get_boundry(pixel_coord))
	//	return 0;
	float density = 0;
	for (int i = 0; i < d_lbm_params.velocity_count; i++) {
		density += _get_lattice_source(pixel_coord, i, lattice_source);
	}
	return density;
}

__device__ float3 _compute_velocity(int2 pixel_coord, float density, const float* lattice_source, const float* lattice_velocity_set, int* boundries) {
	if (_get_boundry(pixel_coord, boundries)) {
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	float3 velocity = make_float3(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < d_lbm_params.velocity_count; i++) {
		float scalar = _get_lattice_source(pixel_coord, i, lattice_source);
		float3 lattice_velocity = _get_lattice_velocity(i, lattice_velocity_set);
		velocity.x += scalar * lattice_velocity.x;
		velocity.y += scalar * lattice_velocity.y;
		velocity.z += scalar * lattice_velocity.z;
	}
	return make_float3(
		velocity.x / density,
		velocity.y / density,
		velocity.z / density
	);
}
__device__ float3 _get_lattice_velocity(int velocity_index, const float* lattice_velocity_set) {
	return make_float3(
		lattice_velocity_set[velocity_index * 4 + 0],
		lattice_velocity_set[velocity_index * 4 + 1],
		lattice_velocity_set[velocity_index * 4 + 2]
	);
}
__device__ bool _get_boundry(int2 pixel_coord, const int* boundries) {
	int pixel_id = pixel_coord.y * d_lbm_params.lattice_resolution.x + pixel_coord.x;
	int byte_id = pixel_id / 32;
	int bit_id = pixel_id % 32;
	return (boundries[byte_id] & (1 << bit_id)) != 0;
}

__device__ float _get_lattice_weight(int velocity_index, const float* lattice_velocity_set) {
	return lattice_velocity_set[velocity_index * 4 + 3];
}

__device__ float _compute_equilibrium(float density, float3 velocity, int population_id, const float* lattice_velocity_set) {
	float w = _get_lattice_weight(population_id, lattice_velocity_set);
	float3 ci = _get_lattice_velocity(population_id, lattice_velocity_set);
	float cs = d_lbm_params.cs;

	float ci_dot_u = ci.x * velocity.x + ci.y * velocity.y + ci.z * velocity.z;
	float ci_dot_u2 = ci_dot_u * ci_dot_u;
	float u_dot_u = velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z;

	float equilibrium = w * density * (
		1 + 3 * ci_dot_u + 9 * ci_dot_u2 / 2 - 3 * u_dot_u / 2
		);
	return equilibrium;
}

__device__ float _apply_bgk_collision(float current_velocity, float equilibrium) {
	return current_velocity - (current_velocity - equilibrium) / d_lbm_params.tau;
}