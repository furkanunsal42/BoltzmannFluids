#pragma once

#include "LBMConstants/LBMParams.cuh"

//#include <cuda.h>
//#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Global constants for simulation
extern __constant__ LBMParams d_lbm_params;	// cudaMemcpyToSymbol(d_lbm_params, &your_params, sizeof(LBM_Constants));

// Stream
__global__ void _cuda_stream_kernel(
	const float* lattice_velocity_set,
	const float* lattice_source,
	float* lattice_target
);

__host__ void cuda_stream(
	const float* lattice_velocity_set,
	const float* lattice_source,
	float* lattice_target
);

// Collide
__global__ void _cuda_collide_kernel(
	const float* lattice_velocity_set,
	const float* lattice_source,
	float* lattice_target,
	int* boundries
);

__host__ void cuda_collide(
	const float* d_lattice_velocity_set,
	const float* d_lattice_source,
	float* d_lattice_target,
	int* boundries
);


// Helper functions
__device__ inline int _calculate_lattice_index(int2 pixel_coord, int velocity_index);

__device__ inline float _get_lattice_source(int2 pixel_coord, int velocity_index, const float* lattice_source);

__device__ inline void _set_lattice_source(int2 pixel_coord, int velocity_index, float* lattice_source, float value);

__device__ float _compute_density(int2 pixel_coord, const float* lattice_source);

__device__ float3 _compute_velocity(int2 pixel_coord, float density, const float* lattice_source, const float* lattice_velocity_set, int* boundries);

__device__ float3 _get_lattice_velocity(int velocity_index, const float* lattice_velocity_set);

__device__ bool _get_boundry(int2 pixel_coord, const int* boundries);

__device__ float _get_lattice_weight(int velocity_index, const float* lattice_velocity_set);

__device__ float _compute_equilibrium(float density, float3 velocity, int population_id, const float* lattice_velocity_set);

__device__ float _apply_bgk_collision(float current_velocity, float equilibrium);