#include "lbm2d_kernels.cuh"

// Select CUDA device
__host__ void selectDevice(int device) {

	// List all devices
	int deviceCount;
	HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

	// Print all devices
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("Device %i: %s", i, deviceProp.name);
	}

	// Set the device
	HANDLE_ERROR(cudaSetDevice(device));

	// Print device info
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	printf("Selected device: %s\n", deviceProp.name);
	printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("Total global memory: %lu\n", deviceProp.totalGlobalMem);
	printf("Shared memory per block: %lu\n", deviceProp.sharedMemPerBlock);
	printf("Registers per block: %d\n", deviceProp.regsPerBlock);
	printf("Warp size: %d\n", deviceProp.warpSize);
	printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("Max threads dimensions: %d, %d, %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("Max grid size: %d, %d, %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("Clock rate: %d\n", deviceProp.clockRate);
	printf("Total constant memory: %lu\n", deviceProp.totalConstMem);
	printf("Device Overlap: %s\n", deviceProp.deviceOverlap ? "Supported" : "Not available");
	printf("Integrated: %s\n", deviceProp.integrated ? "Yes" : "No");
}

__host__ void add_random_population(
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int threads_per_block,
	int blocks_per_grid
) {
	// Check FloatingPointAccuracy
	if (floating_point_accuracy != FloatingPointAccuracy::fp32) {
		fprintf(stderr, "Error: other floating point systems than fp32 aren't supported.\n");
		exit(EXIT_FAILURE);
	}

	// Check Velocity Set
	int volume_dimentionality;
	int velocity_count;
	if (velocity_set == D2Q9) {
		volume_dimentionality	= 2;
		velocity_count			= 9;
	}
	else if (velocity_set == D3Q15) {
		volume_dimentionality	= 3;
		velocity_count			= 15;
	}
	else if (velocity_set == D3Q19) {
		volume_dimentionality	= 3;
		velocity_count			= 19;
	}
	else if (velocity_set == D3Q27) {
		volume_dimentionality	= 3;
		velocity_count			= 27;
	}
	else {
		fprintf(stderr, "velocity set is not supported.\n");
		exit(EXIT_FAILURE);
	}

	// Call the kernel
	add_random_population_kernel<<<blocks_per_grid, threads_per_block>>> (
		floating_point_accuracy,
		volume_dimentionality,
		velocity_count
		);



}

__global__ void add_random_population_kernel(
	FloatingPointAccuracy floating_point_accuracy,
	int volume_dimentionality,
	int velocity_count
) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;


}