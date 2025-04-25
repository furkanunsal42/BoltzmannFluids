#include "LBMConstants/LBMParams.cuh"

__host__ void lbm_set_params(
	LBMParams& lbm_params,
	FloatingPointAccuracy floating_point_accuracy,
	VelocitySet velocity_set,
	int2 lattice_resolution,
	float cs,
	float tau
) {
	// Check FloatingPointAccuracy
	if (floating_point_accuracy == FloatingPointAccuracy::fp32) {
		lbm_params.floating_point_accuracy = FloatingPointAccuracy::fp32;
	}
	else {
		HANDLE_ERROR("Error: other floating point systems than fp32 aren't supported.\n");
	}

	// Check Velocity Set
	if (velocity_set == D2Q9) {
		lbm_params.volume_dimensionality = 2;
		lbm_params.velocity_count = 9;
	}
	else if (velocity_set == D3Q15) {
		lbm_params.volume_dimensionality = 3;
		lbm_params.velocity_count = 15;
	}
	else if (velocity_set == D3Q19) {
		lbm_params.volume_dimensionality = 3;
		lbm_params.velocity_count = 19;
	}
	else if (velocity_set == D3Q27) {
		lbm_params.volume_dimensionality = 3;
		lbm_params.velocity_count = 27;
	}
	else {
		HANDLE_ERROR("Velocity set is not supported.\n");
	}

	lbm_params.lattice_resolution = lattice_resolution;
	lbm_params.cs = cs;
	lbm_params.tau = tau;
}
