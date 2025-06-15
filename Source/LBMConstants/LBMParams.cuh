#pragma once

#include "LBMConstants/FloatingPointAccuracy.h"
#include "LBMConstants/VelocitySet.h"
#include "Kernels/handle_error.h"

#include <cuda_runtime.h>

// My aim with this struct is to reduce the burden of passing extra parameters each time we launch a kernel. 
// Instead, use read-only constant memory. We can do this in such a host code:
// 
// LBMParams h_params;
// lbm_set_params(
//	&h_params,
//	floating_point_accuracy,
//	velocity_set,
//	lattice_resolution,
//	cs,
//	tau
// );
// 
// cudaMemcpyToSymbol(d_params, &h_params, sizeof(h_params));

struct LBMParams {
    FloatingPointAccuracy floating_point_accuracy = FloatingPointAccuracy::fp32;
    int2 lattice_resolution = make_int2(512, 512);// Default resolution
    int volume_dimensionality = 2;                // 2D, 3D
    int velocity_count = 9;                       // 9, 15, 19, 27
    float cs = 1.0f;                              // Speed of sound
    float tau = 0.53f;                            // Relaxation time
};

__host__ void lbm_set_params(
    LBMParams& lbm_params,
    FloatingPointAccuracy floating_point_accuracy,
    VelocitySet velocity_set,
    int2 lattice_resolution,
    float cs,
    float tau
);