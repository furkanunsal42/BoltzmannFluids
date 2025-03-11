#pragma once

#include <stdint.h>
#include <string>

enum VelocitySet {
	D2Q9 = 0,
	D3Q15 = 1,
	D3Q19 = 2,
	D3Q27 = 3,
};

int32_t get_VelocitySet_dimention(VelocitySet velocity_set);
int32_t get_VelocitySet_velocity_count(VelocitySet velocity_set);
std::string get_VelocitySet_to_macro(VelocitySet velocity_set);