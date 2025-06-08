#pragma once

#include <stdint.h>
#include <string>
#include <vector>

#include "vec4.hpp"

enum SimplifiedVelocitySet {
	D2Q5 = 0,
	D3Q7 = 1,
};

enum VelocitySet {
	D2Q9 = 2,
	D3Q15 = 3,
	D3Q19 = 4,
	D3Q27 = 5,
};

int32_t get_VelocitySet_dimention(VelocitySet velocity_set);
int32_t get_VelocitySet_vector_count(VelocitySet velocity_set);
std::string get_VelocitySet_to_macro(VelocitySet velocity_set);

std::vector<glm::vec4> get_velosity_vectors(VelocitySet velocity_set);

int32_t get_SimplifiedVelocitySet_dimention(SimplifiedVelocitySet velocity_set);
int32_t get_SimplifiedVelocitySet_vector_count(SimplifiedVelocitySet velocity_set);
std::string get_SimplifiedVelocitySet_to_macro(SimplifiedVelocitySet velocity_set);

std::vector<glm::vec4> get_velosity_vectors(SimplifiedVelocitySet velocity_set);