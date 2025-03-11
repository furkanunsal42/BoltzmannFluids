#pragma once

#include <memory>

#include "LBMConstants/VelocitySet.h"
#include "ComputeProgram.h"

class LBM2D {
public:
	
	void compile_shaders();
	
	void generate_lattice(glm::ivec2 resolution, glm::vec2 volume_dimentions_meters);

	void iterate_time(double time_milliseconds);
	double get_total_time_elapsed();

	void set_velocity_set(VelocitySet velocity_set);
	VelocitySet get_velocity_set();

	void copy_to_texture_velocity_index(Texture2D& target_texture, int32_t velocity_index);
	void copy_to_texture_velocity_total(Texture2D& target_texture);
	void copy_to_texture_velocity_magnetude(Texture2D& target_texture);
	void copy_to_texture_density(Texture2D& target_texture);

private:
	
	void _advect();
	void _collide();
	void _apply_boundry_conditions();

	double total_time_elapsed = 0;
	VelocitySet velocity_set = D2Q9;

	glm::ivec2 resolution;
	glm::vec2 volume_dimentions_meters;

	std::shared_ptr<Buffer> lattice;

	bool is_programs_compiled = false;
	std::shared_ptr<ComputeProgram> lbm2d_advect;
};