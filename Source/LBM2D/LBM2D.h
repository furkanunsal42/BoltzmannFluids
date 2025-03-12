#pragma once

#include <memory>

#include "LBMConstants/VelocitySet.h"
#include "LBMConstants/FloatingPointAccuracy.h"
#include "Tools/GraphicsOperation/GraphicsOperation.h"

#include "ComputeProgram.h"

class LBM2D {
public:
	
	void compile_shaders();
	
	void generate_lattice(glm::ivec2 resolution, glm::vec2 volume_dimentions_meters);

	void iterate_time(double time_milliseconds);
	double get_total_time_elapsed_ms();

	void set_floating_point_accuracy(FloatingPointAccuracy floating_point_accuracy);
	FloatingPointAccuracy get_floating_point_accuracy();

	void set_velocity_set(VelocitySet velocity_set);
	VelocitySet get_velocity_set();

	void copy_to_texture_velocity_index(Texture2D& target_texture, int32_t velocity_index);
	void copy_to_texture_velocity_total(Texture2D& target_texture);
	void copy_to_texture_velocity_magnetude(Texture2D& target_texture);
	void copy_to_texture_density(Texture2D& target_texture);

	glm::ivec2 get_resolution();
	int32_t get_velocity_count();
	glm::vec2 get_volume_dimentions_meters();

private:
	
	std::vector<std::pair<std::string, std::string>> _generate_shader_macros();
	
	void _stream();
	void _collide();
	void _apply_boundry_conditions();
	void _generate_lattice_buffer();

	double total_time_elapsed_ms = 0;
	VelocitySet velocity_set = D2Q9;
	FloatingPointAccuracy floating_point_accuracy = fp16;

	glm::ivec2 resolution;
	glm::vec2 volume_dimentions_meters;

	bool is_lattice_0_is_source = true;
	std::shared_ptr<Buffer> lattice0;
	std::shared_ptr<Buffer> lattice1;
	std::shared_ptr<Buffer> _get_lattice_source();
	std::shared_ptr<Buffer> _get_lattice_target();
	void _swap_lattice_buffers();

	bool is_programs_compiled = false;
	std::shared_ptr<ComputeProgram> lbm2d_stream;

	std::unique_ptr<UniformBuffer> lattice_velocity_set_buffer = nullptr;
	std::unique_ptr<GraphicsOperation> operation = std::make_unique<GraphicsOperation>();
};