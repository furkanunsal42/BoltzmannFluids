#pragma once

#include <memory>
#include <stdint.h>
#include <functional>

#include "LBMConstants/VelocitySet.h"
#include "LBMConstants/FloatingPointAccuracy.h"
#include "Tools/GraphicsOperation/GraphicsOperation.h"

#include "ComputeProgram.h"

// naming scheme of lattice boltzmann terms in variables and functions 
// 
// f = "discreate particle distribution function" is used as "population" as in "particle population function"
// feq = "equilibrium particle distribution function" is used as "equilibrium"
// c = "velocity set" is used as "velocity_set", ci refers to i'th velocity vector in the set
// q = number of velocity vectors in velocity set is called "velocity_count", "velocity_vector_count", or "population_count"
// 
// u = "velocity"
// rho = "density"

class LBM2D {
public:
	
	// simulation controls
	void iterate_time(std::chrono::duration<double, std::milli> deltatime);
	std::chrono::duration<double, std::milli> get_total_time_elapsed();

	glm::ivec2 get_resolution();
	int32_t get_velocity_set_vector_count();

	// high level field initialization api
	struct FluidProperties {
		glm::vec3 velocity = glm::vec3(0);
		glm::vec3 force = glm::vec3(0);
		float density = 1;
		float temperature = 20;
		float scalar_quantity = 0;
		bool is_boundry = false;
	};
	void initialize_fields(
		std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda,
		glm::ivec2 resolution,
		float relaxation_time,
		VelocitySet velocity_set = VelocitySet::D2Q9, 
		FloatingPointAccuracy fp_accuracy = FloatingPointAccuracy::fp32
	);
	
	// visualization
	void copy_to_texture_population(Texture2D& target_texture, int32_t population_index);
	void copy_to_texture_velocity_vector(Texture2D& target_texture);
	void copy_to_texture_velocity_magnetude(Texture2D& target_texture);
	void copy_to_texture_density(Texture2D& target_texture);
	void copy_to_texture_boundries(Texture2D& target_texture);

	// low level field initialization api
	void compile_shaders(); 
	void generate_lattice(glm::ivec2 resolution);

	void set_velocity_set(VelocitySet velocity_set);
	VelocitySet get_velocity_set();

	void set_floating_point_accuracy(FloatingPointAccuracy floating_point_accuracy);
	FloatingPointAccuracy get_floating_point_accuracy();

	void set_relaxation_time(float relaxation_time);
	float get_relaxation_time();

	void map_boundries();
	void unmap_boundries();
	bool is_boundries_mapped();
	void* get_mapped_boundries();
	void set_boundry(glm::ivec2 voxel_coordinate, bool value);
	void set_boundry(glm::ivec2 voxel_coordinate_begin, glm::ivec2 voxel_coordinate_end, bool value);
	void set_boundry(bool value);
	bool get_boundry(glm::ivec2 voxel_coordinate);

	void set_population(glm::ivec2 voxel_coordinate, int32_t population_index, float value);
	void set_population(glm::ivec2 voxel_coordinate_begin, glm::ivec2 voxel_coordinate_end, int32_t population_index, float value);
	void set_population(int32_t population_index, float value);
	void set_population(float value);

	void add_random_population(glm::ivec2 voxel_coordinate, int32_t population_index, float amplitude);
	void add_random_population(glm::ivec2 voxel_coordinate_begin, glm::ivec2 voxel_coordinate_end, int32_t population_index, float amplitude);
	void add_random_population(int32_t population_index, float amplitude);
	void add_random_population(float amplitude);

private:
	
	std::vector<std::pair<std::string, std::string>> _generate_shader_macros();
	
	void _stream();
	void _collide();
	void _apply_boundry_conditions();
	void _generate_lattice_buffer();

	// initialization functions
	void _collide_with_precomputed_velocities(Buffer& velocity_field);
	void _set_populations_to_equilibrium(Buffer& density_field, Buffer& velocity_field);

	std::chrono::duration<double, std::milli> total_time_elapsed;
	std::chrono::duration<double, std::milli> step_deltatime = std::chrono::duration<double, std::milli>(1);
	std::chrono::duration<double, std::milli> deltatime_overflow = std::chrono::duration<double, std::milli>(0);
	VelocitySet velocity_set = D2Q9;
	FloatingPointAccuracy floating_point_accuracy = fp32;

	glm::ivec2 resolution = glm::ivec2(0);
	float relaxation_time = 0.53f;

	bool is_lattice_0_is_source = true;
	std::shared_ptr<Buffer> lattice0 = nullptr;
	std::shared_ptr<Buffer> lattice1 = nullptr;
	std::shared_ptr<Buffer> boundries = nullptr;
	std::shared_ptr<Buffer> _get_lattice_source();
	std::shared_ptr<Buffer> _get_lattice_target();
	void _swap_lattice_buffers();
	
	bool is_programs_compiled = false;
	std::shared_ptr<ComputeProgram> lbm2d_stream = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_collide = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_boundry_condition = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_collide_with_precomputed_velocity = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_set_equilibrium_populations = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_set_population = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_add_random_population = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_copy_boundries = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_copy_density = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_copy_population = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_copy_velocity_magnitude = nullptr;
	std::shared_ptr<ComputeProgram> lbm2d_copy_velocity_total = nullptr;
	std::unique_ptr<UniformBuffer> lattice_velocity_set_buffer = nullptr;
};