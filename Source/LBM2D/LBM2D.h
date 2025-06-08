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
	constexpr static uint32_t not_a_boundry = 0;
	constexpr static uint32_t max_boundry_count = 255;
	constexpr static float referance_temperature = 1.0;

	// simulation controls
	void iterate_time(std::chrono::duration<double, std::milli> deltatime);
	std::chrono::duration<double, std::milli> get_total_time_elapsed();

	glm::ivec2 get_resolution();
	int32_t get_velocity_set_vector_count();

	// high level field initialization api
	void set_boundry_properties(
		uint32_t boundry_id,
		glm::vec3 velocity_translational,
		glm::vec3 velocity_angular,
		glm::vec3 center_of_mass,
		float temperature
	);
	
	void set_boundry_properties(
		uint32_t boundry_id,
		glm::vec3 velocity_translational, 
		glm::vec3 velocity_angular,
		glm::vec3 center_of_mass
	);

	void set_boundry_properties(
		uint32_t boundry_id,
		glm::vec3 velocity_translational,
		float temperature
	);

	void set_boundry_properties(
		uint32_t boundry_id,
		glm::vec3 velocity_translational
	);

	void set_boundry_properties(
		uint32_t boundry_id,
		float temperature
	);

	struct FluidProperties {
		glm::vec3 velocity = glm::vec3(0);
		glm::vec3 force = glm::vec3(0);
		float density = 1;
		float temperature = referance_temperature;
		float scalar_quantity = 0;
		uint32_t boundry_id = not_a_boundry;
	};
	void initialize_fields(
		std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda,
		glm::ivec2 resolution,
		float relaxation_time,
		bool periodic_x =  true,
		bool periodic_y = true,
		VelocitySet velocity_set = VelocitySet::D2Q9, 
		FloatingPointAccuracy fp_accuracy = FloatingPointAccuracy::fp32
	);
	
	// visualization
	void copy_to_texture_population(Texture2D& target_texture, int32_t population_index);
	void copy_to_texture_velocity_vector(Texture2D& target_texture);
	void copy_to_texture_velocity_magnetude(Texture2D& target_texture);
	void copy_to_texture_density(Texture2D& target_texture);
	void copy_to_texture_boundries(Texture2D& target_texture);
	void copy_to_texture_force_vector(Texture2D& target_texture);

	// low level field initialization api
	void compile_shaders(); 
	void generate_lattice(glm::ivec2 resolution);

	void set_velocity_set(VelocitySet velocity_set);
	VelocitySet get_velocity_set();

	void set_floating_point_accuracy(FloatingPointAccuracy floating_point_accuracy);
	FloatingPointAccuracy get_floating_point_accuracy();

	void set_relaxation_time(float relaxation_time);
	float get_relaxation_time();

	void set_periodic_boundry_x(bool value);
	bool get_periodic_boundry_x();

	void set_periodic_boundry_y(bool value);
	bool get_periodic_boundry_y();

	void set_is_forcing_scheme(bool value);
	bool get_is_forcing_scheme();

	void set_is_force_field_constant(bool value);
	bool get_is_force_field_constant();

	void set_constant_force(glm::vec3 constant_force);
	glm::vec3 get_constant_force();

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
	void _initialize_fields_default_pass(
		std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda,
		glm::ivec2 resolution,
		FloatingPointAccuracy fp_accuracy = FloatingPointAccuracy::fp32
	);
	void _initialize_fields_boundries_pass(
		std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda,
		glm::ivec2 resolution,
		FloatingPointAccuracy fp_accuracy = FloatingPointAccuracy::fp32
	);
	void _initialize_fields_force_pass(
		std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda,
		glm::ivec2 resolution,
		FloatingPointAccuracy fp_accuracy = FloatingPointAccuracy::fp32
	);

	// simulation time controls
	std::chrono::duration<double, std::milli> total_time_elapsed;
	std::chrono::duration<double, std::milli> step_deltatime = std::chrono::duration<double, std::milli>(1);
	std::chrono::duration<double, std::milli> deltatime_overflow = std::chrono::duration<double, std::milli>(0);
	
	// LBM simulation parameters
	VelocitySet velocity_set = D2Q9;
	FloatingPointAccuracy floating_point_accuracy = fp32;

	glm::ivec2 resolution = glm::ivec2(0);
	float relaxation_time = 0.53f;
	
	// forces control flags
	bool is_forcing_scheme = false;
	bool is_force_field_constant = true;
	glm::vec3 constant_force = glm::vec3(0);

	// moving/stationary boundries control flags
	bool periodic_x = true;
	bool periodic_y = true;
	
	struct _object_desc {
	public:
		_object_desc(
			glm::vec3 velocity_translational = glm::vec3(0),
			glm::vec3 velocity_angular = glm::vec3(0),
			glm::vec3 center_of_mass = glm::vec3(0),
			float temperature = referance_temperature
		);

		glm::vec3 velocity_translational;
		glm::vec3 velocity_angular;
		glm::vec3 center_of_mass;
		float temperature;
	};

	// boundries buffer holds the id of the object it is a part of (0 means not_a_boundry)
	// number of bits per voxel can change dynamically basad on how many objects are defined
	// velocity information of each object is held in another buffer in device called "objects"
	// objects buffer schema is [vec4 translational_velcoity, vec4 rotational_velocity, vec4 center_of_mass] 
	std::vector<_object_desc> objects_cpu;
	int32_t bits_per_boundry = 0;
	void _set_bits_per_boundry(int32_t value);
	int32_t _get_bits_per_boundry(int32_t value);

	// thermal flow control flags
	bool is_flow_thermal = false;
	SimplifiedVelocitySet thermal_lattice_velocity_set = SimplifiedVelocitySet::D2Q5;
	float thermal_relaxation_time = 0.53;

	void _set_is_flow_thermal(bool value);
	bool _get_is_flow_thermal();

	void _set_thermal_lattice_velocity_set(SimplifiedVelocitySet set);
	SimplifiedVelocitySet _get_thermal_lattice_velocity_set();

	// device buffers
	std::shared_ptr<Buffer> lattice0 = nullptr;
	std::shared_ptr<Buffer> lattice1 = nullptr;
	std::shared_ptr<Buffer> boundries = nullptr;
	std::shared_ptr<Buffer> objects = nullptr;
	std::shared_ptr<Buffer> forces = nullptr;
	std::shared_ptr<Buffer> temperature_lattice0 = nullptr;
	std::shared_ptr<Buffer> temperature_lattice1 = nullptr;

	// dual buffer control
	bool is_lattice_0_is_source = true;
	std::shared_ptr<Buffer> _get_lattice_source();
	std::shared_ptr<Buffer> _get_lattice_target();
	void _swap_lattice_buffers();

	bool is_temperature_lattice_0_is_source = true;
	std::shared_ptr<Buffer> _get_temperature_lattice_source();
	std::shared_ptr<Buffer> _get_temperature_lattice_target();
	void _swap_temperature_lattice_buffers();
	
	// kernels
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
	std::shared_ptr<ComputeProgram> lbm2d_copy_force_total = nullptr;
	std::unique_ptr<UniformBuffer> lattice_velocity_set_buffer = nullptr;
};