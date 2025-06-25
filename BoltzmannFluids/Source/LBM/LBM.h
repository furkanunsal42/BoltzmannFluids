#pragma once

#include <memory>
#include <stdint.h>
#include <functional>
#include <filesystem>

#include "LBMConstants/VelocitySet.h"
#include "LBMConstants/FloatingPointAccuracy.h"
#include "Tools/GraphicsOperation/GraphicsOperation.h"

#include "ComputeProgram.h"
#include "Mesh.h"
#include "Camera.h"

// naming scheme of lattice boltzmann terms in variables and functions 
// 
// f = "discreate particle distribution function" is used as "population" as in "particle population function"
// feq = "equilibrium particle distribution function" is used as "equilibrium"
// c = "velocity set" is used as "velocity_set", ci refers to i'th velocity vector in the set
// q = number of velocity vectors in velocity set is called "velocity_count", "velocity_vector_count", or "population_count"
// 
// u = "velocity"
// rho = "density"

class LBM {
public:
	constexpr static uint32_t not_a_boundry = 0;
	constexpr static uint32_t max_boundry_count = 255;
	constexpr static float referance_temperature = 1.0;
	constexpr static float referance_boundry_density = 1.0;

	// simulation controls
	void iterate_time(float target_tick_per_second = 0);
	int32_t get_total_ticks_elapsed();
	std::chrono::duration<double, std::milli> get_total_time_elapsed();
	glm::ivec3 get_resolution();
	int32_t get_velocity_set_vector_count();

	// high level field initialization api
	struct FluidProperties {
		glm::vec3 velocity = glm::vec3(0);
		glm::vec3 force = glm::vec3(0);
		float density = 1;
		float temperature = referance_temperature;
		uint32_t boundry_id = not_a_boundry;
	};
	void initialize_fields(
		std::function<void(glm::ivec3, FluidProperties&)> initialization_lambda,
		glm::ivec3 resolution,
		float relaxation_time,
		bool periodic_x = true,
		bool periodic_y = true,
		bool periodic_z = true,
		VelocitySet velocity_set = VelocitySet::D2Q9,
		FloatingPointAccuracy fp_accuracy = FloatingPointAccuracy::fp32,
		bool is_flow_multiphase = false
	);

	void set_boundry_properties(
		uint32_t boundry_id,
		glm::vec3 velocity_translational,
		glm::vec3 velocity_angular,
		glm::vec3 center_of_mass,
		float temperature = referance_temperature,
		float effective_density = referance_boundry_density
	);
	
	void set_boundry_properties(
		uint32_t boundry_id,
		glm::vec3 velocity_translational,
		float temperature = referance_temperature,
		float effective_density = referance_boundry_density
	);

	void set_boundry_properties(
		uint32_t boundry_id,
		float temperature,
		float effective_density = referance_boundry_density
	);

	void clear_boundry_properties();

	// high level visualization api
	void render2d_density();
	void render2d_velocity();
	void render2d_boundries();
	void render2d_forces();
	void render2d_temperature();

	void render3d_density(Camera& camera, int32_t sample_count = 128);
	void render3d_velocity(Camera& camera, int32_t sample_count = 128);
	void render3d_boundries(Camera& camera, int32_t sample_count = 128);
	void render3d_forces(Camera& camera, int32_t sample_count = 128);
	void render3d_temperature(Camera& camera, int32_t sample_count = 128);

	std::shared_ptr<Texture3D> get_velocity_density_texture();
	std::shared_ptr<Texture3D> get_boundry_texture();
	std::shared_ptr<Texture3D> get_force_temperature_texture();

	// low level visualization api
	//void copy_to_texture_population(Texture2D& target_texture, int32_t population_index);

	// low level field initialization api
	float get_relaxation_time();
	VelocitySet get_velocity_set();
	int32_t get_dimentionality();
	FloatingPointAccuracy get_floating_point_accuracy();
	bool get_periodic_boundry_x();
	bool get_periodic_boundry_y();
	bool get_periodic_boundry_z();
	bool get_is_forcing_scheme();
	bool get_is_force_field_constant();
	bool get_is_flow_multiphase();
	glm::vec3 get_constant_force();
	float get_intermolecular_interaction_strength();

	void set_relaxation_time(float relaxation_time);
 	void set_constant_force(glm::vec3 constant_force);
	void set_intermolecular_interaction_strength(float value);

	// high level visualization api
	void save_current_tick_macroscopic(std::filesystem::path save_path);
	void save_current_tick_mesoscropic(std::filesystem::path save_path);
	void load_tick_macroscopic(std::filesystem::path save_path, int32_t target_tick);
	void load_tick_mesoscropic(std::filesystem::path save_path, int32_t target_tick);

	//void set_population(glm::ivec2 voxel_coordinate, int32_t population_index, float value);
	//void set_population(glm::ivec2 voxel_coordinate_begin, glm::ivec2 voxel_coordinate_end, int32_t population_index, float value);
	//void set_population(int32_t population_index, float value);
	//void set_population(float value);
	//
	//void add_random_population(glm::ivec2 voxel_coordinate, int32_t population_index, float amplitude);
	//void add_random_population(glm::ivec2 voxel_coordinate_begin, glm::ivec2 voxel_coordinate_end, int32_t population_index, float amplitude);
	//void add_random_population(int32_t population_index, float amplitude);
	//void add_random_population(float amplitude);


    bool is_lattice_texture3d = false;
    Texture3D::ColorTextureFormat lattice_tex_internal_format = Texture3D::ColorTextureFormat::R16F;
	bool is_collide_esoteric = false;

    float velocity_limit = 0.25;
    float velocity_limit_extreme = 0.30;

//private:
	
	std::vector<std::pair<std::string, std::string>> _generate_shader_macros();
	void _compile_shaders();
	void _generate_lattice(glm::ivec3 resolution);
	
	size_t _coord_to_id(glm::uvec3 coord);
	size_t _coord_to_id(uint32_t x, uint32_t y, uint32_t z);
	size_t _get_voxel_count();

	void _stream();
	void _collide(bool save_macrsoscopic_results);
	void _generate_lattice_buffer();

	// initialization functions
	void _collide_with_precomputed_velocities(Buffer& velocity_field);
	void _set_populations_to_equilibrium(Buffer& density_field, Buffer& velocity_field);

	void _initialize_fields_default_pass(
		std::function<void(glm::ivec3, FluidProperties&)> initialization_lambda,
		std::shared_ptr<Buffer>& out_density_field,
		std::shared_ptr<Buffer>& out_velocity_field
		);
	void _initialize_fields_boundries_pass(
		std::function<void(glm::ivec3, FluidProperties&)> initialization_lambda
		);
	void _initialize_fields_force_pass(
		std::function<void(glm::ivec3, FluidProperties&)> initialization_lambda
		);
	void _initialize_fields_thermal_pass(
		std::function<void(glm::ivec3, FluidProperties&)> initialization_lambda,
		std::shared_ptr<Buffer> in_velocity_field
		);

	// simulation time controls
	bool first_iteration = true;
	size_t total_ticks_elapsed = 0;
	std::chrono::time_point<std::chrono::system_clock> simulation_begin;
	std::chrono::time_point<std::chrono::system_clock> last_visual_update;

	// LBM simulation parameters
	VelocitySet velocity_set = D2Q9;
	FloatingPointAccuracy floating_point_accuracy = fp32;

	glm::ivec3 resolution = glm::ivec3(0);
	float relaxation_time = 0.53f;
	
	void _set_velocity_set(VelocitySet velocity_set);
	void _set_floating_point_accuracy(FloatingPointAccuracy floating_point_accuracy);

	// forces control flags
	bool is_forcing_scheme = false;
	bool is_force_field_constant = true;
	glm::vec3 constant_force = glm::vec3(0);
	void _set_is_forcing_scheme(bool value);
	void _set_is_force_field_constant(bool value);

	// moving/stationary boundries control flags
	bool periodic_x = true;
	bool periodic_y = true;
	bool periodic_z = true;
	void _set_periodic_boundry_x(bool value);
	void _set_periodic_boundry_y(bool value);
	void _set_periodic_boundry_z(bool value);

	struct _object_desc {
	public:
		_object_desc(
			glm::vec3 velocity_translational = glm::vec3(0),
			glm::vec3 velocity_angular = glm::vec3(0),
			glm::vec3 center_of_mass = glm::vec3(0),
			float temperature = referance_temperature,
			float effective_density = referance_boundry_density
		);

		glm::vec3 velocity_translational;
		glm::vec3 velocity_angular;
		glm::vec3 center_of_mass;
		float temperature;
		float effective_density;
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
	float thermal_expension_coeficient = 0.5f;

	void _set_is_flow_thermal(bool value);
	bool _get_is_flow_thermal();

	void _set_thermal_lattice_velocity_set(SimplifiedVelocitySet set);
	SimplifiedVelocitySet _get_thermal_lattice_velocity_set();
	
	// thermal flow physics
	void _stream_thermal();
	void _set_populations_to_equilibrium_thermal(Buffer& temperature_field, Buffer& velocity_field);

	// multiphase flow control flags
	bool is_flow_multiphase = true;
	float intermolecular_interaction_strength = -6.0f;
	void _set_is_flow_multiphase(bool value);

	// device buffers
	std::shared_ptr<Texture3D> lattice0_tex = nullptr;
	std::shared_ptr<Texture3D> lattice1_tex = nullptr;
	std::shared_ptr<Buffer> lattice0 = nullptr;
	std::shared_ptr<Buffer> lattice1 = nullptr;
	std::shared_ptr<Buffer> boundries = nullptr;
	std::shared_ptr<Buffer> objects = nullptr;
	std::shared_ptr<Buffer> forces = nullptr;
	std::shared_ptr<Buffer> thermal_lattice0 = nullptr;
	std::shared_ptr<Buffer> thermal_lattice1 = nullptr;

	std::unique_ptr<UniformBuffer> lattice_velocity_set_buffer = nullptr;
	std::unique_ptr<UniformBuffer> thermal_lattice_velocity_set_buffer = nullptr;

	// dual buffer control
	bool is_lattice_0_is_source = true;
	std::shared_ptr<Buffer> _get_lattice_source();
	std::shared_ptr<Buffer> _get_lattice_target();
	std::shared_ptr<Texture3D> _get_lattice_tex_source();
	std::shared_ptr<Texture3D> _get_lattice_tex_target();
	void _swap_lattice_buffers();

	bool is_thermal_lattice_0_is_source = true;
	std::shared_ptr<Buffer> _get_thermal_lattice_source();
	std::shared_ptr<Buffer> _get_thermal_lattice_target();
	void _swap_thermal_lattice_buffers();
	
	// macroscopic variable textures
	std::shared_ptr<Texture3D> velocity_density_texture = nullptr;
	std::shared_ptr<Texture3D> boundry_texture = nullptr;
	std::shared_ptr<Texture3D> force_temperature_texture = nullptr;
	void _generate_macroscopic_textures();

	Texture3D::ColorTextureFormat velocity_density_texture_internal_format = Texture3D::ColorTextureFormat::RGBA32F;
	Texture3D::ColorTextureFormat boundry_texture_internal_format = Texture3D::ColorTextureFormat::R8;
	Texture3D::ColorTextureFormat force_temperature_texture_internal_format = Texture3D::ColorTextureFormat::RGBA32F;

	// kernels
	bool is_programs_compiled = false;
	std::shared_ptr<ComputeProgram> lbm_stream = nullptr;
	std::shared_ptr<ComputeProgram> lbm_stream_thermal = nullptr;
	std::shared_ptr<ComputeProgram> lbm_collide = nullptr;
	std::shared_ptr<ComputeProgram> lbm_collide_save = nullptr;
	std::shared_ptr<ComputeProgram> lbm_collide_with_precomputed_velocity = nullptr;
	std::shared_ptr<ComputeProgram> lbm_set_equilibrium_populations = nullptr;
	std::shared_ptr<ComputeProgram> lbm_set_equilibrium_populations_thermal = nullptr;
	std::shared_ptr<ComputeProgram> lbm_set_population = nullptr;
	std::shared_ptr<ComputeProgram> lbm_add_random_population = nullptr;
	std::shared_ptr<ComputeProgram> lbm_copy_population = nullptr;
	
	// renderers
	std::shared_ptr<Program> program_render2d_density = nullptr;
	std::shared_ptr<Program> program_render2d_velocity = nullptr;
	std::shared_ptr<Program> program_render2d_boundries = nullptr;
	std::shared_ptr<Program> program_render2d_forces = nullptr;
	std::shared_ptr<Program> program_render2d_temperature = nullptr;

	std::shared_ptr<Program> program_render_volumetric_density = nullptr;
	std::shared_ptr<Program> program_render_volumetric_velocity = nullptr;
	std::shared_ptr<Program> program_render_volumetric_boundries = nullptr;
	std::shared_ptr<Program> program_render_volumetric_forces = nullptr;
	std::shared_ptr<Program> program_render_volumetric_temperature = nullptr;

	std::shared_ptr<Mesh> plane_mesh = nullptr;
	std::shared_ptr<Mesh> plane_cube = nullptr;
};
