#include "LBM.h"
#include "Application/ProgramSourcePaths.h"
#include "PrimitiveRenderer.h"
#include "Camera.h"

void LBM::_compile_shaders()
{
	if (is_programs_compiled)
		return;

	is_programs_compiled = true;

	// simulation kernels

	auto definitions = _generate_shader_macros();

	auto definitions_plus_not_save = definitions;
	definitions_plus_not_save.push_back(std::pair("save_macroscopic_variables", "0"));

	auto definitions_plus_save = definitions;
	definitions_plus_save.push_back(std::pair("save_macroscopic_variables", "1"));

	std::cout << "[LBM Info] kernels are compiled with configuration : " << std::endl;
	for (auto& definition : definitions)
		std::cout << "\t" << definition.first << " : " << definition.second << std::endl;

	lbm_stream									= std::make_shared<ComputeProgram>(Shader(lbm_shader_directory / "stream.comp"), definitions);
	lbm_stream_thermal							= std::make_shared<ComputeProgram>(Shader(lbm_shader_directory / "stream_thermal.comp"), definitions);
	lbm_collide									= std::make_shared<ComputeProgram>(Shader(lbm_shader_directory / "collide.comp"), definitions_plus_not_save);
	lbm_collide_save							= std::make_shared<ComputeProgram>(Shader(lbm_shader_directory / "collide.comp"), definitions_plus_save);
	lbm_collide_with_precomputed_velocity		= std::make_shared<ComputeProgram>(Shader(lbm_shader_directory / "collide_with_precomputed_velocity.comp"), definitions);
	lbm_set_equilibrium_populations				= std::make_shared<ComputeProgram>(Shader(lbm_shader_directory / "set_equilibrium_populations.comp"), definitions);
	lbm_set_equilibrium_populations_thermal		= std::make_shared<ComputeProgram>(Shader(lbm_shader_directory / "set_equilibrium_populations_thermal.comp"), definitions);
	lbm_set_population							= std::make_shared<ComputeProgram>(Shader(lbm_shader_directory / "set_population.comp"), definitions);
	lbm_copy_population							= std::make_shared<ComputeProgram>(Shader(lbm_shader_directory / "copy_population.comp"), definitions);
	lbm_add_random_population					= std::make_shared<ComputeProgram>(Shader(lbm_shader_directory / "add_random_population.comp"), definitions);

	// renderer2d

	program_render2d_density					= std::make_shared<Program>(Shader(renderer2d_shader_directory / "basic.vert", renderer2d_shader_directory / "density_2d.frag"));
	program_render2d_velocity					= std::make_shared<Program>(Shader(renderer2d_shader_directory / "basic.vert", renderer2d_shader_directory / "velocity_2d.frag"));;
	program_render2d_boundries					= std::make_shared<Program>(Shader(renderer2d_shader_directory / "basic.vert", renderer2d_shader_directory / "boundries_2d.frag"));;
	program_render2d_forces						= std::make_shared<Program>(Shader(renderer2d_shader_directory / "basic.vert", renderer2d_shader_directory / "forces_2d.frag"));;
	program_render2d_temperature				= std::make_shared<Program>(Shader(renderer2d_shader_directory / "basic.vert", renderer2d_shader_directory / "temperature_2d.frag"));;

	program_render_volumetric_density			= std::make_shared<Program>(Shader(renderer3d_shader_directory / "basic.vert", renderer3d_shader_directory / "density_volumetric.frag"));
	program_render_volumetric_velocity			= std::make_shared<Program>(Shader(renderer3d_shader_directory / "basic.vert", renderer3d_shader_directory / "velocity_volumetric.frag"));
	program_render_volumetric_boundries			= std::make_shared<Program>(Shader(renderer3d_shader_directory / "basic.vert", renderer3d_shader_directory / "boundries_volumetric.frag"));
	program_render_volumetric_forces			= std::make_shared<Program>(Shader(renderer3d_shader_directory / "basic.vert", renderer3d_shader_directory / "forces_volumetric.frag"));
	program_render_volumetric_temperature		= std::make_shared<Program>(Shader(renderer3d_shader_directory / "basic.vert", renderer3d_shader_directory / "temperature_volumetric.frag"));

	SingleModel plane_model;
	plane_model.verticies = {
		glm::vec3(-1, -1, 0),
		glm::vec3( 1, -1, 0),
		glm::vec3(-1,  1, 0),
		glm::vec3( 1,  1, 0),
	};
	plane_model.texture_coordinates_0 = {
		glm::vec2(0, 0),
		glm::vec2(1, 0),
		glm::vec2(0, 1),
		glm::vec2(1, 1),
	};
	plane_model.indicies = {
		0, 1, 2,
		2, 1, 3
	};

	plane_mesh = std::make_shared<Mesh>();
	plane_mesh->load_model(plane_model);

	glm::vec3 scale(1, 1, 1);

	SingleModel cube_model;
	cube_model.verticies = {
		glm::vec3(-0.5f * scale.x, -0.5f * scale.y,  0.5f * scale.z),//front
		glm::vec3( 0.5f * scale.x, -0.5f * scale.y,  0.5f * scale.z),
		glm::vec3( 0.5f * scale.x,  0.5f * scale.y,  0.5f * scale.z),
		glm::vec3(-0.5f * scale.x,  0.5f * scale.y,  0.5f * scale.z),
		
		glm::vec3( 0.5f * scale.x, -0.5f * scale.y,  0.5f * scale.z),//right
		glm::vec3( 0.5f * scale.x, -0.5f * scale.y, -0.5f * scale.z),
		glm::vec3( 0.5f * scale.x,  0.5f * scale.y, -0.5f * scale.z),
		glm::vec3( 0.5f * scale.x,  0.5f * scale.y,  0.5f * scale.z),
		
		glm::vec3(-0.5f * scale.x,  0.5f * scale.y, -0.5f * scale.z),//top
		glm::vec3(-0.5f * scale.x,  0.5f * scale.y,  0.5f * scale.z),
		glm::vec3( 0.5f * scale.x,  0.5f * scale.y,  0.5f * scale.z),
		glm::vec3( 0.5f * scale.x,  0.5f * scale.y, -0.5f * scale.z),
		
		glm::vec3( 0.5f * scale.x, -0.5f * scale.y, -0.5f * scale.z),//back
		glm::vec3(-0.5f * scale.x, -0.5f * scale.y, -0.5f * scale.z),
		glm::vec3(-0.5f * scale.x,  0.5f * scale.y, -0.5f * scale.z),
		glm::vec3( 0.5f * scale.x,  0.5f * scale.y, -0.5f * scale.z),
		
		glm::vec3(-0.5f * scale.x, -0.5f * scale.y, -0.5f * scale.z),//left
		glm::vec3(-0.5f * scale.x, -0.5f * scale.y,  0.5f * scale.z),
		glm::vec3(-0.5f * scale.x,  0.5f * scale.y,  0.5f * scale.z),
		glm::vec3(-0.5f * scale.x,  0.5f * scale.y, -0.5f * scale.z),
		
		glm::vec3( 0.5f * scale.x,  -0.5f * scale.y,  0.5f * scale.z),//bottom
		glm::vec3(-0.5f * scale.x,  -0.5f * scale.y,  0.5f * scale.z),
		glm::vec3(-0.5f * scale.x,  -0.5f * scale.y, -0.5f * scale.z),
		glm::vec3( 0.5f * scale.x,  -0.5f * scale.y, -0.5f * scale.z),
	};

	cube_model.texture_coordinates_0 = {
		glm::vec2(0.0f, 0.0f),
		glm::vec2(1.0f, 0.0f),
		glm::vec2(1.0f, 1.0f),
		glm::vec2(0.0f, 1.0f),

		glm::vec2(0.0f, 0.0f),
		glm::vec2(1.0f, 0.0f),
		glm::vec2(1.0f, 1.0f),
		glm::vec2(0.0f, 1.0f),

		glm::vec2(0.0f, 0.0f),
		glm::vec2(1.0f, 0.0f),
		glm::vec2(1.0f, 1.0f),
		glm::vec2(0.0f, 1.0f),

		glm::vec2(0.0f, 0.0f),
		glm::vec2(1.0f, 0.0f),
		glm::vec2(1.0f, 1.0f),
		glm::vec2(0.0f, 1.0f),

		glm::vec2(0.0f, 0.0f),
		glm::vec2(1.0f, 0.0f),
		glm::vec2(1.0f, 1.0f),
		glm::vec2(0.0f, 1.0f),

		glm::vec2(0.0f, 0.0f),
		glm::vec2(0.0f, 1.0f),
		glm::vec2(1.0f, 1.0f),
		glm::vec2(1.0f, 0.0f),
	};

	cube_model.vertex_normals = {
		glm::vec3(0.0f, 0.0f, 1.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
	
		glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(1.0f, 0.0f, 0.0f),
	
		glm::vec3(0.0f, 1.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
	
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
	
		glm::vec3(-1.0f, 0.0f, 0.0f),
		glm::vec3(-1.0f, 0.0f, 0.0f),
		glm::vec3(-1.0f, 0.0f, 0.0f),
		glm::vec3(-1.0f, 0.0f, 0.0f),
	
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, -1.0f, 0.0f),
	};

	cube_model.indicies = {
			0, 1, 2, 0, 2, 3,
			4, 5, 6, 4, 6, 7,
			8, 9, 10, 8, 10, 11,
			12, 13, 14, 12, 14, 15,
			16, 17, 18, 16, 18, 19,
			20, 21, 22, 20, 22, 23,
	};
	
	plane_cube = std::make_shared<Mesh>();
	plane_cube->load_model(cube_model);
}

void LBM::_generate_lattice(glm::ivec3 resolution)
{
	if (glm::any(glm::equal(resolution, glm::ivec3(0)))) {
		std::cout << "[LBM Error] LBM::generate_lattice() is called with zero resolution" << std::endl;
		ASSERT(false);
	}

	this->resolution = resolution;

	_generate_lattice_buffer();
}

size_t LBM::_coord_to_id(glm::uvec3 coord)
{
	return _coord_to_id(coord.x, coord.y, coord.z);
}

size_t LBM::_coord_to_id(uint32_t x, uint32_t y, uint32_t z)
{
	return z * resolution.y * resolution.x + y * resolution.x + x;
}

size_t LBM::_get_voxel_count()
{
	return resolution.x * resolution.y * resolution.z;
}

void LBM::iterate_time(float target_tick_per_second)
{
	if (first_iteration) {
		simulation_begin = std::chrono::system_clock::now();
		last_visual_update = std::chrono::system_clock::now();
	}

	size_t targeted_tick_count = target_tick_per_second * get_total_time_elapsed().count() / 1000.0f;
	if (target_tick_per_second <= 0 || total_ticks_elapsed < targeted_tick_count) {
		
		auto time_since_visual_update = std::chrono::system_clock::now() - last_visual_update;
		bool should_update_visuals = first_iteration || std::chrono::duration_cast<std::chrono::milliseconds>(time_since_visual_update).count() > 1000.0 / 60;
		if (should_update_visuals) last_visual_update = std::chrono::system_clock::now();

		_collide(should_update_visuals);
		
		if (!is_collide_esoteric)
			_stream();
		
		if (is_flow_thermal)
			_stream_thermal();

		total_ticks_elapsed++;
	}
	
	first_iteration = false;
}

int32_t LBM::get_total_ticks_elapsed()
{
	return total_ticks_elapsed;
}

std::chrono::duration<double, std::milli> LBM::get_total_time_elapsed()
{
	return std::chrono::system_clock::now() - simulation_begin;
}

void LBM::_set_floating_point_accuracy(FloatingPointAccuracy floating_point_accuracy)
{
	if (floating_point_accuracy == this->floating_point_accuracy)
		return;
	
	this->floating_point_accuracy = floating_point_accuracy;
	is_programs_compiled = false;
}

FloatingPointAccuracy LBM::get_floating_point_accuracy()
{
	return floating_point_accuracy;
}

void LBM::_set_velocity_set(VelocitySet velocity_set)
{
	if (velocity_set == this->velocity_set)
		return;

	this->velocity_set = velocity_set;
	is_programs_compiled = false;
}

VelocitySet LBM::get_velocity_set()
{
	return velocity_set;
}

int32_t LBM::get_dimentionality()
{
	return get_VelocitySet_dimention(get_velocity_set());
}

void LBM::set_relaxation_time(float relaxation_time)
{
	this->relaxation_time = relaxation_time;
}

float LBM::get_relaxation_time()
{
	return relaxation_time;
}

void LBM::_set_periodic_boundry_x(bool value)
{
	if (periodic_x == value)
		return;

	periodic_x = value;
	is_programs_compiled = false;
}

bool LBM::get_periodic_boundry_x()
{
	return periodic_x;
}

void LBM::_set_periodic_boundry_y(bool value)
{
	if (periodic_y == value)
		return;

	periodic_y = value;
	is_programs_compiled = false;
}


bool LBM::get_periodic_boundry_y()
{
	return periodic_y;
}

bool LBM::get_periodic_boundry_z()
{
	return periodic_z;
}

void LBM::_set_periodic_boundry_z(bool value)
{
	if (periodic_z == value)
		return;

	periodic_z = value;
	is_programs_compiled = false;
}

void LBM::_set_is_forcing_scheme(bool value)
{
	if (is_forcing_scheme == value)
		return;

	is_forcing_scheme = value;
	is_programs_compiled = false;
}

bool LBM::get_is_forcing_scheme()
{
	return is_forcing_scheme;
}

void LBM::_set_is_force_field_constant(bool value)
{
	if (is_force_field_constant == value)
		return;

	is_force_field_constant = value;
	is_programs_compiled = false;
}

bool LBM::get_is_force_field_constant()
{
	return is_force_field_constant;
}

void LBM::set_constant_force(glm::vec3 constant_force)
{
	this->constant_force = constant_force;
}

void LBM::set_intermolecular_interaction_strength(float value)
{
	intermolecular_interaction_strength = value;
}

glm::vec3 LBM::get_constant_force()
{
	return constant_force;
}

float LBM::get_intermolecular_interaction_strength()
{
	return intermolecular_interaction_strength;
}

void LBM::_set_is_flow_multiphase(bool value)
{
	if (is_flow_multiphase == value)
		return;

	is_flow_multiphase = value;
	is_programs_compiled = false;
}

bool LBM::get_is_flow_multiphase()
{
	return is_flow_multiphase;
}

void LBM::initialize_fields(
	std::function<void(glm::ivec3, FluidProperties&)> initialization_lambda, 
	glm::ivec3 resolution, 
	float relaxation_time, 
	bool periodic_x, 
	bool periodic_y,
	bool periodic_z,
	VelocitySet velocity_set, 
	FloatingPointAccuracy fp_accuracy,
	bool is_flow_multiphase
) {
	is_programs_compiled = false;
	first_iteration = true;

	if (glm::any(glm::lessThanEqual(resolution, glm::ivec3(0)))) {
		std::cout << "[LBM Error] LBM::initialize_fields() is called but given resolution is not valid, all components must be greater than zero" << std::endl;
		ASSERT(false);
	}

	if (get_VelocitySet_dimention(velocity_set) == 2 && resolution.z != 1) {
		std::cout << "[LBM Error] LBM::initialize_fields() is called but given velocity set dimention doesn't match the resolution dimention" << std::endl;
		ASSERT(false);
	}

	if (objects_cpu.size() == 0)
		objects_cpu.resize(1);
	objects_cpu[0] = _object_desc();

	set_relaxation_time(relaxation_time);
	_set_velocity_set(velocity_set);
	_set_floating_point_accuracy(fp_accuracy);
	_set_periodic_boundry_x(periodic_x);
	_set_periodic_boundry_y(periodic_y);
	_set_periodic_boundry_z(periodic_z);
	_set_thermal_lattice_velocity_set(
		get_VelocitySet_dimention(velocity_set) == 2 ? 
		SimplifiedVelocitySet::D2Q5 : SimplifiedVelocitySet::D3Q7
	);
	thermal_relaxation_time = 2;
	_set_is_flow_multiphase(is_flow_multiphase);
	
	_generate_lattice(resolution);

	std::shared_ptr<Buffer> density_field = nullptr;
	std::shared_ptr<Buffer> velocity_field = nullptr;
	
	_initialize_fields_default_pass(
		initialization_lambda,
		density_field,
		velocity_field
	);
	
	_initialize_fields_boundries_pass(
		initialization_lambda
	);
	
	_initialize_fields_force_pass(
		initialization_lambda
	);
	
	_initialize_fields_thermal_pass(
		initialization_lambda,
		velocity_field
	);
	
	int32_t relaxation_iteration_count = 0;
	std::cout << "[LBM Info] _initialize_fields_default_pass() initialization of particle population distributions from given veloicty and density fields is initiated" << std::endl;
	
	_set_populations_to_equilibrium(*density_field, *velocity_field);
	
	for (int32_t i = 0; i < relaxation_iteration_count; i++) {
		_collide_with_precomputed_velocities(*velocity_field);
		//_apply_boundry_conditions(); //the book doesn't specitfy whether or not to enforce boundry conditions in initialization algorithm
		_stream();
	}

	iterate_time();

	std::cout << "[LBM Info] _initialize_fields_default_pass() fields initialization scheme completed with relaxation_iteration_count(" << relaxation_iteration_count << ")" << std::endl;
}

void LBM::_initialize_fields_default_pass(
	std::function<void(glm::ivec3, FluidProperties&)> initialization_lambda,
	std::shared_ptr<Buffer>& out_density_field,
	std::shared_ptr<Buffer>& out_velocity_field
) {

	std::shared_ptr<Buffer> velocity_buffer = std::make_shared<Buffer>(_get_voxel_count() * sizeof(glm::vec4));
	std::shared_ptr<Buffer> density_buffer = std::make_shared<Buffer>(_get_voxel_count() * sizeof(float));

	velocity_buffer->map(Buffer::MapInfo(Buffer::MapInfo::Upload, Buffer::MapInfo::Temporary));
	density_buffer->map(Buffer::MapInfo(Buffer::MapInfo::Upload, Buffer::MapInfo::Temporary));

	glm::vec4* velocity_buffer_data = (glm::vec4*)velocity_buffer->get_mapped_pointer();
	float* density_buffer_data = (float*)density_buffer->get_mapped_pointer();

	FluidProperties temp_properties;
	initialization_lambda(glm::ivec3(0, 0, 0), temp_properties);

	_set_is_force_field_constant(true);
	set_constant_force(temp_properties.force);

	_set_is_flow_thermal(false);
	
	uint32_t object_count = 1;

	for (int32_t z = 0; z < resolution.z; z++) {
		for (int32_t y = 0; y < resolution.y; y++) {
			for (int32_t x = 0; x < resolution.x; x++) {
				FluidProperties properties;
				initialization_lambda(glm::ivec3(x, y, z), properties);

				if (properties.boundry_id != 0)
					properties.velocity = glm::vec3(0);

				object_count = std::max(object_count, properties.boundry_id + 1);

				if (properties.force != constant_force)
					_set_is_force_field_constant(false);
				if (properties.temperature != temp_properties.temperature)
					_set_is_flow_thermal(true);
				if (properties.boundry_id != not_a_boundry && objects_cpu[properties.boundry_id].temperature != temp_properties.temperature)
					_set_is_flow_thermal(true);

				velocity_buffer_data[_coord_to_id(x, y, z)] = glm::vec4(properties.velocity, 0.0f);
				density_buffer_data[_coord_to_id(x, y, z)] = properties.density;
			}
		}
	}

	_set_is_forcing_scheme(!(is_force_field_constant && constant_force == glm::vec3(0)));

	// compute equilibrium and non-equilibrium populations according to chapter 5.
	velocity_buffer->unmap();
	density_buffer->unmap();

	velocity_buffer_data = nullptr;
	density_buffer_data = nullptr;

	objects_cpu.resize(object_count);

	bool does_contain_boundry = object_count > 1;	// first object slot is indexed by non-boundry id (fluid)

	// bits per boudnry can only be 1, 2, 4, 8 to not cause a boundry spanning over 2 bytes
	_set_bits_per_boundry(std::exp2(std::ceil(std::log2f(std::ceil(std::log2f(object_count))))));
	if (bits_per_boundry > 8) {
		std::cout << "[LBM Error] _initialize_fields_default_pass() is called but too many objects are defined, maximum of 255 bits are possible but number of objets were: " << object_count << std::endl;
		ASSERT(false);
	}
	
	std::cout << "[LBM Info] _initialize_fields_default_pass() is called and " << object_count << " boundry types are defined" << std::endl;
	std::cout << "[LBM Info] _initialize_fields_default_pass() is called and " << bits_per_boundry << " bits per voxel is allocated for boundries" << std::endl;
	
	if (is_forcing_scheme && is_force_field_constant)
		std::cout << "[LBM Info] _initialize_fields_default_pass() is called and the simulation is determined to be a forcing scheme with constant_force_field(" << constant_force.x << ", " << constant_force.y << ", " << constant_force.z << ")" << std::endl;
	if (is_forcing_scheme && !is_force_field_constant)
		std::cout << "[LBM Info] _initialize_fields_default_pass() is called and the simulation is determined to be a forcing scheme with a varying force field" << std::endl;
	if (!is_forcing_scheme)
		std::cout << "[LBM Info] _initialize_fields_default_pass() is called and the simulation is determined to be a non-forcing scheme" << std::endl;

	_compile_shaders();

	out_density_field = density_buffer;
	out_velocity_field = velocity_buffer;

	std::cout << "[LBM Info] _initialize_fields_default_pass() completed" << std::endl;
}

void LBM::_initialize_fields_boundries_pass(std::function<void(glm::ivec3, FluidProperties&)> initialization_lambda)
{
	uint32_t object_count = objects_cpu.size();

	bool does_contain_boundry = object_count > 1;	// first object slot is indexed by non-boundry id (fluid)
	if (!does_contain_boundry) {
		boundries = nullptr;
		objects = nullptr;
		_set_bits_per_boundry(0);
		return;
	}

	// bits per boudnry can only be 1, 2, 4, 8 to not cause a boundry spanning over 2 bytes
	_set_bits_per_boundry(std::exp2(std::ceil(std::log2f(std::ceil(std::log2f(object_count))))));
	if (bits_per_boundry < 1 || bits_per_boundry > 8) {
		std::cout << "[LBM Error] _initialize_fields_boundries_pass() is called but too many or too few objects are defined, maximum of 255 bits are possible but number of objets were: " << object_count << std::endl;
		ASSERT(false);
	}
	_compile_shaders();

	// objects initialization	   a=temperature	   a=scalar			   a=boundry_effective_density
	//							   rgb=trans_vel	   rgb=angular_vel	   rgb=center_of_mass
	size_t object_size_on_device = sizeof(glm::vec4) + sizeof(glm::vec4) + sizeof(glm::vec4);
	objects = std::make_shared<Buffer>(object_size_on_device * object_count);

	objects->map(Buffer::MapInfo(Buffer::MapInfo::Bothways, Buffer::MapInfo::Temporary));
	glm::vec4* objects_mapped_buffer = (glm::vec4*)objects->get_mapped_pointer();

	for (uint32_t i = 0; i < objects_cpu.size(); i++) {
		_object_desc& desc = objects_cpu[i];
		objects_mapped_buffer[3 * i + 0] = glm::vec4(desc.velocity_translational, desc.temperature);
		objects_mapped_buffer[3 * i + 1] = glm::vec4(desc.velocity_angular, 0);
		objects_mapped_buffer[3 * i + 2] = glm::vec4(desc.center_of_mass, desc.effective_density);
	}

	objects->unmap();
	objects_mapped_buffer = nullptr;


	// boundries initialization

	size_t boundries_buffer_size = std::ceil(std::ceil((_get_voxel_count() * bits_per_boundry) / 8.0f) / 4.0f) * 4;
	boundries = std::make_shared<Buffer>(boundries_buffer_size);

	boundries->map();
	uint32_t* boundries_mapped_buffer = (uint32_t*)boundries->get_mapped_pointer();

	for (size_t i = 0; i < boundries_buffer_size / 4; i++) {
		(boundries_mapped_buffer)[i] = 0;
	}

	for (int32_t z = 0; z < resolution.z; z++) {
		for (int32_t y = 0; y < resolution.y; y++) {
			for (int32_t x = 0; x < resolution.x; x++) {
				FluidProperties properties;
				initialization_lambda(glm::ivec3(x, y, z), properties);

				size_t voxel_id = _coord_to_id(x, y, z);
				size_t bits_begin = voxel_id * bits_per_boundry;

				size_t dword_offset = bits_begin / 32;
				int32_t subdword_offset_in_bits = bits_begin % 32;

				(boundries_mapped_buffer)[dword_offset] |= (properties.boundry_id << subdword_offset_in_bits);
			}
		}
	}

	// debug
	for (int32_t z = 0; z < resolution.z; z++) {
		for (int32_t y = 0; y < resolution.y; y++) {
			for (int32_t x = 0; x < resolution.x; x++) {
				FluidProperties properties;
				initialization_lambda(glm::ivec3(x, y, z), properties);

				unsigned int voxel_id = _coord_to_id(x, y, z);
				unsigned int bits_begin = voxel_id * bits_per_boundry;

				unsigned int dword_offset = bits_begin / 32;
				unsigned int subdword_offset_in_bits = bits_begin % 32;

				uint32_t bitmask = (1 << bits_per_boundry) - 1;
				uint32_t boundry = (boundries_mapped_buffer)[dword_offset] & (bitmask << subdword_offset_in_bits);
				boundry = boundry >> subdword_offset_in_bits;

				if (boundry != properties.boundry_id) {
					std::cout << "[LBM Error] _initialize_fields_boundries_pass() is called but an error occured during writing or reading the boundries bits" << std::endl;
					std::cout << "[LBM Error] boundry value read(" << boundry << ") mismatch the value written(" << properties.boundry_id << ")" << " at the coordinates(" << x << ", " << y << ", " << z << ")" << std::endl;
					ASSERT(false);
				}
			}
		}
	}

	boundries->unmap();
	boundries_mapped_buffer = nullptr;

	std::cout << "[LBM Info] _initialize_fields_boundries_pass() completed" << std::endl;
}

void LBM::_initialize_fields_force_pass(std::function<void(glm::ivec3, FluidProperties&)> initialization_lambda) {
	if (is_force_field_constant)
		this->forces = nullptr;
	else {
		forces = std::make_shared<Buffer>(_get_voxel_count() * sizeof(glm::vec4));
		forces->map();

		glm::vec4* forces_buffer_data = (glm::vec4*)forces->get_mapped_pointer();

		for (int32_t z = 0; z < resolution.z; z++) {
			for (int32_t y = 0; y < resolution.y; y++) {
				for (int32_t x = 0; x < resolution.x; x++) {
					FluidProperties properties;
					initialization_lambda(glm::ivec3(x, y, z), properties);
					forces_buffer_data[_coord_to_id(x, y, z)] = glm::vec4(properties.force, 0.0f);
				}
			}
		}

		forces->unmap();
		forces_buffer_data = nullptr;
	}

	std::cout << "[LBM Info] _initialize_fields_force_pass() completed" << std::endl;
}

void LBM::_initialize_fields_thermal_pass(
	std::function<void(glm::ivec3, FluidProperties&)> initialization_lambda,
	std::shared_ptr<Buffer> in_velocity_field
) {
	if (!is_flow_thermal) {
		this->thermal_lattice0 = nullptr;
		this->thermal_lattice1 = nullptr;
	}
	else {

		thermal_lattice_velocity_set_buffer = std::make_unique<UniformBuffer>();
		thermal_lattice_velocity_set_buffer->push_variable_array(get_SimplifiedVelocitySet_vector_count(thermal_lattice_velocity_set)); // a vec4 for every velocity direction

		auto thermal_lattice_velocities_vector = get_velosity_vectors(thermal_lattice_velocity_set);

		thermal_lattice_velocity_set_buffer->set_data(0, 0, thermal_lattice_velocities_vector.size() * sizeof(glm::vec4), thermal_lattice_velocities_vector.data());
		thermal_lattice_velocity_set_buffer->upload_data();

		int32_t lattice_vector_count = get_SimplifiedVelocitySet_vector_count(thermal_lattice_velocity_set);
		thermal_lattice0 = std::make_shared<Buffer>(sizeof(float) * lattice_vector_count * _get_voxel_count());
		thermal_lattice1 = std::make_shared<Buffer>(sizeof(float) * lattice_vector_count * _get_voxel_count());

		Buffer temperature_field(sizeof(float) * _get_voxel_count());
		temperature_field.map();

		float* temperature_field_buffer_data = (float*)temperature_field.get_mapped_pointer();

		for (int32_t z = 0; z < resolution.z; z++) {
			for (int32_t y = 0; y < resolution.y; y++) {
				for (int32_t x = 0; x < resolution.x; x++) {
					FluidProperties properties;
					initialization_lambda(glm::ivec3(x, y, z), properties);
					temperature_field_buffer_data[_coord_to_id(x, y, z)] = properties.temperature;
				}
			}
		}

		temperature_field.unmap();
		temperature_field_buffer_data = nullptr;

		_set_populations_to_equilibrium_thermal(temperature_field, *in_velocity_field);
	}

	std::cout << "[LBM Info] _initialize_fields_themral_pass() completed" << std::endl;
}

//void LBM::copy_to_texture_population(Texture2D& target_texture, int32_t population_index)
//{
//	if (_get_lattice_source() == nullptr) {
//		std::cout << "[LBM Error] LBM::copy_to_texture_population() is called but lattice wasn't generated" << std::endl;
//		ASSERT(false);
//	}
//
//	if ((boundries == nullptr || objects == nullptr) && bits_per_boundry != 0) {
//		std::cout << "[LBM Error] LBM::copy_to_texture_population() is called with boundries activated but boundries buffer wasn't generated" << std::endl;
//		ASSERT(false);
//	}
//
//	if (target_texture.get_internal_format_color() != Texture2D::ColorTextureFormat::R32F) {
//		std::cout << "[LBM Error] LBM::copy_to_texture_population() is called but target_texture's format wasn't compatible" << std::endl;
//		ASSERT(false);
//	}
//
//	if (population_index < 0 || population_index >= get_velocity_set_vector_count()) {
//		std::cout << "[LBM Error] LBM::copy_to_texture_population() is called but population_index is out of bounds" << std::endl;
//		ASSERT(false);
//	}
//
//
//	_compile_shaders();
//
//	ComputeProgram& kernel = *lbm_copy_population;
//
//	Buffer& lattice = *_get_lattice_source();
//
//	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
//	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
//	if (bits_per_boundry != 0) {
//		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
//		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
//	}
//	kernel.update_uniform_as_image("target_texture", target_texture, 0);
//	kernel.update_uniform("lattice_resolution", resolution);
//	kernel.update_uniform("texture_resolution", target_texture.get_size());
//	kernel.update_uniform("population_index", population_index);
//
//	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
//}

void LBM::_generate_lattice_buffer()
{
	size_t voxel_count = _get_voxel_count();
	size_t total_buffer_size_in_bytes =
		voxel_count *
		get_velocity_set_vector_count() *
		get_FloatingPointAccuracy_size_in_bytes(floating_point_accuracy);

	if (is_lattice_texture3d) {
		lattice0_tex = std::make_shared<Texture3D>(
			resolution.x * get_velocity_set_vector_count(),
			resolution.y,
			resolution.z,
			lattice_tex_internal_format,
			1,
			0
		);
		lattice1_tex = std::make_shared<Texture3D>(
			resolution.x * get_velocity_set_vector_count(),
			resolution.y,
			resolution.z,
			lattice_tex_internal_format,
			1,
			0
		);

		//lattice0_tex->clear(0.0f);
		//lattice1_tex->clear(0.0f);
	}
	else {
		lattice0 = std::make_shared<Buffer>(total_buffer_size_in_bytes);
		lattice1 = std::make_shared<Buffer>(total_buffer_size_in_bytes);

		//lattice0->clear(0.0f);
		//lattice1->clear(0.0f);
	}

	
	lattice_velocity_set_buffer = std::make_unique<UniformBuffer>();
	lattice_velocity_set_buffer->push_variable_array(get_velocity_set_vector_count()); // a vec4 for every velocity direction

	auto lattice_velocities_vector = get_velosity_vectors(velocity_set);

	lattice_velocity_set_buffer->set_data(0, 0, lattice_velocities_vector.size() * sizeof(glm::vec4), lattice_velocities_vector.data());
	lattice_velocity_set_buffer->upload_data();
}

void LBM::_set_bits_per_boundry(int32_t value)
{
	if (value == bits_per_boundry)
		return;

	bits_per_boundry = value;
	is_programs_compiled = false;
}

int32_t LBM::_get_bits_per_boundry(int32_t value)
{
	return bits_per_boundry;
}

void LBM::_set_is_flow_thermal(bool value) {
	if (is_flow_thermal == value)
		return;

	is_flow_thermal = value;
	is_programs_compiled = false;
}

bool LBM::_get_is_flow_thermal() {
	return is_flow_thermal;
}

void LBM::_set_thermal_lattice_velocity_set(SimplifiedVelocitySet set) {
	if (thermal_lattice_velocity_set == set)
		return;

	thermal_lattice_velocity_set = set;
	is_programs_compiled = false;
}

SimplifiedVelocitySet LBM::_get_thermal_lattice_velocity_set() {
	return thermal_lattice_velocity_set;
}


std::shared_ptr<Buffer> LBM::_get_lattice_source()
{
	return is_lattice_0_is_source ? lattice0 : lattice1;
}

std::shared_ptr<Buffer> LBM::_get_lattice_target()
{
	return is_lattice_0_is_source ? lattice1 : lattice0;
}

std::shared_ptr<Texture3D> LBM::_get_lattice_tex_source()
{
	return is_lattice_0_is_source ? lattice0_tex : lattice1_tex;
}

std::shared_ptr<Texture3D> LBM::_get_lattice_tex_target()
{
	return is_lattice_0_is_source ? lattice1_tex : lattice0_tex;
}

void LBM::_swap_lattice_buffers()
{
	is_lattice_0_is_source = !is_lattice_0_is_source;
}

std::shared_ptr<Buffer> LBM::_get_thermal_lattice_source()
{
	return is_thermal_lattice_0_is_source ? thermal_lattice0 : thermal_lattice1;
}

std::shared_ptr<Buffer> LBM::_get_thermal_lattice_target()
{
	return is_thermal_lattice_0_is_source ? thermal_lattice1 : thermal_lattice0;
}

void LBM::_swap_thermal_lattice_buffers()
{
	is_thermal_lattice_0_is_source = !is_thermal_lattice_0_is_source;
}

void LBM::_generate_macroscopic_textures()
{
	bool velocity_densty_needs_init = velocity_density_texture == nullptr || velocity_density_texture->get_size() != resolution;
	if (velocity_densty_needs_init) {
		velocity_density_texture = std::make_shared<Texture3D>(
			resolution.x,
			resolution.y,
			resolution.z,
			velocity_density_texture_internal_format,
			1,
			0
		);

		velocity_density_texture->min_filter = Texture3D::SamplingFilter::NEAREST;
		velocity_density_texture->mag_filter = Texture3D::SamplingFilter::NEAREST;
	}

	bool boundry_needs_init = boundry_texture == nullptr || boundry_texture->get_size() != resolution;
	bool boundries_activated = bits_per_boundry != 0;
	if (boundries_activated && boundry_needs_init) {
		boundry_texture = std::make_shared<Texture3D>(
			resolution.x,
			resolution.y,
			resolution.z,
			boundry_texture_internal_format,
			1,
			0
		);

	}

	bool force_temperature_needs_init = force_temperature_texture == nullptr || force_temperature_texture->get_size() != resolution;
	bool force_temperature_activated = is_forcing_scheme || is_flow_thermal || is_flow_multiphase;
	if (force_temperature_activated && force_temperature_needs_init) {
		force_temperature_texture = std::make_shared<Texture3D>(
			resolution.x,
			resolution.y,
			resolution.z,
			force_temperature_texture_internal_format,
			1,
			0
		);

		force_temperature_texture->min_filter = Texture3D::SamplingFilter::NEAREST;
		force_temperature_texture->mag_filter = Texture3D::SamplingFilter::NEAREST;
	}
}

glm::ivec3 LBM::get_resolution()
{
	return resolution;
}

int32_t LBM::get_velocity_set_vector_count()
{
	return get_VelocitySet_vector_count(velocity_set);
}

void LBM::set_boundry_properties(
	uint32_t boundry_id,
	glm::vec3 velocity_translational,
	glm::vec3 velocity_angular,
	glm::vec3 center_of_mass,
	float temperature,
	float effective_density

) {
	if (boundry_id > max_boundry_count) {
		std::cout << "[LBM Error] LBM::set_boundry_properties() is called but boundry_id(" << boundry_id << ") is greater than maximum(" << max_boundry_count << ")" << std::endl;
		ASSERT(false);
	}

	if (boundry_id == 0) {
		std::cout << "[LBM Error] LBM::set_boundry_properties() is called but boundry_id(0) is defined to be fluid, it cannot be treated as an object" << std::endl;
		ASSERT(false);
	}

	if (boundry_id >= objects_cpu.size())
		objects_cpu.resize(boundry_id + 1);
	objects_cpu[boundry_id] = _object_desc(velocity_translational, velocity_angular, center_of_mass, temperature, effective_density);
}

void LBM::set_boundry_properties(
	uint32_t boundry_id,
	glm::vec3 velocity_translational,
	float temperature,
	float effective_density
) {
	set_boundry_properties(
		boundry_id,
		velocity_translational,
		glm::vec3(0),
		glm::vec3(0),
		temperature,
		effective_density
	);
}

void LBM::set_boundry_properties(uint32_t boundry_id, float temperature, float effective_density)
{
	set_boundry_properties(
		boundry_id,
		glm::vec3(0),
		glm::vec3(0),
		glm::vec3(0),
		temperature,
		effective_density
	);
}

void LBM::clear_boundry_properties()
{
	objects_cpu.clear();
}

void LBM::render2d_density()
{
	_compile_shaders();

	if (velocity_density_texture == nullptr)
		return;

	Program& program = *program_render2d_density;

	program.update_uniform("source_texture", *velocity_density_texture);
	program.update_uniform("model", glm::identity<glm::mat4>());
	program.update_uniform("view", glm::identity<glm::mat4>());
	program.update_uniform("projection", glm::identity<glm::mat4>());
	program.update_uniform("texture_resolution", glm::vec3(velocity_density_texture->get_size()));
	program.update_uniform("render_depth", 0);

	glCullFace(GL_BACK);

	primitive_renderer::render(
		program,
		*plane_mesh->get_mesh(0),
		RenderParameters(),
		1,
		0
	);
}

void LBM::render2d_velocity()
{
	_compile_shaders();

	if (velocity_density_texture == nullptr)
		return;

	Program& program = *program_render2d_velocity;

	program.update_uniform("source_texture", *velocity_density_texture);
	program.update_uniform("model", glm::identity<glm::mat4>());
	program.update_uniform("view", glm::identity<glm::mat4>());
	program.update_uniform("projection", glm::identity<glm::mat4>());
	program.update_uniform("texture_resolution", glm::vec3(velocity_density_texture->get_size()));
	program.update_uniform("render_depth", 0);

	glCullFace(GL_BACK);

	primitive_renderer::render(
		program,
		*plane_mesh->get_mesh(0),
		RenderParameters(),
		1,
		0
	);
}

void LBM::render2d_boundries()
{
	_compile_shaders();

	if (boundry_texture == nullptr)
		return;

	Program& program = *program_render2d_boundries;

	program.update_uniform("source_texture", *boundry_texture);
	program.update_uniform("model", glm::identity<glm::mat4>());
	program.update_uniform("view", glm::identity<glm::mat4>());
	program.update_uniform("projection", glm::identity<glm::mat4>());
	program.update_uniform("texture_resolution", glm::vec3(boundry_texture->get_size()));
	program.update_uniform("render_depth", 0);

	glCullFace(GL_BACK);

	primitive_renderer::render(
		program,
		*plane_mesh->get_mesh(0),
		RenderParameters(),
		1,
		0
	);
}

void LBM::render2d_forces()
{
	_compile_shaders();

	if (force_temperature_texture == nullptr)
		return;

	Program& program = *program_render2d_forces;

	program.update_uniform("source_texture", *force_temperature_texture);
	program.update_uniform("model", glm::identity<glm::mat4>());
	program.update_uniform("view", glm::identity<glm::mat4>());
	program.update_uniform("projection", glm::identity<glm::mat4>());
	program.update_uniform("texture_resolution", glm::vec3(force_temperature_texture->get_size()));
	program.update_uniform("render_depth", 0);

	glCullFace(GL_BACK);

	primitive_renderer::render(
		program,
		*plane_mesh->get_mesh(0),
		RenderParameters(),
		1,
		0
	);
}

void LBM::render2d_temperature()
{
	_compile_shaders();

	if (force_temperature_texture == nullptr)
		return;

	Program& program = *program_render2d_temperature;

	program.update_uniform("source_texture", *force_temperature_texture);
	program.update_uniform("model", glm::identity<glm::mat4>());
	program.update_uniform("view", glm::identity<glm::mat4>());
	program.update_uniform("projection", glm::identity<glm::mat4>());
	program.update_uniform("texture_resolution", glm::vec3(force_temperature_texture->get_size()));
	program.update_uniform("render_depth", 0);

	glCullFace(GL_BACK);

	primitive_renderer::render(
		program,
		*plane_mesh->get_mesh(0),
		RenderParameters(),
		1,
		0
	);
}

void LBM::render3d_density(Camera& camera, int32_t sample_count)
{
	_compile_shaders();

	if (velocity_density_texture == nullptr)
		return;

	Program& program = *program_render_volumetric_density;

	camera.update_matrixes();
	camera.update_default_uniforms(program);

	glm::mat4 model = glm::identity<glm::mat4>();
	program.update_uniform("model", model);
	program.update_uniform("inverse_model", glm::inverse(model));
	program.update_uniform("inverse_view", glm::inverse(camera.view_matrix));
	
	program.update_uniform("sample_count", sample_count);
	program.update_uniform("volume", *velocity_density_texture);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	primitive_renderer::render(
		program,
		*plane_cube->get_mesh(0),
		RenderParameters(),
		1,
		0
	);
}

void LBM::render3d_velocity(Camera& camera, int32_t sample_count)
{
	_compile_shaders();

	if (velocity_density_texture == nullptr)
		return;

	Program& program = *program_render_volumetric_velocity;

	camera.update_matrixes();
	camera.update_default_uniforms(program);

	program.update_uniform("model", glm::identity<glm::mat4>());
	program.update_uniform("inverse_model", glm::identity<glm::mat4>());
	program.update_uniform("inverse_view", glm::inverse(camera.view_matrix));

	program.update_uniform("sample_count", sample_count);
	program.update_uniform("volume", *velocity_density_texture);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	primitive_renderer::render(
		program,
		*plane_cube->get_mesh(0),
		RenderParameters(),
		1,
		0
	);
}

void LBM::render3d_boundries(Camera& camera, int32_t sample_count)
{
	_compile_shaders();

	if (boundry_texture == nullptr)
		return;

	Program& program = *program_render_volumetric_boundries;

	camera.update_matrixes();
	camera.update_default_uniforms(program);

	program.update_uniform("model", glm::identity<glm::mat4>());
	program.update_uniform("inverse_model", glm::identity<glm::mat4>());
	program.update_uniform("inverse_view", glm::inverse(camera.view_matrix));

	program.update_uniform("sample_count", sample_count);
	program.update_uniform("volume", *boundry_texture);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	primitive_renderer::render(
		program,
		*plane_cube->get_mesh(0),
		RenderParameters(),
		1,
		0
	);
}

void LBM::render3d_forces(Camera& camera, int32_t sample_count)
{
	_compile_shaders();

	if (force_temperature_texture == nullptr)
		return;

	Program& program = *program_render_volumetric_forces;

	camera.update_matrixes();
	camera.update_default_uniforms(program);

	program.update_uniform("model", glm::identity<glm::mat4>());
	program.update_uniform("inverse_model", glm::identity<glm::mat4>());
	program.update_uniform("inverse_view", glm::inverse(camera.view_matrix));

	program.update_uniform("sample_count", sample_count);
	program.update_uniform("volume", *force_temperature_texture);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	primitive_renderer::render(
		program,
		*plane_cube->get_mesh(0),
		RenderParameters(),
		1,
		0
	);
}

void LBM::render3d_temperature(Camera& camera, int32_t sample_count)
{
	_compile_shaders();

	if (force_temperature_texture == nullptr)
		return;

	Program& program = *program_render_volumetric_temperature;

	camera.update_matrixes();
	camera.update_default_uniforms(program);

	program.update_uniform("model", glm::identity<glm::mat4>());
	program.update_uniform("inverse_model", glm::identity<glm::mat4>());
	program.update_uniform("inverse_view", glm::inverse(camera.view_matrix));

	program.update_uniform("sample_count", sample_count);
	program.update_uniform("volume", *force_temperature_texture);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	primitive_renderer::render(
		program,
		*plane_cube->get_mesh(0),
		RenderParameters(),
		1,
		0
	);
}

//void LBM::set_population(glm::ivec2 voxel_coordinate_begin, glm::ivec2 voxel_coordinate_end, int32_t population_index, float value)
//{
//	_compile_shaders();
//
//	ComputeProgram& kernel = *lbm_set_population;
//
//	Buffer& lattice = *_get_lattice_source();
//
//	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
//	kernel.update_uniform("lattice_resolution", resolution);
//	kernel.update_uniform("lattice_region_begin", voxel_coordinate_begin);
//	kernel.update_uniform("lattice_region_end", voxel_coordinate_end);
//	kernel.update_uniform("population_id", population_index);
//	kernel.update_uniform("value", value);
//
//	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
//}
//
//void LBM::set_population(glm::ivec2 voxel_coordinate, int32_t population_index, float value)
//{
//	set_population(voxel_coordinate, voxel_coordinate + glm::ivec2(1), population_index, value);
//}
//
//void LBM::set_population(int32_t population_index, float value)
//{
//	set_population(glm::ivec2(0), resolution, population_index, value);
//}
//
//void LBM::set_population(float value)
//{
//	for (int index = 0; index < get_velocity_set_vector_count(); index++)
//		set_population(glm::ivec2(0), resolution, index, value);
//}
//
//void LBM::add_random_population(glm::ivec2 voxel_coordinate_begin, glm::ivec2 voxel_coordinate_end, int32_t population_index, float amplitude)
//{
//	_compile_shaders();
//
//	ComputeProgram& kernel = *lbm_add_random_population;
//
//	Buffer& lattice = *_get_lattice_source();
//
//	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
//	kernel.update_uniform("lattice_resolution", resolution);
//	kernel.update_uniform("lattice_region_begin", voxel_coordinate_begin);
//	kernel.update_uniform("lattice_region_end", voxel_coordinate_end);
//	kernel.update_uniform("population_id", population_index);
//	kernel.update_uniform("amplitude", amplitude);
//
//	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
//}
//
//void LBM::add_random_population(glm::ivec2 voxel_coordinate, int32_t population_index, float value) {
//	add_random_population(voxel_coordinate, voxel_coordinate + glm::ivec2(1), population_index, value);
//}
//
//void LBM::add_random_population(int32_t population_index, float amplitude) {
//	add_random_population(glm::ivec2(0), resolution, population_index, amplitude);
//
//}
//
//void LBM::add_random_population(float amplitude) {
//	for (int index = 0; index < get_velocity_set_vector_count(); index++)
//		add_random_population(glm::ivec2(0), resolution, index, amplitude);
//}

std::shared_ptr<Texture3D> LBM::get_velocity_density_texture()
{
	return velocity_density_texture;
}

std::shared_ptr<Texture3D> LBM::get_boundry_texture()
{
	return boundry_texture;
}

std::shared_ptr<Texture3D> LBM::get_force_temperature_texture()
{
	return force_temperature_texture;
}


std::vector<std::pair<std::string, std::string>> LBM::_generate_shader_macros()
{
	std::vector<std::pair<std::string, std::string>> definitions{
		{"floating_point_accuracy", get_FloatingPointAccuracy_to_macro(floating_point_accuracy)},
		{"velocity_set",			get_VelocitySet_to_macro(velocity_set)},
		{"boundry_count",			std::to_string(objects_cpu.size())},
		{"bits_per_boundry",		std::to_string(bits_per_boundry)},
		{"periodic_x",				periodic_x ? "1" : "0"},
		{"periodic_y",				periodic_y ? "1" : "0"},
		{"periodic_z",				periodic_z ? "1" : "0"},
		{"forcing_scheme",			is_forcing_scheme ? "1" : "0"},
		{"constant_force",			is_force_field_constant ? "1" : "0"},
		{"thermal_flow",			is_flow_thermal ? "1" : "0"},
		{"velocity_set_thermal",	get_SimplifiedVelocitySet_to_macro(thermal_lattice_velocity_set)},
		{"multiphase_flow",			is_flow_multiphase ? "1" : "0"},
		{"lattice_is_texutre3d",	is_lattice_texture3d ? "1" : "0"},
		{"esoteric_pull",			is_collide_esoteric ? "1" : "0"},
	};

	return definitions;
}

void LBM::_stream()
{
	_compile_shaders();

	ComputeProgram& kernel = *lbm_stream;

	if (is_lattice_texture3d) {
		Texture3D& lattice_tex_source = *_get_lattice_tex_source();
		Texture3D& lattice_tex_target = *_get_lattice_tex_target();

		kernel.update_uniform_as_image("lattice_source", lattice_tex_source, 0);
		kernel.update_uniform_as_image("lattice_target", lattice_tex_target, 0);
	}
	else {
		Buffer& lattice_source = *_get_lattice_source();
		Buffer& lattice_target = *_get_lattice_target();

		kernel.update_uniform_as_storage_buffer("lattice_buffer_source", lattice_source, 0);
		kernel.update_uniform_as_storage_buffer("lattice_buffer_target", lattice_target, 0);
	}

	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);

	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3.0)));
	kernel.update_uniform("relaxation_time", relaxation_time);
	kernel.update_uniform("lattice_resolution", resolution);

	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}

	if (is_forcing_scheme && !is_force_field_constant)
		kernel.update_uniform_as_storage_buffer("forces_buffer", *forces, 0);
	if (is_forcing_scheme && is_force_field_constant)
		kernel.update_uniform("force_constant", constant_force);

	if (is_flow_thermal) {
		Buffer& thermal_lattice_source = *_get_thermal_lattice_source();
		Buffer& thermal_lattice_target = *_get_thermal_lattice_target();
		kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer_source", thermal_lattice_source, 0);
		kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer_target", thermal_lattice_target, 0);
		kernel.update_uniform_as_uniform_buffer("thermal_velocity_set_buffer", *thermal_lattice_velocity_set_buffer, 0);

		kernel.update_uniform("thermal_lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
		kernel.update_uniform("thermal_relaxation_time", thermal_relaxation_time);
		kernel.update_uniform("thermal_expension_coefficient", thermal_expension_coeficient);
	}

	if (is_flow_multiphase) {
		kernel.update_uniform("intermolecular_interaction_strength", intermolecular_interaction_strength);
	}

	if (is_lattice_texture3d)
		kernel.dispatch_thread(resolution.x * get_velocity_set_vector_count(), resolution.y, resolution.z);
	else 
		kernel.dispatch_thread(_get_voxel_count() * get_velocity_set_vector_count(), 1, 1);

	if (!is_collide_esoteric)
		_swap_lattice_buffers();
}

void LBM::_collide(bool save_macrsoscopic_results)
{
	_compile_shaders();

	ComputeProgram& kernel = save_macrsoscopic_results ? *lbm_collide_save : *lbm_collide;

	if (is_lattice_texture3d) {
		Texture3D& lattice_tex_source = *_get_lattice_tex_source();
		kernel.update_uniform_as_image("lattice_source", lattice_tex_source, 0);
		
		if (!is_collide_esoteric) {
			Texture3D& lattice_tex_target = *_get_lattice_tex_target();
			kernel.update_uniform_as_image("lattice_target", lattice_tex_target, 0);
		}
	}
	else {
		Buffer& lattice_source = *_get_lattice_source();
		kernel.update_uniform_as_storage_buffer("lattice_buffer_source", lattice_source, 0);

		if (!is_collide_esoteric) {
			Buffer& lattice_target = *_get_lattice_target();
			kernel.update_uniform_as_storage_buffer("lattice_buffer_target", lattice_target, 0);
		}
	}

	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);

	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3.0)));
	kernel.update_uniform("relaxation_time", relaxation_time);
	kernel.update_uniform("lattice_resolution", resolution);

	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}

	if (is_forcing_scheme && !is_force_field_constant)
		kernel.update_uniform_as_storage_buffer("forces_buffer", *forces, 0);
	if (is_forcing_scheme && is_force_field_constant)
		kernel.update_uniform("force_constant", constant_force);

	if (is_flow_thermal) {
		Buffer& thermal_lattice_source = *_get_thermal_lattice_source();
		Buffer& thermal_lattice_target = *_get_thermal_lattice_target();
		kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer_source", thermal_lattice_source, 0);
		kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer_target", thermal_lattice_target, 0);
		kernel.update_uniform_as_uniform_buffer("thermal_velocity_set_buffer", *thermal_lattice_velocity_set_buffer, 0);

		kernel.update_uniform("thermal_lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
		kernel.update_uniform("thermal_relaxation_time", thermal_relaxation_time);
		kernel.update_uniform("thermal_expension_coefficient", thermal_expension_coeficient);
	}

	if (is_flow_multiphase) {
		kernel.update_uniform("intermolecular_interaction_strength", intermolecular_interaction_strength);
	}

	if (save_macrsoscopic_results) {
		_generate_macroscopic_textures();

		kernel.update_uniform_as_image("velocity_density_texture", *velocity_density_texture, 0);
		if (bits_per_boundry != 0)
			kernel.update_uniform_as_image("boundry_texture", *boundry_texture, 0);
		if (is_forcing_scheme || is_flow_thermal || is_flow_multiphase)
			kernel.update_uniform_as_image("force_temperature_texture", *force_temperature_texture, 0);
	}

	if (is_lattice_texture3d)
		kernel.dispatch_thread(resolution.x, resolution.y, resolution.z);
	else 
		kernel.dispatch_thread(_get_voxel_count(), 1, 1);

	if (is_collide_esoteric) {
		kernel.update_uniform("is_time_step_odd", (get_total_ticks_elapsed() % 2 == 1) ? 1 : 0);
	}

	if (!is_collide_esoteric)
		_swap_lattice_buffers();

	_swap_thermal_lattice_buffers();
}

void LBM::_collide_with_precomputed_velocities(Buffer& velocity_field)
{
	_compile_shaders();

	ComputeProgram& kernel = *lbm_collide_with_precomputed_velocity;

	Buffer& lattice_source = *_get_lattice_source();
	Buffer& lattice_target = *_get_lattice_target();
	kernel.update_uniform_as_storage_buffer("lattice_buffer_source", lattice_source, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer_target", lattice_target, 0);
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);

	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3.0)));
	kernel.update_uniform("relaxation_time", relaxation_time);
	kernel.update_uniform("lattice_resolution", resolution);

	kernel.update_uniform_as_storage_buffer("velocity_buffer", velocity_field, 0);

	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}

	if (is_forcing_scheme && !is_force_field_constant)
		kernel.update_uniform_as_storage_buffer("forces_buffer", *forces, 0);
	if (is_forcing_scheme && is_force_field_constant)
		kernel.update_uniform("force_constant", constant_force);

	if (is_flow_thermal) {
		Buffer& thermal_lattice_source = *_get_thermal_lattice_source();
		Buffer& thermal_lattice_target = *_get_thermal_lattice_target();
		kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer_source", thermal_lattice_source, 0);
		kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer_target", thermal_lattice_target, 0);
		kernel.update_uniform_as_uniform_buffer("thermal_velocity_set_buffer", *thermal_lattice_velocity_set_buffer, 0);

		kernel.update_uniform("thermal_lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
		kernel.update_uniform("thermal_relaxation_time", thermal_relaxation_time);
		kernel.update_uniform("thermal_expension_coefficient", thermal_expension_coeficient);
	}

	if (is_flow_multiphase) {
		kernel.update_uniform("intermolecular_interaction_strength", intermolecular_interaction_strength);
	}

	kernel.dispatch_thread(_get_voxel_count(), 1, 1);
}

void LBM::_set_populations_to_equilibrium(Buffer& density_field, Buffer& velocity_field)
{
	_compile_shaders();

	ComputeProgram& kernel = *lbm_set_equilibrium_populations;


	if (is_lattice_texture3d) {
		Texture3D& lattice_tex = *_get_lattice_tex_source();
		kernel.update_uniform_as_image("lattice", lattice_tex, 0);
	}
	else {
		Buffer& lattice = *_get_lattice_source();
		kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	}

	kernel.update_uniform_as_storage_buffer("density_buffer", density_field, 0);
	kernel.update_uniform_as_storage_buffer("velocity_buffer", velocity_field, 0);
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
	kernel.update_uniform("relaxation_time", relaxation_time);

	kernel.dispatch_thread(_get_voxel_count(), 1, 1);
}

void LBM::_stream_thermal()
{
	_compile_shaders();

	ComputeProgram& kernel = *lbm_stream_thermal;

	Buffer& lattice_source = *_get_lattice_source();
	Buffer& lattice_target = *_get_lattice_target();
	kernel.update_uniform_as_storage_buffer("lattice_buffer_source", lattice_source, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer_target", lattice_target, 0);
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);

	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3.0)));
	kernel.update_uniform("relaxation_time", relaxation_time);
	kernel.update_uniform("lattice_resolution", resolution);

	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}

	if (is_forcing_scheme && !is_force_field_constant)
		kernel.update_uniform_as_storage_buffer("forces_buffer", *forces, 0);
	if (is_forcing_scheme && is_force_field_constant)
		kernel.update_uniform("force_constant", constant_force);

	if (is_flow_thermal) {
		Buffer& thermal_lattice_source = *_get_thermal_lattice_source();
		Buffer& thermal_lattice_target = *_get_thermal_lattice_target();
		kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer_source", thermal_lattice_source, 0);
		kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer_target", thermal_lattice_target, 0);
		kernel.update_uniform_as_uniform_buffer("thermal_velocity_set_buffer", *thermal_lattice_velocity_set_buffer, 0);

		kernel.update_uniform("thermal_lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
		kernel.update_uniform("thermal_relaxation_time", thermal_relaxation_time);
		kernel.update_uniform("thermal_expension_coefficient", thermal_expension_coeficient);
	}

	if (is_flow_multiphase) {
		kernel.update_uniform("intermolecular_interaction_strength", intermolecular_interaction_strength);
	}

	kernel.dispatch_thread(_get_voxel_count() * get_SimplifiedVelocitySet_vector_count(thermal_lattice_velocity_set), 1, 1);

	_swap_thermal_lattice_buffers();
}

void LBM::_set_populations_to_equilibrium_thermal(Buffer& temperature_field, Buffer& velocity_field)
{
	_compile_shaders();

	ComputeProgram& kernel = *lbm_set_equilibrium_populations_thermal;

	Buffer& lattice_thermal = *_get_thermal_lattice_source();

	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice_thermal, 0);
	kernel.update_uniform_as_storage_buffer("temperature_buffer", temperature_field, 0);
	kernel.update_uniform_as_storage_buffer("velocity_buffer", velocity_field, 0);
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *thermal_lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3))); // is thermal speed of sound heat conductivity?
	kernel.update_uniform("relaxation_time", thermal_relaxation_time);

	kernel.dispatch_thread(_get_voxel_count(), 1, 1);
}

LBM::_object_desc::_object_desc(
	glm::vec3 velocity_translational, 
	glm::vec3 velocity_angular, 
	glm::vec3 center_of_mass, 
	float temperature,
	float effective_density
) : 
	velocity_translational(velocity_translational), 
	velocity_angular(velocity_angular), 
	center_of_mass(center_of_mass), 
	temperature(temperature),
	effective_density(effective_density)
{

}
