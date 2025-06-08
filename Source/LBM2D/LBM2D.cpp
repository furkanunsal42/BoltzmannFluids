#include "LBM2D.h"
#include "Application/ProgramSourcePaths.h"

void LBM2D::compile_shaders()
{
	if (is_programs_compiled)
		return;

	auto definitions = _generate_shader_macros();

	lbm2d_stream							= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "stream.comp"), definitions);
	lbm2d_collide							= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "collide.comp"), definitions);
	lbm2d_boundry_condition					= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "boundry_condition.comp"), definitions);
	lbm2d_collide_with_precomputed_velocity	= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "collide_with_precomputed_velocity.comp"), definitions);
	lbm2d_set_equilibrium_populations		= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "set_equilibrium_populations.comp"), definitions);
	lbm2d_set_population					= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "set_population.comp"), definitions);
	lbm2d_copy_boundries					= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_boundries.comp"), definitions);
	lbm2d_copy_density 						= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_density.comp"), definitions);
	lbm2d_copy_population					= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_population.comp"), definitions);
	lbm2d_copy_velocity_magnitude			= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_velocity_magnitude.comp"), definitions);
	lbm2d_copy_velocity_total				= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_velocity_total.comp"), definitions);
	lbm2d_copy_force_total					= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_force_total.comp"), definitions);				
	lbm2d_add_random_population				= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "add_random_population.comp"), definitions);
	is_programs_compiled = true;

	std::cout << "[LBM Info] kernels are compiled with configuration : " << std::endl;
	for (auto& definition : definitions)
		std::cout << "\t" << definition.first << " : " << definition.second << std::endl;
}

void LBM2D::generate_lattice(glm::ivec2 resolution)
{
	this->resolution = resolution;

	_generate_lattice_buffer();
}

void LBM2D::iterate_time(std::chrono::duration<double, std::milli> deltatime)
{
	deltatime_overflow += deltatime;
	
	for (int32_t i = 0; i < 3; i++) {
		if (deltatime_overflow >= step_deltatime) {
		
			_stream();
			_collide();
			_apply_boundry_conditions();

			deltatime_overflow -= step_deltatime;
			total_time_elapsed += step_deltatime;
		}
	}
}

std::chrono::duration<double, std::milli> LBM2D::get_total_time_elapsed()
{
	return total_time_elapsed;
}

void LBM2D::set_floating_point_accuracy(FloatingPointAccuracy floating_point_accuracy)
{
	if (floating_point_accuracy == this->floating_point_accuracy)
		return;
	
	this->floating_point_accuracy = floating_point_accuracy;
	is_programs_compiled = false;
}

FloatingPointAccuracy LBM2D::get_floating_point_accuracy()
{
	return floating_point_accuracy;
}

void LBM2D::set_velocity_set(VelocitySet velocity_set)
{
	if (velocity_set == this->velocity_set)
		return;

	this->velocity_set = velocity_set;
	is_programs_compiled = false;
}

VelocitySet LBM2D::get_velocity_set()
{
	return velocity_set;
}

void LBM2D::set_relaxation_time(float relaxation_time)
{
	this->relaxation_time = relaxation_time;
}

float LBM2D::get_relaxation_time()
{
	return relaxation_time;
}

void LBM2D::initialize_fields(std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda, glm::ivec2 resolution, float relaxation_time, bool periodic_x, bool periodic_y, VelocitySet velocity_set, FloatingPointAccuracy fp_accuracy)
{
	this->periodic_x = periodic_x;
	this->periodic_y = periodic_y;

	is_programs_compiled = false;
	objects_cpu[0] = _object_desc();

	set_relaxation_time(relaxation_time);
	set_velocity_set(velocity_set);
	set_floating_point_accuracy(fp_accuracy);
	generate_lattice(resolution);

	_initialize_fields_default_pass(
		initialization_lambda,
		resolution,
		fp_accuracy
	);

	_initialize_fields_boundries_pass(
		initialization_lambda,
		resolution,
		fp_accuracy
	);
	
	_initialize_fields_force_pass(
		initialization_lambda,
		resolution,
		fp_accuracy
	);
}

void LBM2D::_initialize_fields_default_pass(std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda, glm::ivec2 resolution, FloatingPointAccuracy fp_accuracy)
{
	Buffer velocity_buffer(resolution.x * resolution.y * sizeof(glm::vec4));
	Buffer density_buffer(resolution.x * resolution.y * sizeof(float));

	velocity_buffer.map(Buffer::MapInfo(Buffer::MapInfo::Upload, Buffer::MapInfo::Temporary));
	density_buffer.map(Buffer::MapInfo(Buffer::MapInfo::Upload, Buffer::MapInfo::Temporary));

	glm::vec4* velocity_buffer_data = (glm::vec4*)velocity_buffer.get_mapped_pointer();
	float* density_buffer_data = (float*)density_buffer.get_mapped_pointer();

	FluidProperties temp_properties;
	initialization_lambda(glm::ivec2(0, 0), temp_properties);

	is_force_field_constant = true;
	constant_force = temp_properties.force;

	uint32_t object_count = 1;

	for (int32_t x = 0; x < resolution.x; x++) {
		for (int32_t y = 0; y < resolution.y; y++) {
			FluidProperties properties;
			initialization_lambda(glm::ivec2(x, y), properties);

			if (properties.boundry_id != 0)
				properties.velocity = glm::vec3(0);

			object_count = std::max(object_count, properties.boundry_id + 1);

			if (properties.force != constant_force)
				is_force_field_constant = false;

			velocity_buffer_data[y * resolution.x + x] = glm::vec4(properties.velocity, 0.0f);
			density_buffer_data[y * resolution.x + x] = properties.density;
		}
	}

	is_forcing_scheme = !(is_force_field_constant && constant_force == glm::vec3(0));

	// compute equilibrium and non-equilibrium populations according to chapter 5.
	velocity_buffer.unmap();
	density_buffer.unmap();

	velocity_buffer_data = nullptr;
	density_buffer_data = nullptr;

	objects_cpu.resize(object_count);

	bool does_contain_boundry = object_count > 1;	// first object slot is indexed by non-boundry id (fluid)

	// bits per boudnry can only be 1, 2, 4, 8 to not cause a boundry spanning over 2 bytes
	bits_per_boundry = std::exp2(std::ceil(std::log2f(std::ceil(std::log2f(object_count)))));
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

	compile_shaders();

	int32_t relaxation_iteration_count = 1;
	std::cout << "[LBM Info] _initialize_fields_default_pass() initialization of particle population distributions from given veloicty and density fields is initiated" << std::endl;

	_set_populations_to_equilibrium(density_buffer, velocity_buffer);

	for (int32_t i = 0; i < relaxation_iteration_count; i++) {
		_collide_with_precomputed_velocities(velocity_buffer);
		//_apply_boundry_conditions(); //the book doesn't specitfy whether or not to enforce boundry conditions in initialization algorithm
		_stream();
	}

	std::cout << "[LBM Info] _initialize_fields_default_pass() fields initialization scheme completed with relaxation_iteration_count(" << relaxation_iteration_count << ")" << std::endl;
	std::cout << "[LBM Info] _initialize_fields_default_pass() completed" << std::endl;
}

void LBM2D::_initialize_fields_boundries_pass(std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda, glm::ivec2 resolution, FloatingPointAccuracy fp_accuracy)
{
	uint32_t object_count = objects_cpu.size();

	bool does_contain_boundry = object_count > 1;	// first object slot is indexed by non-boundry id (fluid)
	if (!does_contain_boundry) {
		boundries = nullptr;
		objects = nullptr;
		bits_per_boundry = 0;
		return;
	}

	// bits per boudnry can only be 1, 2, 4, 8 to not cause a boundry spanning over 2 bytes
	bits_per_boundry = std::exp2(std::ceil(std::log2f(std::ceil(std::log2f(object_count)))));
	if (bits_per_boundry < 1 || bits_per_boundry > 8) {
		std::cout << "[LBM Error] _initialize_fields_boundries_pass() is called but too many or too few objects are defined, maximum of 255 bits are possible but number of objets were: " << object_count << std::endl;
		ASSERT(false);
	}
	compile_shaders();

	
	// objects initialization
	//							   trans_vel		   angular_vel		   center_of_mass
	size_t object_size_on_device = sizeof(glm::vec4) + sizeof(glm::vec4) + sizeof(glm::vec4);
	objects = std::make_shared<Buffer>(object_size_on_device * object_count);

	objects->map(Buffer::MapInfo(Buffer::MapInfo::Bothways, Buffer::MapInfo::Temporary));
	glm::vec4* objects_mapped_buffer = (glm::vec4*)objects->get_mapped_pointer();
	
	for (uint32_t i = 0; i < objects_cpu.size(); i++) {
		_object_desc& desc = objects_cpu[i];
		objects_mapped_buffer[3*i + 0] = glm::vec4(desc.velocity_translational, 0);
		objects_mapped_buffer[3*i + 1] = glm::vec4(desc.velocity_angular, 0);
		objects_mapped_buffer[3*i + 2] = glm::vec4(desc.center_of_mass, 0);
	}

	objects->unmap();
	objects_mapped_buffer = nullptr;

	
	// boundries initialization
	
	size_t boundries_buffer_size = std::ceil(std::ceil((bits_per_boundry * resolution.x * resolution.y) / 8.0f) / 4.0f) * 4;
	boundries = std::make_shared<Buffer>(boundries_buffer_size);

	boundries->map();
	uint32_t* boundries_mapped_buffer = (uint32_t*)boundries->get_mapped_pointer();

	for (size_t i = 0; i < boundries_buffer_size / 4; i++) {
		(boundries_mapped_buffer)[i] = 0;
	}

	for (int32_t x = 0; x < resolution.x; x++) {
		for (int32_t y = 0; y < resolution.y; y++) {
			FluidProperties properties;
			initialization_lambda(glm::ivec2(x, y), properties);
			
			size_t voxel_id = y * resolution.x + x;
			size_t bits_begin = voxel_id * bits_per_boundry;
			
			size_t dword_offset = bits_begin / 32;
			int32_t subdword_offset_in_bits = bits_begin % 32;
			
			(boundries_mapped_buffer)[dword_offset] |= (properties.boundry_id << subdword_offset_in_bits);
		}
	}

	// debug
	for (int32_t x = 0; x < resolution.x; x++) {
		for (int32_t y = 0; y < resolution.y; y++) {
			FluidProperties properties;
			initialization_lambda(glm::ivec2(x, y), properties);
			
			int voxel_id = y * resolution.x + x;
			int bits_begin = voxel_id * bits_per_boundry;
	
			int dword_offset = bits_begin / 32;
			int subdword_offset_in_bits = bits_begin % 32;
	
			uint32_t bitmask = (1 << bits_per_boundry) - 1;
			uint32_t boundry = (boundries_mapped_buffer)[dword_offset] & (bitmask << subdword_offset_in_bits);
			boundry = boundry >> subdword_offset_in_bits;
	 
			if (boundry != properties.boundry_id) {
				std::cout << "[LBM Error] _initialize_fields_boundries_pass() is called but an error occured during writing or reading the boundries bits" << std::endl;
				std::cout << "[LBM Error] boundry value read(" << boundry << ") mismatch the value written(" << properties.boundry_id << ")" << " at the coordinates(" << x << ", " << y << ")" << std::endl;
				ASSERT(false);
			}
		}
	}

	boundries->unmap();
	boundries_mapped_buffer = nullptr;

	std::cout << "[LBM Info] _initialize_fields_boundries_pass() completed" << std::endl;
}

void LBM2D::_initialize_fields_force_pass(std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda, glm::ivec2 resolution, FloatingPointAccuracy fp_accuracy)
{
	if (is_force_field_constant)
		this->forces = nullptr;
	else {
		forces = std::make_shared<Buffer>(sizeof(glm::vec4) * resolution.x * resolution.y);
		forces->map();
		
		glm::vec4* forces_buffer_data = (glm::vec4*)forces->get_mapped_pointer();

		for (int32_t x = 0; x < resolution.x; x++) {
			for (int32_t y = 0; y < resolution.y; y++) {
				FluidProperties properties;
				initialization_lambda(glm::ivec2(x, y), properties);
				forces_buffer_data[y*resolution.x + x] = glm::vec4(properties.force, 0.0f);
			}
		}

		forces->unmap();
		forces_buffer_data = nullptr;
	}

	std::cout << "[LBM Info] _initialize_fields_force_pass() completed" << std::endl;
}

void LBM2D::copy_to_texture_population(Texture2D& target_texture, int32_t population_index)
{
	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_population() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (boundries == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_population() is called but boundries wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (target_texture.get_internal_format_color() != Texture2D::ColorTextureFormat::R32F) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_population() is called but target_texture's format wasn't compatible" << std::endl;
		ASSERT(false);
	}

	if (population_index < 0 || population_index >= get_velocity_set_vector_count()) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_population() is called but population_index is out of bounds" << std::endl;
		ASSERT(false);
	}


	compile_shaders();

	ComputeProgram& kernel = *lbm2d_copy_population;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
	kernel.update_uniform_as_image("target_texture", target_texture, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("texture_resolution", target_texture.get_size());
	kernel.update_uniform("population_index", population_index);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::copy_to_texture_density(Texture2D& target_texture)
{
	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_density() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (boundries == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_density() is called but boundries wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (target_texture.get_internal_format_color() != Texture2D::ColorTextureFormat::R32F) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_density() is called but target_texture's format wasn't compatible" << std::endl;
		ASSERT(false);
	}

	compile_shaders();

	ComputeProgram& kernel = *lbm2d_copy_density;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
	kernel.update_uniform_as_image("target_texture", target_texture, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("texture_resolution", target_texture.get_size());

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::copy_to_texture_velocity_vector(Texture2D& target_texture)
{
	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_vector() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (boundries == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_vector() is called but boundries wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (target_texture.get_internal_format_color() != Texture2D::ColorTextureFormat::RGBA32F) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_vector() is called but target_texture's format wasn't compatible" << std::endl;
		ASSERT(false);
	}

	compile_shaders();

	ComputeProgram& kernel = *lbm2d_copy_velocity_total;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
	kernel.update_uniform_as_image("target_texture", target_texture, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("texture_resolution", target_texture.get_size());

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::copy_to_texture_velocity_magnetude(Texture2D& target_texture) {
	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_magnetude() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (boundries == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_magnetude() is called but boundries wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (target_texture.get_internal_format_color() != Texture2D::ColorTextureFormat::R32F) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_magnetude() is called but target_texture's format wasn't compatible" << std::endl;
		ASSERT(false);
	}

	compile_shaders();

	ComputeProgram& kernel = *lbm2d_copy_velocity_magnitude;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
	kernel.update_uniform_as_image("target_texture", target_texture, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("texture_resolution", target_texture.get_size());

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::copy_to_texture_boundries(Texture2D& target_texture)
{
	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_boundries() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (boundries == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_boundries() is called but boundries wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (target_texture.get_internal_format_color() != Texture2D::ColorTextureFormat::RGBA32F) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_boundries() is called but target_texture's format wasn't compatible" << std::endl;
		ASSERT(false);
	}

	compile_shaders();

	ComputeProgram& kernel = *lbm2d_copy_boundries;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
	kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	kernel.update_uniform_as_image("target_texture", target_texture, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("texture_resolution", target_texture.get_size());

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::copy_to_texture_force_vector(Texture2D& target_texture)
{
	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_force_vector() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (forces == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_force_vector() is called but boundries wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (target_texture.get_internal_format_color() != Texture2D::ColorTextureFormat::RGBA32F) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_force_vector() is called but target_texture's format wasn't compatible" << std::endl;
		ASSERT(false);
	}

	compile_shaders();

	ComputeProgram& kernel = *lbm2d_copy_force_total;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
	kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	kernel.update_uniform_as_storage_buffer("forces_buffer", *forces, 0);
	kernel.update_uniform_as_image("target_texture", target_texture, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("texture_resolution", target_texture.get_size());

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::_generate_lattice_buffer()
{
	size_t voxel_count = resolution.x * resolution.y;
	size_t total_buffer_size_in_bytes = 
		voxel_count * 
		get_velocity_set_vector_count() * 
		get_FloatingPointAccuracy_size_in_bytes(floating_point_accuracy);

	size_t boundry_buffer_size = (size_t)std::ceil(voxel_count / 8.0);

	lattice0 = std::make_shared<Buffer>(total_buffer_size_in_bytes);
	lattice1 = std::make_shared<Buffer>(total_buffer_size_in_bytes);
	boundries = std::make_shared<Buffer>(boundry_buffer_size);

	lattice0->clear(0.0f);
	lattice1->clear(0.0f);
	boundries->clear(0);

	lattice_velocity_set_buffer = std::make_unique<UniformBuffer>();
	lattice_velocity_set_buffer->push_variable_array(get_velocity_set_vector_count()); // a vec4 for every velocity direction
	
	auto lattice_velocities_vector = get_velosity_vectors(velocity_set);
	
	lattice_velocity_set_buffer->set_data(0, 0, lattice_velocities_vector.size() * sizeof(glm::vec4), lattice_velocities_vector.data());
	lattice_velocity_set_buffer->upload_data();
}

std::shared_ptr<Buffer> LBM2D::_get_lattice_source()
{
	return is_lattice_0_is_source ? lattice0 : lattice1;
}

std::shared_ptr<Buffer> LBM2D::_get_lattice_target()
{
	return is_lattice_0_is_source ? lattice1 : lattice0;
}

void LBM2D::_swap_lattice_buffers()
{
	is_lattice_0_is_source = !is_lattice_0_is_source;
}

glm::ivec2 LBM2D::get_resolution()
{
	return resolution;
}

int32_t LBM2D::get_velocity_set_vector_count()
{
	return get_VelocitySet_vector_count(velocity_set);
}

void LBM2D::set_boundry_velocity(uint32_t boundry_id, glm::vec3 velocity_translational, glm::vec3 velocity_angular, glm::vec3 center_of_mass)
{
	if (boundry_id > max_boundry_count){
		std::cout << "[LBM Error] LBM2D::set_boundry_velocity() is called but boundry_id(" << boundry_id << ") is greater than maximum(" << max_boundry_count << ")" << std::endl;
		ASSERT(false);
	}

	if (boundry_id == 0) {
		std::cout << "[LBM Error] LBM2D::set_boundry_velocity() is called but boundry_id(0) is defined to be fluid, it cannot be treated as an object" << std::endl;
		ASSERT(false);
	}

	if (boundry_id >= objects_cpu.size())
		objects_cpu.resize(boundry_id + 1);
	objects_cpu[boundry_id] = _object_desc(velocity_translational, velocity_angular, center_of_mass);
}

void LBM2D::set_boundry_velocity(uint32_t boundry_id, glm::vec3 velocity_translational)
{
	set_boundry_velocity(boundry_id, velocity_translational, glm::vec3(0), glm::vec3(0));
}

void LBM2D::set_population(glm::ivec2 voxel_coordinate_begin, glm::ivec2 voxel_coordinate_end, int32_t population_index, float value)
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_set_population;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("lattice_region_begin", voxel_coordinate_begin);
	kernel.update_uniform("lattice_region_end", voxel_coordinate_end);
	kernel.update_uniform("population_id", population_index);
	kernel.update_uniform("value", value);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::set_population(glm::ivec2 voxel_coordinate, int32_t population_index, float value)
{
	set_population(voxel_coordinate, voxel_coordinate + glm::ivec2(1), population_index, value);
}

void LBM2D::set_population(int32_t population_index, float value)
{
	set_population(glm::ivec2(0), resolution, population_index, value);
}

void LBM2D::set_population(float value)
{
	for (int index = 0; index < get_velocity_set_vector_count(); index++)
		set_population(glm::ivec2(0), resolution, index, value);
}

void LBM2D::add_random_population(glm::ivec2 voxel_coordinate_begin, glm::ivec2 voxel_coordinate_end, int32_t population_index, float amplitude)
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_add_random_population;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("lattice_region_begin", voxel_coordinate_begin);
	kernel.update_uniform("lattice_region_end", voxel_coordinate_end);
	kernel.update_uniform("population_id", population_index);
	kernel.update_uniform("amplitude", amplitude);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::add_random_population(glm::ivec2 voxel_coordinate, int32_t population_index, float value) {
	add_random_population(voxel_coordinate, voxel_coordinate + glm::ivec2(1), population_index, value);
}

void LBM2D::add_random_population(int32_t population_index, float amplitude) {
	add_random_population(glm::ivec2(0), resolution, population_index, amplitude);

}

void LBM2D::add_random_population(float amplitude) {
	for (int index = 0; index < get_velocity_set_vector_count(); index++)
		add_random_population(glm::ivec2(0), resolution, index, amplitude);
}


std::vector<std::pair<std::string, std::string>> LBM2D::_generate_shader_macros()
{
	std::vector<std::pair<std::string, std::string>> definitions{
		{"floating_point_accuracy", get_FloatingPointAccuracy_to_macro(floating_point_accuracy)},
		{"velocity_set", get_VelocitySet_to_macro(velocity_set)},
		{"boundry_count", std::to_string(objects_cpu.size())},
		{"bits_per_boundry", std::to_string(bits_per_boundry)},
		{"periodic_x", periodic_x ? "1" : "0"},
		{"periodic_y", periodic_y ? "1" : "0"},
		{"forcing_scheme", is_forcing_scheme ? "1" : "0"},
		{"constant_force", is_force_field_constant ? "1" : "0"},
	};

	return definitions;
}

void LBM2D::_stream()
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_stream;

	Buffer& lattice_source = *_get_lattice_source();
	Buffer& lattice_target = *_get_lattice_target();

	// add forces implementation
	kernel.update_uniform_as_storage_buffer("lattice_buffer_source", lattice_source, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer_target", lattice_target, 0);
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);

	kernel.dispatch_thread(resolution.x * resolution.y * get_velocity_set_vector_count(), 1, 1);

	_swap_lattice_buffers();
}

void LBM2D::_collide()
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_collide;

	Buffer& lattice_source = *_get_lattice_source();
	Buffer& lattice_target = *_get_lattice_target();

	kernel.update_uniform_as_storage_buffer("lattice_buffer_source", lattice_source, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer_target", lattice_target, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
	kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);

	if (is_forcing_scheme && !is_force_field_constant)
		kernel.update_uniform_as_storage_buffer("forces_buffer", *forces, 0);
	if (is_forcing_scheme && is_force_field_constant)
		kernel.update_uniform("force_constant", constant_force);
	
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
	kernel.update_uniform("relaxation_time", relaxation_time);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);

	_swap_lattice_buffers();
}

void LBM2D::_apply_boundry_conditions() {

	if (bits_per_boundry == 0)
		return;

	compile_shaders();
	
	if (objects == nullptr || boundries == nullptr) {
		std::cout << "[LBM2D Error] LBM2D::_apply_boundry_conditions() is called and bits_per_boundry is non-zero but objects or boundries is nullptr" << std::endl;
		ASSERT(false);
	}
	
	ComputeProgram& kernel = *lbm2d_boundry_condition;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
	kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::_collide_with_precomputed_velocities(Buffer& velocity_field)
{
	ComputeProgram& kernel = *lbm2d_collide_with_precomputed_velocity;

}

void LBM2D::_set_populations_to_equilibrium(Buffer& density_field, Buffer& velocity_field)
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_set_equilibrium_populations;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform_as_storage_buffer("density_buffer", density_field, 0);
	kernel.update_uniform_as_storage_buffer("velocity_buffer", velocity_field, 0);
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	//kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
	//kernel.update_uniform("relaxation_time", relaxation_time);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

LBM2D::_object_desc::_object_desc(glm::vec3 velocity_translational, glm::vec3 velocity_angular, glm::vec3 center_of_mass) :
	velocity_translational(velocity_translational), velocity_angular(velocity_angular), center_of_mass(center_of_mass)
{

}
