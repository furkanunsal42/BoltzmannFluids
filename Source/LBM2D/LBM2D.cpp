#include "LBM2D.h"
#include "Application/ProgramSourcePaths.h"

void LBM2D::compile_shaders()
{
	if (is_programs_compiled)
		return;

	auto definitions = _generate_shader_macros();

	lbm2d_stream								= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "stream.comp"), definitions);
	lbm2d_stream_thermal						= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "stream_thermal.comp"), definitions);
	lbm2d_collide								= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "collide.comp"), definitions);
	lbm2d_collide_thermal						= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "collide_thermal.comp"), definitions);
	lbm2d_boundry_condition						= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "boundry_condition.comp"), definitions);
	lbm2d_boundry_condition_thermal				= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "boundry_condition_thermal.comp"), definitions);
	lbm2d_collide_with_precomputed_velocity		= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "collide_with_precomputed_velocity.comp"), definitions);
	lbm2d_set_equilibrium_populations			= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "set_equilibrium_populations.comp"), definitions);
	lbm2d_set_equilibrium_populations_thermal	= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "set_equilibrium_populations_thermal.comp"), definitions);
	lbm2d_set_population						= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "set_population.comp"), definitions);
	lbm2d_copy_boundries						= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_boundries.comp"), definitions);
	lbm2d_copy_density 							= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_density.comp"), definitions);
	lbm2d_copy_population						= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_population.comp"), definitions);
	lbm2d_copy_velocity_magnitude				= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_velocity_magnitude.comp"), definitions);
	lbm2d_copy_velocity_total					= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_velocity_total.comp"), definitions);
	lbm2d_copy_force_total						= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_force_total.comp"), definitions);				
	lbm2d_add_random_population					= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "add_random_population.comp"), definitions);
	lbm2d_copy_temperature						= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_temperature.comp"), definitions);
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
		
			if (is_flow_thermal) {
				_stream_thermal();
				_collide_thermal();
				_apply_boundry_conditions_thermal();
			}

			_collide();
			_stream();
			//_apply_boundry_conditions();

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

void LBM2D::set_periodic_boundry_x(bool value)
{
	if (periodic_x == value)
		return;

	periodic_x = value;
	is_programs_compiled = false;
}

bool LBM2D::get_periodic_boundry_x()
{
	return periodic_x;
}

void LBM2D::set_periodic_boundry_y(bool value)
{
	if (periodic_y == value)
		return;

	periodic_y = value;
	is_programs_compiled = false;
}

bool LBM2D::get_periodic_boundry_y()
{
	return periodic_y;
}

void LBM2D::set_is_forcing_scheme(bool value)
{
	if (is_forcing_scheme == value)
		return;

	is_forcing_scheme = value;
	is_programs_compiled = false;
}

bool LBM2D::get_is_forcing_scheme()
{
	return is_forcing_scheme;
}

void LBM2D::set_is_force_field_constant(bool value)
{
	if (is_force_field_constant == value)
		return;

	is_force_field_constant = value;
	is_programs_compiled = false;
}

bool LBM2D::get_is_force_field_constant()
{
	return is_force_field_constant;
}

void LBM2D::set_constant_force(glm::vec3 constant_force)
{
	this->constant_force = constant_force;
}

glm::vec3 LBM2D::get_constant_force()
{
	return constant_force;
}

void LBM2D::set_is_flow_multiphase(bool value)
{
	if (is_flow_multiphase == value)
		return;

	is_flow_multiphase = value;
	is_programs_compiled = false;
}

bool LBM2D::get_is_flow_multiphase()
{
	return is_flow_multiphase;
}

void LBM2D::initialize_fields(
	std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda, 
	glm::ivec2 resolution, 
	float relaxation_time, 
	bool periodic_x, 
	bool periodic_y, 
	VelocitySet velocity_set, 
	FloatingPointAccuracy fp_accuracy,
	bool is_flow_multiphase
) {
	is_programs_compiled = false;

	if (objects_cpu.size() == 0)
		objects_cpu.resize(1);
	objects_cpu[0] = _object_desc();

	set_relaxation_time(relaxation_time);
	set_velocity_set(velocity_set);
	set_floating_point_accuracy(fp_accuracy);
	set_periodic_boundry_x(periodic_x);
	set_periodic_boundry_y(periodic_y);
	_set_thermal_lattice_velocity_set(
		get_VelocitySet_dimention(velocity_set) == 2 ? 
		SimplifiedVelocitySet::D2Q5 : SimplifiedVelocitySet::D3Q7
	);
	set_is_flow_multiphase(is_flow_multiphase);
	
	generate_lattice(resolution);

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

	std::cout << "[LBM Info] _initialize_fields_default_pass() fields initialization scheme completed with relaxation_iteration_count(" << relaxation_iteration_count << ")" << std::endl;

}

void LBM2D::_initialize_fields_default_pass(
	std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda,
	std::shared_ptr<Buffer>& out_density_field,
	std::shared_ptr<Buffer>& out_velocity_field
) {

	std::shared_ptr<Buffer> velocity_buffer = std::make_shared<Buffer>(resolution.x * resolution.y * sizeof(glm::vec4));
	std::shared_ptr<Buffer> density_buffer = std::make_shared<Buffer>(resolution.x * resolution.y * sizeof(float));

	velocity_buffer->map(Buffer::MapInfo(Buffer::MapInfo::Upload, Buffer::MapInfo::Temporary));
	density_buffer->map(Buffer::MapInfo(Buffer::MapInfo::Upload, Buffer::MapInfo::Temporary));

	glm::vec4* velocity_buffer_data = (glm::vec4*)velocity_buffer->get_mapped_pointer();
	float* density_buffer_data = (float*)density_buffer->get_mapped_pointer();

	FluidProperties temp_properties;
	initialization_lambda(glm::ivec2(0, 0), temp_properties);

	set_is_force_field_constant(true);
	set_constant_force(temp_properties.force);

	_set_is_flow_thermal(false);
	
	uint32_t object_count = 1;

	for (int32_t x = 0; x < resolution.x; x++) {
		for (int32_t y = 0; y < resolution.y; y++) {
			FluidProperties properties;
			initialization_lambda(glm::ivec2(x, y), properties);

			if (properties.boundry_id != 0)
				properties.velocity = glm::vec3(0);

			object_count = std::max(object_count, properties.boundry_id + 1);

			if (properties.force != constant_force)
				set_is_force_field_constant(false);
			if (properties.temperature != temp_properties.temperature)
				_set_is_flow_thermal(true);
			if (properties.boundry_id != not_a_boundry && objects_cpu[properties.boundry_id].temperature != temp_properties.temperature)
				_set_is_flow_thermal(true);
			
			velocity_buffer_data[y * resolution.x + x] = glm::vec4(properties.velocity, 0.0f);
			density_buffer_data[y * resolution.x + x] = properties.density;
		}
	}

	set_is_forcing_scheme(!(is_force_field_constant && constant_force == glm::vec3(0)));

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

	compile_shaders();

	out_density_field = density_buffer;
	out_velocity_field = velocity_buffer;

	std::cout << "[LBM Info] _initialize_fields_default_pass() completed" << std::endl;
}

void LBM2D::_initialize_fields_boundries_pass(std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda)
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
	compile_shaders();

	// objects initialization	   a=temperature	   a=scalar			   a=boundry_effective_density
	//							   rgb=trans_vel	   rgb=angular_vel	   rgb=center_of_mass
	size_t object_size_on_device = sizeof(glm::vec4) + sizeof(glm::vec4) + sizeof(glm::vec4);
	objects = std::make_shared<Buffer>(object_size_on_device * object_count);

	objects->map(Buffer::MapInfo(Buffer::MapInfo::Bothways, Buffer::MapInfo::Temporary));
	glm::vec4* objects_mapped_buffer = (glm::vec4*)objects->get_mapped_pointer();
	
	for (uint32_t i = 0; i < objects_cpu.size(); i++) {
		_object_desc& desc = objects_cpu[i];
		objects_mapped_buffer[3*i + 0] = glm::vec4(desc.velocity_translational, desc.temperature);
		objects_mapped_buffer[3*i + 1] = glm::vec4(desc.velocity_angular, 0);
		objects_mapped_buffer[3*i + 2] = glm::vec4(desc.center_of_mass, desc.effective_density);
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

void LBM2D::_initialize_fields_force_pass(std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda) {
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

void LBM2D::_initialize_fields_thermal_pass(
	std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda,
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
		thermal_lattice0 = std::make_shared<Buffer>(sizeof(float) * lattice_vector_count * resolution.x * resolution.y);
		thermal_lattice1 = std::make_shared<Buffer>(sizeof(float) * lattice_vector_count * resolution.x * resolution.y);
		
		Buffer temperature_field(sizeof(float) * resolution.x * resolution.y);
		temperature_field.map();
		
		float* temperature_field_buffer_data = (float*)temperature_field.get_mapped_pointer();

		for (int32_t x = 0; x < resolution.x; x++) {
			for (int32_t y = 0; y < resolution.y; y++) {
				FluidProperties properties;
				initialization_lambda(glm::ivec2(x, y), properties);
				temperature_field_buffer_data[y * resolution.x + x] = properties.temperature;
			}
		}

		temperature_field.unmap();
		temperature_field_buffer_data = nullptr;

		// INIT THERMAL_LATTICE
		_set_populations_to_equilibrium_thermal(temperature_field, *in_velocity_field);
	}

	std::cout << "[LBM Info] _initialize_fields_themral_pass() completed" << std::endl;
}

void LBM2D::copy_to_texture_population(Texture2D& target_texture, int32_t population_index)
{
	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_population() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if ((boundries == nullptr || objects == nullptr) && bits_per_boundry != 0) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_population() is called with boundries activated but boundries buffer wasn't generated" << std::endl;
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
	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}
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

	if ((boundries == nullptr || objects == nullptr) && bits_per_boundry != 0) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_density() is called with boundries activated but boundries buffer wasn't generated" << std::endl;
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
	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}
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

	if ((boundries == nullptr || objects == nullptr) && bits_per_boundry != 0) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_vector() is called with boundries activated but boundries buffer wasn't generated" << std::endl;
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
	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}
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

	if ((boundries == nullptr || objects == nullptr) && bits_per_boundry != 0) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_magnetude() is called with boundries activated but boundries buffer wasn't generated" << std::endl;
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
	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}
	kernel.update_uniform_as_image("target_texture", target_texture, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("texture_resolution", target_texture.get_size());

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::copy_to_texture_boundries(Texture2D& target_texture)
{
	if (bits_per_boundry == 0) {
		target_texture.clear(glm::vec4(0));
		return;
	}

	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_boundries() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if ((boundries == nullptr || objects == nullptr) && bits_per_boundry != 0) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_boundries() is called with boundries activated but boundries buffer wasn't generated" << std::endl;
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
	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}
	kernel.update_uniform_as_image("target_texture", target_texture, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("texture_resolution", target_texture.get_size());

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::copy_to_texture_force_vector(Texture2D& target_texture)
{
	if (!is_forcing_scheme) {
		target_texture.clear(glm::vec4(0));
		return;
	}
	
	if (is_forcing_scheme && is_force_field_constant) {
		target_texture.clear(glm::vec4(constant_force, 1));
		return;
	}

	if (is_forcing_scheme && !is_force_field_constant && forces == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_force_vector() is called but forces wasn't generated" << std::endl;
		ASSERT(false);
	}

	if ((boundries == nullptr || objects == nullptr) && bits_per_boundry != 0) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_force_vector() is called with boundries activated but boundries buffer wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_force_vector() is called but lattice wasn't generated" << std::endl;
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
	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}
	kernel.update_uniform_as_storage_buffer("forces_buffer", *forces, 0);
	kernel.update_uniform_as_image("target_texture", target_texture, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("texture_resolution", target_texture.get_size());

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::copy_to_texture_temperature(Texture2D& target_texture)
{
	if (!is_flow_thermal) {
		target_texture.clear(glm::vec4(referance_temperature));
		return;
	}

	if (_get_thermal_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_temperature() is called but thermal_lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if ((boundries == nullptr || objects == nullptr) && bits_per_boundry) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_temperature() is called but boundries wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (target_texture.get_internal_format_color() != Texture2D::ColorTextureFormat::R32F) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_temperature() is called but target_texture's format wasn't compatible" << std::endl;
		ASSERT(false);
	}

	compile_shaders();

	ComputeProgram& kernel = *lbm2d_copy_temperature;

	Buffer& thermal_lattice = *_get_thermal_lattice_source();

	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer", thermal_lattice, 0);
	if (bits_per_boundry != 0) {
		kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
		kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	}
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

void LBM2D::_set_bits_per_boundry(int32_t value)
{
	if (value == bits_per_boundry)
		return;

	bits_per_boundry = value;
	is_programs_compiled = false;
}

int32_t LBM2D::_get_bits_per_boundry(int32_t value)
{
	return bits_per_boundry;
}

void LBM2D::_set_is_flow_thermal(bool value){
	if (is_flow_thermal == value)
		return;

	is_flow_thermal = value;
	is_programs_compiled = false;
}

bool LBM2D::_get_is_flow_thermal(){
	return is_flow_thermal;
}

void LBM2D::_set_thermal_lattice_velocity_set(SimplifiedVelocitySet set){
	if (thermal_lattice_velocity_set == set)
		return;

	thermal_lattice_velocity_set = set;
	is_programs_compiled = false;
}

SimplifiedVelocitySet LBM2D::_get_thermal_lattice_velocity_set(){
	return thermal_lattice_velocity_set;
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

std::shared_ptr<Buffer> LBM2D::_get_thermal_lattice_source()
{
	return is_thermal_lattice_0_is_source ? thermal_lattice0 : thermal_lattice1;
}

std::shared_ptr<Buffer> LBM2D::_get_thermal_lattice_target()
{
	return is_thermal_lattice_0_is_source ? thermal_lattice1 : thermal_lattice0;
}

void LBM2D::_swap_thermal_lattice_buffers()
{
	is_thermal_lattice_0_is_source = !is_thermal_lattice_0_is_source;
}

glm::ivec2 LBM2D::get_resolution()
{
	return resolution;
}

int32_t LBM2D::get_velocity_set_vector_count()
{
	return get_VelocitySet_vector_count(velocity_set);
}

void LBM2D::set_boundry_properties(
	uint32_t boundry_id, 
	glm::vec3 velocity_translational, 
	glm::vec3 velocity_angular, 
	glm::vec3 center_of_mass, 
	float temperature,
	float effective_density

){
	if (boundry_id > max_boundry_count) {
		std::cout << "[LBM Error] LBM2D::set_boundry_properties() is called but boundry_id(" << boundry_id << ") is greater than maximum(" << max_boundry_count << ")" << std::endl;
		ASSERT(false);
	}

	if (boundry_id == 0) {
		std::cout << "[LBM Error] LBM2D::set_boundry_properties() is called but boundry_id(0) is defined to be fluid, it cannot be treated as an object" << std::endl;
		ASSERT(false);
	}

	if (boundry_id >= objects_cpu.size())
		objects_cpu.resize(boundry_id + 1);
	objects_cpu[boundry_id] = _object_desc(velocity_translational, velocity_angular, center_of_mass, temperature, effective_density);
}

void LBM2D::set_boundry_properties(
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

void LBM2D::set_boundry_properties(uint32_t boundry_id, float temperature, float effective_density)
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

void LBM2D::clear_boundry_properties()
{
	objects_cpu.clear();
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
		{"velocity_set",			get_VelocitySet_to_macro(velocity_set)},
		{"boundry_count",			std::to_string(objects_cpu.size())},
		{"bits_per_boundry",		std::to_string(bits_per_boundry)},
		{"periodic_x",				periodic_x				? "1" : "0"},
		{"periodic_y",				periodic_y				? "1" : "0"},
		{"forcing_scheme",			is_forcing_scheme		? "1" : "0"},
		{"constant_force",			is_force_field_constant ? "1" : "0"},
		{"thermal_flow",			is_flow_thermal			? "1" : "0"},
		{"velocity_set_thermal",	get_SimplifiedVelocitySet_to_macro(thermal_lattice_velocity_set)},
		{"multiphase_flow",			is_flow_multiphase		? "1" : "0"},
	};

	return definitions;
}

void LBM2D::_stream()
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_stream;


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
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_collide_with_precomputed_velocity;
	
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

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
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
	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
	kernel.update_uniform("relaxation_time", relaxation_time);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::_stream_thermal()
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_stream_thermal;

	Buffer& thermal_lattice_source = *_get_thermal_lattice_source();
	Buffer& thermal_lattice_target = *_get_thermal_lattice_target();

	// add forces implementation
	kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer_source", thermal_lattice_source, 0);
	kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer_target", thermal_lattice_target, 0);
	kernel.update_uniform_as_uniform_buffer("thermal_velocity_set_buffer", *thermal_lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);

	kernel.dispatch_thread(resolution.x * resolution.y * get_SimplifiedVelocitySet_vector_count(thermal_lattice_velocity_set), 1, 1);

	_swap_thermal_lattice_buffers();
}

void LBM2D::_collide_thermal()
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_collide_thermal;

	Buffer& lattice_source = *_get_lattice_source();
	Buffer& lattice_target = *_get_lattice_target();
	kernel.update_uniform_as_storage_buffer("lattice_buffer_source", lattice_source, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer_target", lattice_target, 0);
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);

	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
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

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);

	_swap_thermal_lattice_buffers();
}

void LBM2D::_apply_boundry_conditions_thermal()
{
	if (bits_per_boundry == 0)
		return;

	compile_shaders();

	if (objects == nullptr || boundries == nullptr) {
		std::cout << "[LBM2D Error] LBM2D::_apply_boundry_conditions_thermal() is called and bits_per_boundry is non-zero but objects or boundries is nullptr" << std::endl;
		ASSERT(false);
	}

	ComputeProgram& kernel = *lbm2d_boundry_condition_thermal;

	Buffer& thermal_lattice = *_get_thermal_lattice_source();

	kernel.update_uniform_as_storage_buffer("thermal_lattice_buffer", thermal_lattice, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
	kernel.update_uniform_as_storage_buffer("objects_buffer", *objects, 0);
	kernel.update_uniform_as_uniform_buffer("thermal_velocity_set_buffer", *thermal_lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);

}

void LBM2D::_set_populations_to_equilibrium_thermal(Buffer& temperature_field, Buffer& velocity_field)
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_set_equilibrium_populations_thermal;

	Buffer& lattice_thermal = *_get_thermal_lattice_source();

	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice_thermal, 0);
	kernel.update_uniform_as_storage_buffer("temperature_buffer", temperature_field, 0);
	kernel.update_uniform_as_storage_buffer("velocity_buffer", velocity_field, 0);
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *thermal_lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3))); // is thermal speed of sound heat conductivity?
	kernel.update_uniform("relaxation_time", thermal_relaxation_time);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

LBM2D::_object_desc::_object_desc(
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
