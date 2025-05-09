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
	lbm2d_add_random_population				= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "add_random_population.comp"), definitions);
	is_programs_compiled = true;
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
			_apply_boundry_conditions();
			_collide();

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

void LBM2D::initialize_fields(std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda, glm::ivec2 resolution, float relaxation_time, VelocitySet velocity_set, FloatingPointAccuracy fp_accuracy)
{
	set_relaxation_time(relaxation_time);
	set_velocity_set(velocity_set);
	set_floating_point_accuracy(fp_accuracy);
	generate_lattice(resolution);

	bool is_force_field_constant;
	glm::vec3 constant_force_field;

	_initialize_fields_default_pass(
		initialization_lambda,
		resolution,
		is_force_field_constant,
		constant_force_field,
		fp_accuracy
	);

	_initialize_fields_boundries_pass(
		initialization_lambda,
		resolution,
		fp_accuracy
	);
	
	_initialize_fields_force_pass(
		is_force_field_constant,
		constant_force_field,
		initialization_lambda,
		resolution,
		fp_accuracy
	);
}

void LBM2D::_initialize_fields_default_pass(std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda, glm::ivec2 resolution, bool& out_is_force_field_constant, glm::vec3& out_constant_force_field, FloatingPointAccuracy fp_accuracy)
{
	Buffer velocity_buffer(resolution.x * resolution.y * sizeof(glm::vec4));
	Buffer density_buffer(resolution.x * resolution.y * sizeof(float));

	velocity_buffer.map(Buffer::MapInfo(Buffer::MapInfo::Upload, Buffer::MapInfo::Temporary));
	density_buffer.map(Buffer::MapInfo(Buffer::MapInfo::Upload, Buffer::MapInfo::Temporary));

	glm::vec4* velocity_buffer_data = (glm::vec4*)velocity_buffer.get_mapped_pointer();
	float* density_buffer_data = (float*)density_buffer.get_mapped_pointer();

	FluidProperties temp_properties;
	initialization_lambda(glm::ivec2(0, 0), temp_properties);

	bool is_force_field_constant = false;
	glm::vec3 constant_force_field = temp_properties.force;

	uint32_t object_count = 0;

	for (int32_t x = 0; x < resolution.x; x++) {
		for (int32_t y = 0; y < resolution.y; y++) {
			FluidProperties properties;
			initialization_lambda(glm::ivec2(x, y), properties);

			object_count = std::max(object_count, properties.boundry_id + 1);

			if (properties.force != constant_force_field)
				is_force_field_constant = false;

			velocity_buffer_data[y * resolution.x + x] = glm::vec4(properties.velocity, 0.0f);
			density_buffer_data[y * resolution.x + x] = properties.density;
		}
	}

	// compute equilibrium and non-equilibrium populations according to chapter 5.
	velocity_buffer.unmap();
	density_buffer.unmap();

	velocity_buffer_data = nullptr;
	density_buffer_data = nullptr;

	_set_populations_to_equilibrium(density_buffer, velocity_buffer);

	int32_t relaxation_iteration_count = 1;
	for (int32_t i = 0; i < relaxation_iteration_count; i++) {
		_collide_with_precomputed_velocities(velocity_buffer);
		// _apply_boundry_conditions(); the book doesn't specitfy whether or not to enforce boundry conditions in initialization algorithm
		_stream();
	}

	objects_cpu.resize(object_count);
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

	
	// objects initialization
	//							   trans_vel + id	   angular_vel		   center_of_mass
	size_t object_size_on_device = sizeof(glm::vec4) + sizeof(glm::vec4) + sizeof(glm::vec4);
	objects = std::make_shared<Buffer>(object_size_on_device * object_count);

	objects->map(Buffer::MapInfo(Buffer::MapInfo::Bothways, Buffer::MapInfo::Temporary));
	glm::vec4* objects_mapped_buffer = (glm::vec4*)objects->get_mapped_pointer();
	
	for (uint32_t i = 0; i < objects_cpu.size(); i++) {
		_object_desc& desc = objects_cpu[i];
		objects_mapped_buffer[0] = glm::vec4(desc.velocity_translational, desc.boundry_id);
		objects_mapped_buffer[1] = glm::vec4(desc.velocity_angular, 0);
		objects_mapped_buffer[2] = glm::vec4(desc.center_of_mass, 0);
	}

	objects->unmap();
	objects_mapped_buffer = nullptr;

	
	// boundries initialization

	size_t boundries_buffer_size = std::ceil((bits_per_boundry * resolution.x * resolution.y) / 8.0f);
	boundries = std::make_shared<Buffer>(boundries_buffer_size);

	boundries->map();
	int8_t* boundries_mapped_buffer = (int8_t*)boundries->get_mapped_pointer();

	for (int32_t x = 0; x < resolution.x; x++) {
		for (int32_t y = 0; y < resolution.y; y++) {
			FluidProperties properties;
			initialization_lambda(glm::ivec2(x, y), properties);
			
			size_t voxel_id = y * resolution.x + x;
			size_t bits_begin = voxel_id * bits_per_boundry;
			
			size_t byte_offset = bits_begin / 8;
			int32_t subbyte_offset_in_bits = bits_begin % 8;

			boundries_mapped_buffer[byte_offset] |= (properties.boundry_id << subbyte_offset_in_bits);
		}
	}

	boundries->unmap();
	boundries_mapped_buffer = nullptr;

}

void LBM2D::_initialize_fields_force_pass(bool is_force_field_constant, glm::vec3 constant_force_field, std::function<void(glm::ivec2, FluidProperties&)> initialization_lambda, glm::ivec2 resolution, FloatingPointAccuracy fp_accuracy)
{
	if (is_force_field_constant) {
		if (constant_force_field == glm::vec3(0)) {
			this->forces = nullptr;
			this->is_force_field_constant = false;
		}
		else {
			this->forces = std::make_shared<Buffer>(sizeof(glm::vec4) * 1);
			this->is_force_field_constant = true;
		}
	}
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

	if (target_texture.get_internal_format_color() != Texture2D::ColorTextureFormat::R32F) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_boundries() is called but target_texture's format wasn't compatible" << std::endl;
		ASSERT(false);
	}

	compile_shaders();

	ComputeProgram& kernel = *lbm2d_copy_boundries;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
	kernel.update_uniform_as_image("target_texture", target_texture, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("texture_resolution", target_texture.get_size());

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}

void LBM2D::map_boundries()
{
	if (boundries == nullptr) {
		std::cout << "[LBM Error] LBM2D::map_boundries() is called but lattice isn't initialized yet. call generate_lattice() first" << std::endl;
		ASSERT(false);
	}

	if (is_boundries_mapped())
		return;

	Buffer::MapInfo map_info;
	map_info.lifetime = Buffer::MapInfo::Temporary;
	map_info.direction = Buffer::MapInfo::Bothways;
	boundries->map(map_info);
}

void LBM2D::unmap_boundries()
{
	boundries->unmap();
}

bool LBM2D::is_boundries_mapped() {
	if (boundries == nullptr)
		return false;
	return boundries->is_mapped();
}

void* LBM2D::get_mapped_boundries() {
	if (boundries == nullptr)
		return nullptr;
	return boundries->get_mapped_pointer();
}

void LBM2D::set_boundry(glm::ivec2 voxel_coordinate, bool value) {
	
	if (boundries == nullptr) {
		std::cout << "[LBM Error] LBM2D::set_boundry() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (!is_boundries_mapped())
		map_boundries();

	int8_t* mapped_buffer = (int8_t*)boundries->get_mapped_pointer();
	size_t voxel_id = (voxel_coordinate.y * resolution.x + voxel_coordinate.x);
	size_t byte_id = voxel_id / 8;
	int8_t bit_id = voxel_id % 8;

	if (mapped_buffer == nullptr) {
		std::cout << "[LBM Error] LBM2D::set_boundry() mapped pointer was nullptr" << std::endl;
		ASSERT(false);
	}
	if (byte_id < 0 || byte_id >= boundries->get_mapped_buffer_size()) {
		std::cout << "[LBM Error] LBM2D::set_boundry() voxel_coordinate was out of bounds" << std::endl;
		ASSERT(false);
	}

	if (value)
		mapped_buffer[byte_id] |= 1 << bit_id;
	else
		mapped_buffer[byte_id] &= ~(1 << bit_id);
}

void LBM2D::set_boundry(glm::ivec2 voxel_coordinate_begin, glm::ivec2 voxel_coordinate_end, bool value)
{
	if (voxel_coordinate_begin.x > voxel_coordinate_end.x)
		std::swap(voxel_coordinate_begin.x, voxel_coordinate_end.x);
	if (voxel_coordinate_begin.y > voxel_coordinate_end.y)
		std::swap(voxel_coordinate_begin.y, voxel_coordinate_end.y);
	
	for (int32_t x = voxel_coordinate_begin.x; x < voxel_coordinate_end.x; x++) {
		for (int32_t y = voxel_coordinate_begin.y; x < voxel_coordinate_end.y; x++) {
			set_boundry(glm::ivec2(x, y), value);
		}
	}
}

void LBM2D::set_boundry(bool value)
{
	set_boundry(glm::ivec2(0, 0), resolution, value);
}

bool LBM2D::get_boundry(glm::ivec2 voxel_coordinate) {
	if (boundries == nullptr) {
		std::cout << "[LBM Error] LBM2D::get_boundry() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	if (!is_boundries_mapped())
		map_boundries();

	int8_t* mapped_buffer = (int8_t*)boundries->get_mapped_pointer();
	size_t voxel_id = (voxel_coordinate.y * resolution.x + voxel_coordinate.x);
	size_t byte_id = voxel_id / 8;
	int32_t bit_id = voxel_id % 8;

	if (mapped_buffer == nullptr) {
		std::cout << "[LBM Error] LBM2D::get_boundry() mapped pointer was nullptr" << std::endl;
		ASSERT(false);
	}
	if (byte_id < 0 || byte_id >= boundries->get_mapped_buffer_size()) {
		std::cout << "[LBM Error] LBM2D::get_boundry() voxel_coordinate was out of bounds" << std::endl;
		ASSERT(false);
	}

	return mapped_buffer[byte_id] &= 1 << bit_id;
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
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	//kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
	kernel.update_uniform("relaxation_time", relaxation_time);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);

	_swap_lattice_buffers();
}

void LBM2D::_apply_boundry_conditions() {

	if (is_boundries_mapped())
		unmap_boundries();
	
	compile_shaders();
	ComputeProgram& kernel = *lbm2d_boundry_condition;

	Buffer& lattice = *_get_lattice_source();

	kernel.update_uniform_as_storage_buffer("lattice_buffer", lattice, 0);
	kernel.update_uniform_as_storage_buffer("boundries_buffer", *boundries, 0);
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

