#include "LBM2D.h"
#include "Application/ProgramSourcePaths.h"

void LBM2D::compile_shaders()
{
	if (is_programs_compiled)
		return;

	auto definitions = _generate_shader_macros();

	lbm2d_stream					= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "stream.comp"), definitions);
	lbm2d_collide					= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "collide.comp"), definitions);
	lbm2d_boundry_condition			= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "boundry_condition.comp"), definitions);
	lbm2d_set_population			= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "set_population.comp"), definitions);
	lbm2d_copy_boundries			= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_boundries.comp"), definitions);
	lbm2d_copy_density 				= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_density.comp"), definitions);
	lbm2d_copy_population			= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_population.comp"), definitions);
	lbm2d_copy_velocity_magnitude	= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_velocity_magnitude.comp"), definitions);
	lbm2d_copy_velocity_total		= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "copy_velocity_total.comp"), definitions);
	lbm2d_add_random_population		= std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "add_random_population.comp"), definitions);
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

	std::vector<glm::ivec3> velocity_field;

	size_t voxel_count = resolution.x * resolution.y;
	velocity_field.reserve(voxel_count);

	for (int32_t x = 0; x < resolution.x; x++) {
		for (int32_t y = 0; y < resolution.y; y++) {
			FluidProperties properties;
			initialization_lambda(glm::ivec2(x, y), properties);
			
			velocity_field.push_back(properties.velocity);
			set_boundry(glm::ivec2(x, y), properties.is_boundry);
		}
	}

	// compute equilibrium and non-equilibrium populations according to chapter 5.

	// TEMP
	set_population(0.7);
	add_random_population(1, 0.7f);
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
