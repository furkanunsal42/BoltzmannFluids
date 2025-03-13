#include "LBM2D.h"
#include "Application/ProgramSourcePaths.h"

void LBM2D::compile_shaders()
{
	if (is_programs_compiled)
		return;

	auto definitions = _generate_shader_macros();

	lbm2d_stream = std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "stream.comp"), definitions);
	lbm2d_collide = std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "collide.comp"), definitions);

	is_programs_compiled = true;
}

void LBM2D::generate_lattice(glm::ivec2 resolution, glm::vec2 volume_dimentions_meters)
{
	this->resolution = resolution;
	this->volume_dimentions_meters = volume_dimentions_meters;

	_generate_lattice_buffer();
}

void LBM2D::iterate_time(std::chrono::duration<double, std::milli> deltatime)
{
	total_time_elapsed += deltatime;

	_stream();
	_collide();
	//_apply_boundry_conditions(time_milliseconds);

	_swap_lattice_buffers();
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

void LBM2D::copy_to_texture_velocity_index(Texture2D& target_texture, int32_t velocity_index)
{
	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_index() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}
	
	if (velocity_index < 0 || velocity_index >= get_velocity_count()) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_index() is called but given index is out of bounds" << std::endl;
		ASSERT(false);
	}

	//if (Texture2D::ColorTextureFormat_channels(target_texture.get_internal_format_color()) != 1) {
	//	std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_index() is called but given texture is not 1 dimentional" << std::endl;
	//	ASSERT(false);
	//}

	operation->push_constant("texture_resolution", target_texture.get_size());
	operation->push_constant("velocity_count_per_voxel", get_velocity_count());
	operation->push_constant("velocity_index", velocity_index);

	operation->compute(
		target_texture,
		*_get_lattice_source(), "float",
		"source[(id.y * texture_resolution.x + id.x) * velocity_count_per_voxel + velocity_index]"
	);
}

void LBM2D::_generate_lattice_buffer()
{
	size_t voxel_count = resolution.x * resolution.y;
	size_t total_buffer_size_in_bytes = voxel_count * get_velocity_count() * get_FLoatingPointAccuracy_size_in_bytes(floating_point_accuracy);

	lattice0 = std::make_shared<Buffer>(total_buffer_size_in_bytes);
	lattice1 = std::make_shared<Buffer>(total_buffer_size_in_bytes);
	
	lattice0->clear(.5f);
	lattice1->clear(.5f);

	lattice_velocity_set_buffer = std::make_unique<UniformBuffer>();
	lattice_velocity_set_buffer->push_variable_array(get_velocity_count()); // a vec4 for every velocity direction
	
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

void LBM2D::copy_to_texture_density(Texture2D& target_texture)
{
	if (_get_lattice_source() == nullptr) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_density() is called but lattice wasn't generated" << std::endl;
		ASSERT(false);
	}

	operation->push_constant("texture_resolution", target_texture.get_size());
	operation->push_constant("velocity_count_per_voxel", get_velocity_count());

	operation->set_precomputation_statement(
		"float compute_density(uvec2 pixel_id) {"
		"	float density = 0;"
		"	for (int velocity_index = 0; velocity_index < velocity_count_per_voxel; velocity_index++)"
		"		density += source[(pixel_id.y * texture_resolution.x + pixel_id.x) * velocity_count_per_voxel + velocity_index];"
		"	return density;"
		"}"
	);

	operation->compute(
		target_texture,
		*_get_lattice_source(), "float",
		"compute_density(id.xy)"
	);
}

glm::ivec2 LBM2D::get_resolution()
{
	return resolution;
}

int32_t LBM2D::get_velocity_count()
{
	return get_VelocitySet_velocity_count(velocity_set);
}

glm::vec2 LBM2D::get_volume_dimentions_meters()
{
	return volume_dimentions_meters;
}

std::vector<std::pair<std::string, std::string>> LBM2D::_generate_shader_macros()
{
	std::vector<std::pair<std::string, std::string>> definitions{
		{"floating_point_accuracy", get_FLoatingPointAccuracy_to_macro(floating_point_accuracy)},
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
	
	kernel.dispatch_thread(resolution.x * resolution.y * get_velocity_count(), 1, 1);
}

void LBM2D::_collide()
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_collide;

	Buffer& lattice_source = *_get_lattice_source();
	Buffer& lattice_target = *_get_lattice_target();

	kernel.update_uniform_as_storage_buffer("lattice_buffer_source", lattice_target, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer_target", lattice_target, 0);
	kernel.update_uniform_as_uniform_buffer("velocity_set_buffer", *lattice_velocity_set_buffer, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("lattice_speed_of_sound", (float)(1.0 / glm::sqrt(3)));
	kernel.update_uniform("relaxation_time", 0.01f);

	kernel.dispatch_thread(resolution.x * resolution.y, 1, 1);
}
