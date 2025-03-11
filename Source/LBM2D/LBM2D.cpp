#include "LBM2D.h"
#include "Application/ProgramSourcePaths.h"

void LBM2D::compile_shaders()
{
	if (is_programs_compiled)
		return;

	auto definitions = _generate_shader_macros();

	lbm2d_stream = std::make_shared<ComputeProgram>(Shader(lbm2d_shader_directory / "stream.comp"), definitions);

	is_programs_compiled = true;
}

void LBM2D::generate_lattice(glm::ivec2 resolution, glm::vec2 volume_dimentions_meters)
{
	this->resolution = resolution;
	this->volume_dimentions_meters = volume_dimentions_meters;

	_generate_lattice_buffer();
}

void LBM2D::iterate_time(double time_milliseconds)
{
	total_time_elapsed_ms += time_milliseconds;

	_stream(time_milliseconds);
	//_collide(time_milliseconds);
	//_apply_boundry_conditions(time_milliseconds);
}

double LBM2D::get_total_time_elapsed_ms()
{
	return total_time_elapsed_ms;
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
	if (velocity_index < 0 || velocity_index >= get_velocity_count()) {
		std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_index() is called but given index is out of bounds" << std::endl;
		ASSERT(false);
	}

	//if (Texture2D::ColorTextureFormat_channels(target_texture.get_internal_format_color()) != 1) {
	//	std::cout << "[LBM Error] LBM2D::copy_to_texture_velocity_index() is called but given texture is not 1 dimentional" << std::endl;
	//	ASSERT(false);
	//}

	operation->set_constant("texture_resolution", target_texture.get_size());
	operation->set_constant("velocity_count_per_voxel", get_velocity_count());
	operation->set_constant("velocity_index", velocity_index);

	operation->compute(
		target_texture,
		*lattice, "float",
		"source[(id.y * texture_resolution.x + id.x) * velocity_count_per_voxel + velocity_index]"
	);
}

void LBM2D::copy_to_texture_velocity_total(Texture2D& target_texture)
{
	
}

void LBM2D::_generate_lattice_buffer()
{
	size_t voxel_count = resolution.x * resolution.y;
	size_t total_buffer_size_in_bytes = voxel_count * get_velocity_count() * get_FLoatingPointAccuracy_size_in_bytes(floating_point_accuracy);

	lattice = std::make_shared<Buffer>(total_buffer_size_in_bytes);
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

void LBM2D::_stream(double time_milliseconds)
{
	compile_shaders();

	ComputeProgram& kernel = *lbm2d_stream;

	//kernel.update_uniform_as_storage_buffer("lattice_buffer_source", *lattice, 0);
	kernel.update_uniform_as_storage_buffer("lattice_buffer_target", *lattice, 0);
	kernel.update_uniform("lattice_resolution", resolution);
	kernel.update_uniform("velocity_count", get_velocity_count());
	
	kernel.dispatch_thread(resolution.x * resolution.y * get_velocity_count(), 1, 1);
}
