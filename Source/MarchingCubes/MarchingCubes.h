#pragma once

#include "Application/ProgramSourcePaths.h"

#include "ComputeProgram.h"
#include "Mesh.h"
#include "Texture3D.h"

#include "memory"

#include "vec3.hpp"

class MarchingCubes {
public:
	MarchingCubes() = default;

	size_t get_index_buffer_needed_size_in_bytes(glm::ivec3 volume_resolution);
	size_t get_vertex_buffer_needed_size_in_bytes(glm::ivec3 volume_resolution);

	std::unique_ptr<Mesh> compute(Texture3D& source_texture, int32_t mipmap);
	void compute(Texture3D& source_texture, int32_t mipmap, Buffer& out_vertex, Buffer& out_index);
	void compute_verticies_only(Texture3D& source_texture, int32_t mipmap, Mesh& out_mesh);
	void compute_indicies_only(Texture3D& source_texture, int32_t mipmap, Mesh& out_mesh);

	void render(Framebuffer framebuffer, Camera& camera, Texture3D& source_texture);
	void render(Camera& camera, Texture3D& source_texture);

private:

	bool compiled = false;
	Texture3D::ColorTextureFormat compiled_format = Texture3D::ColorTextureFormat::RGB4;
	void _compile_shaders(Texture3D::ColorTextureFormat volume_internal_format);

	std::shared_ptr<ComputeProgram> marching_cubes = nullptr;
	std::shared_ptr<ComputeProgram> marching_cubes_vertex_only = nullptr;
	std::shared_ptr<ComputeProgram> marching_cubes_index_only = nullptr;

	std::shared_ptr<UniformBuffer> indicies_look_up_table = nullptr;
};
