#pragma once

#include "Mesh.h"
#include "Texture3D.h"
#include "Texture2D.h"

#include "FrameBuffer.h"
#include "ShaderCompiler.h"
#include "ComputeProgram.h"

class MeshRasterizer3D {
public:

	MeshRasterizer3D() = default;

	void rasterize(Mesh& mesh,
		Texture3D& target_texture,
		glm::mat4 model_matrix = glm::identity<glm::mat4>()
	);

private:
	void compile_shaders(Texture3D::ColorTextureFormat texture_format);
	bool is_compiled = false;
	Texture3D::ColorTextureFormat compiled_format = Texture3D::ColorTextureFormat::RGB10_A2UI;

	std::shared_ptr<Framebuffer> framebuffer = nullptr;
	std::shared_ptr<Program> mesh_renderer = nullptr;
	std::shared_ptr<ComputeProgram> raster_unifier = nullptr;
	std::shared_ptr<ComputeProgram> copy_to_3d = nullptr;

};