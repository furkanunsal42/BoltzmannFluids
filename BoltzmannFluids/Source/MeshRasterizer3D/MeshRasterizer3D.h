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
		glm::mat4 model_matrix,
		Texture3D& target_texture, 
		glm::vec4 filled_value = glm::vec4(1),
		glm::vec4 blank_value = glm::vec4(0)
	);

private:
	void compile_shaders();
	bool is_compiled = false;

	std::shared_ptr<Framebuffer> framebuffer = nullptr;
	std::shared_ptr<Program> mesh_renderer = nullptr;
	std::shared_ptr<ComputeProgram> raster_unifier = nullptr;

};