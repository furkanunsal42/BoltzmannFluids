#include "MeshRasterizer3D.h"

#include "PrimitiveRenderer.h"
#include "Application/ProgramSourcePaths.h"
#include "Camera.h"

void MeshRasterizer3D::rasterize(
	Mesh& mesh, 
	glm::mat4 model_matrix, 
	Texture3D& target_texture, 
	glm::vec4 filled_value, 
	glm::vec4 blank_value
) {
	compile_shaders();

	glm::vec4 viewport = primitive_renderer::get_viewport_position_size();
	primitive_renderer::set_viewport(
		glm::ivec2(0),
		glm::ivec2(target_texture.get_size().x, target_texture.get_size().y)
	);

	Camera camera;
	camera.screen_width = target_texture.get_size().x;
	camera.screen_height = target_texture.get_size().y;
	camera.position = glm::vec3(0, 0, 1);
	camera.ortho_size = 1.0f;
	camera.perspective = false;

	float voxel_depth = 2.0f / target_texture.get_size().z;

	camera.update_matrixes();
	camera.update_default_uniforms(*mesh_renderer);
	mesh_renderer->update_uniform("filled_value", filled_value);
	mesh_renderer->update_uniform("blank_value", blank_value);

	Texture2D slice_texture(
		target_texture.get_size().x,
		target_texture.get_size().y,
		Texture2D::ColorTextureFormat::R8,
		1,
		0
	);

	Texture2D temp_texture(
		target_texture.get_size().x,
		target_texture.get_size().y,
		Texture2D::ColorTextureFormat::R8,
		1,
		0
	);

	slice_texture.clear(glm::vec4(0));
	temp_texture.clear(glm::vec4(0));

	for (int32_t z = 0; z < target_texture.get_size().z; z++) {
		
		framebuffer->attach_color(0, slice_texture, 0);
		framebuffer->activate_draw_buffer(0);
		framebuffer->bind_draw();
		
		primitive_renderer::clear(0, 0, 0, 0);

		camera.min_distance = z * voxel_depth;
		camera.max_distance = (z+1) * voxel_depth;

		camera.update_matrixes();
		camera.update_default_uniforms(*mesh_renderer);

		mesh.traverse([&](Mesh::Node& node, glm::mat4 submodel_matrix) {
			mesh_renderer->update_uniform("model", model_matrix);
			for (int32_t submodel : node.get_submeshes())
				primitive_renderer::render(
					*mesh_renderer,
					*mesh.get_mesh(submodel),
					RenderParameters(),
					1,
					0
				);
			});

		raster_unifier->update_uniform_as_image("slice_texture", slice_texture, 0);
		raster_unifier->update_uniform_as_image("temp_texture", temp_texture, 0);
		raster_unifier->update_uniform_as_image("volume", target_texture, 0);
		raster_unifier->update_uniform("volume_resolution", target_texture.get_size());
		raster_unifier->update_uniform("current_slice", z);

		raster_unifier->dispatch_thread(slice_texture.get_size().x, slice_texture.get_size().y, 1);
	}

	Framebuffer::get_screen().bind_read_draw();
	primitive_renderer::set_viewport(viewport);

}

void MeshRasterizer3D::compile_shaders()
{
	if (is_compiled)
		return;

	mesh_renderer	= std::make_shared<Program>(Shader(mesh_rasterizer3d_shader_directory / "rasterizer.vert", mesh_rasterizer3d_shader_directory / "rasterizer.frag"));
	raster_unifier	= std::make_shared<ComputeProgram>(Shader(mesh_rasterizer3d_shader_directory / "raster_unifier.comp"));
	framebuffer		= std::make_shared<Framebuffer>();

	is_compiled = true;
}
