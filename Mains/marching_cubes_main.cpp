#include "GraphicsCortex.h"
#include "MarchingCubes/MarchingCubes.h"
#include "LBM/LBM.h"

#include "Demos/LBMDemo2D.h"
#include "Demos/LBMDemo3D.h"

int main() {
	WindowDescription desc;
	desc.w_scale_framebuffer_size = false;
	desc.w_scale_window_size = false;
	desc.f_swap_interval = 0;
	desc.w_resolution = glm::ivec2(1024);
	Window window(desc);

	LBM solver;
	demo3d::multiphase_droplet_collision(solver);

	solver.iterate_time();
	
	MarchingCubes marching_cubes;
	std::shared_ptr<Mesh> mesh = marching_cubes.compute(*solver.get_velocity_density_texture(), 0);

	Camera camera;
	camera.screen_width = window.get_window_resolution().x;
	camera.screen_height = window.get_window_resolution().y;

	Program debug_renderer = default_program::debug::flatcolor_program();
	debug_renderer.update_uniform("color", glm::vec4(1, 1, 1, 1));
	debug_renderer.update_uniform("model", glm::identity<glm::mat4>());

	while (!window.should_close()) {
		double deltatime = window.handle_events(true);
		camera.handle_movements((GLFWwindow*)window.get_handle(), deltatime);
		
		camera.update_matrixes();
		camera.update_default_uniforms(debug_renderer);

		primitive_renderer::clear(0, 0, 0, 1);

		primitive_renderer::render(
			debug_renderer,
			*mesh->get_vertex_attribute_buffer(),
			*mesh->get_index_buffer(),
			PrimitiveType::triangle,
			IndexType::i_ui32,
			RenderParameters(),
			1,
			0
		);

		window.swap_buffers();
	}
}