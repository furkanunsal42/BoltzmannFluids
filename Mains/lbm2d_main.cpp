#include "GraphicsCortex.h"
#include "LBM2D/LBM2D.h"

using namespace std::chrono_literals;

int main() {

	glm::ivec2 simulation_resolution(1024, 1024);

	WindowDescription desc;
	desc.f_swap_interval = 1;
	desc.w_resolution = simulation_resolution;
	Window window(desc);

	LBM2D lbm2d_solver;

	lbm2d_solver.set_velocity_set(VelocitySet::D2Q9);
	lbm2d_solver.set_floating_point_accuracy(FloatingPointAccuracy::fp32);
	lbm2d_solver.compile_shaders();
	
	lbm2d_solver.generate_lattice(glm::ivec2(simulation_resolution.x, simulation_resolution.y), glm::vec2(1));

	for (int x = 0; x < simulation_resolution.x; x++) {
		for (int y = 0; y < simulation_resolution.y; y++) {
			if (glm::distance(glm::vec2(x, y), glm::vec2(simulation_resolution) / glm::vec2(4, 2)) < 32)
				lbm2d_solver.set_boundry(glm::ivec2(x, y), true);
		}
	}

	lbm2d_solver.set_velocity(1.0f);
	lbm2d_solver.add_random_velocity(0.3f);
	lbm2d_solver.set_velocity(1, 2.3f);

	Texture2D texture(simulation_resolution.x, simulation_resolution.y, Texture2D::ColorTextureFormat::R32F, 1, 0, 0);
	Framebuffer fb;
	fb.attach_color(0, texture, 0);
	fb.activate_draw_buffer(0);

	uint32_t display_mode = 1;

	while (!window.should_close()) {
		double deltatime = window.handle_events(true);

		lbm2d_solver.iterate_time();
		
		if (window.get_key(Window::Key::NUM_1) == Window::PressAction::PRESS)
			display_mode = 1;
		if (window.get_key(Window::Key::NUM_2) == Window::PressAction::PRESS)
			display_mode = 2;
		if (window.get_key(Window::Key::NUM_3) == Window::PressAction::PRESS)
			display_mode = 3;

		if (display_mode == 1)
			lbm2d_solver.copy_to_texture_density(texture);
		else if (display_mode == 2)
			lbm2d_solver.copy_to_texture_curl(texture);
		else if (display_mode == 3)
			lbm2d_solver.copy_to_texture_boundries(texture);

		fb.blit_to_screen(simulation_resolution, simulation_resolution, Framebuffer::Channel::COLOR, Framebuffer::Filter::LINEAR);

		window.swap_buffers();
	}

}