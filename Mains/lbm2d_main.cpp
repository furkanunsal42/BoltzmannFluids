#include "GraphicsCortex.h"
#include "LBM2D/LBM2D.h"

using namespace std::chrono_literals;

int main() {

	glm::ivec2 simulation_resolution(1024);

	WindowDescription desc;
	desc.w_scale_framebuffer_size = false;
	desc.w_scale_window_size = false;
	desc.f_swap_interval = 0;
	desc.w_resolution = simulation_resolution;
	Window window(desc);

	uint32_t display_mode = 3;

	window.newsletters->on_key_events.subscribe([&](const Window::KeyPressResult& key_press) {
		if (key_press.action == Window::PressAction::PRESS) {
			switch (key_press.key) {
			case Window::Key::NUM_1:
				display_mode = 1;
				break;
			case Window::Key::NUM_2:
				display_mode = 2;
				break;
			case Window::Key::NUM_3:
				display_mode = 3;
				break;
			case Window::Key::NUM_4:
				display_mode = 4;
				break;
			case Window::Key::ESCAPE:
				exit(0);
				break;
			}
		}
		});

	window.newsletters->on_should_close_events.subscribe([&]() {
		exit(0);
		});

	//window.newsletters->on_window_refresh_events.subscribe([&]() {
	//	//std::cout << "here" << std::endl;
	//	//primitive_renderer::set_viewport_size(window.get_framebuffer_resolution());
	//	});

	LBM2D lbm2d_solver;

	lbm2d_solver.set_boundry_velocity(1, glm::vec3(1, 1, 1), glm::vec3(1, 1, 1), glm::vec3(1, 1, 1));

	lbm2d_solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {
			properties.boundry_id = false;
			//properties.boundry_id |= glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 1 / 4.0, simulation_resolution.y / 2)) < 32;
			properties.boundry_id |= glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 3 / 4.0, simulation_resolution.y / 2)) < 32;
			properties.velocity = glm::vec3(1, 1, 0) / 16.0f;
			properties.density = 1.0f;
			properties.force = glm::vec3(0, -1, 0);
		},
		glm::ivec2(simulation_resolution),
		0.53f,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32
	);

	Texture2D texture_1c(simulation_resolution.x, simulation_resolution.y, Texture2D::ColorTextureFormat::R32F, 1, 0, 0);
	Texture2D texture_4c(simulation_resolution.x, simulation_resolution.y, Texture2D::ColorTextureFormat::RGBA32F, 1, 0, 0);

	Framebuffer fb;
	fb.attach_color(0, texture_1c, 0);
	fb.activate_draw_buffer(0);

	while (true) {
		double deltatime = window.handle_events(true);
		lbm2d_solver.iterate_time(std::chrono::duration<double, std::milli>(deltatime*10));
		
		if (display_mode == 1) {
			Texture2D& texture_target = texture_1c;
			lbm2d_solver.copy_to_texture_density(texture_target);
			fb.attach_color(0, texture_target, 0);
		}
		else if (display_mode == 2){
			Texture2D& texture_target = texture_1c;
			lbm2d_solver.copy_to_texture_velocity_magnetude(texture_target);
			fb.attach_color(0, texture_target, 0);
		}
		else if (display_mode == 3){
			Texture2D& texture_target = texture_4c;
			lbm2d_solver.copy_to_texture_velocity_vector(texture_target);
			fb.attach_color(0, texture_target, 0);
		}
		else if (display_mode == 4){
			Texture2D& texture_target = texture_4c;
			lbm2d_solver.copy_to_texture_boundries(texture_target);
			fb.attach_color(0, texture_target, 0);
		}

		fb.blit_to_screen(simulation_resolution, window.get_framebuffer_resolution(), Framebuffer::Channel::COLOR, Framebuffer::Filter::NEAREST);

		window.swap_buffers();
	}

}