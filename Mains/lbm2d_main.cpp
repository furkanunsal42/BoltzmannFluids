#include <thread>
#include "GUI/GuiManager.h"
#include "GraphicsCortex.h"
#include "LBM2D/LBM2D.h"

using namespace std::chrono_literals;

void run_rendering_loop() {

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

	LBM2D lbm2d_solver;

	lbm2d_solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {
			properties.is_boundry = glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution) / glm::vec2(4, 2)) < 32;
			properties.velocity = glm::vec3(1, 1, 0) / 16.0f;
			properties.density = 1.0f;
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
			Texture2D& texture_target = texture_1c;
			lbm2d_solver.copy_to_texture_boundries(texture_target);
			fb.attach_color(0, texture_target, 0);
		}

		fb.blit_to_screen(simulation_resolution, simulation_resolution, Framebuffer::Channel::COLOR, Framebuffer::Filter::LINEAR);

		window.swap_buffers();
	}
}

int main(int argc, char** argv) {
	GuiManager guiManager(argc, argv);

	std::thread renderingThread(run_rendering_loop); // OpenGL rendering part (will be embedded into Main UI using QTWidget)

	guiManager.run();	// Main UI (QT)


	renderingThread.join();	// Will be removed when embedded
	return 0;
}