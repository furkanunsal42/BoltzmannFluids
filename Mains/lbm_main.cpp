#include "GraphicsCortex.h"
#include "LBM/LBM.h"

#include "Demos/LBMDemo2D.h"
#include "Demos/LBMDemo3D.h"

int main() {

	WindowDescription desc;
	desc.w_scale_framebuffer_size = false;
	desc.w_scale_window_size = false;
	desc.f_swap_interval = 0;
	desc.w_resolution = glm::ivec2(1);
	Window window(desc);

	LBM solver;
	demo2d::poiseuille_flow(solver);

	window.set_window_resolution(solver.get_resolution());
	primitive_renderer::set_viewport_size(solver.get_resolution());

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);

	uint32_t display_mode = 1;
	bool pause = true;

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
			case Window::Key::NUM_5:
				display_mode = 5;
				break;
			case Window::Key::NUM_6:
				display_mode = 6;
				break;
			case Window::Key::ESCAPE:
				exit(0);
				break;
			case Window::Key::SPACE:
				pause = !pause;
				break;
			}
		}
		});

	window.newsletters->on_should_close_events.subscribe([&]() {
		exit(0);
		});

	auto last_ticks_print = std::chrono::system_clock::now();
	auto update_function = [&](double deltatime) {
		if (!pause) {
			solver.iterate_time(0);

			if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_ticks_print).count() >= 1000) {
				last_ticks_print = std::chrono::system_clock::now();
				std::cout << "[LBM Info] total ticks elapsed : " << solver.get_total_ticks_elapsed() << std::endl;
			}
		}

		primitive_renderer::clear(0, 0, 0, 0);

		if (display_mode == 1) {
			solver.render2d_density();
		}
		else if (display_mode == 2) {
			solver.render2d_boundries();
		}
		else if (display_mode == 3) {
			solver.render2d_velocity();
		}
		else if (display_mode == 4) {
			solver.render2d_forces();
		}
		else if (display_mode == 5) {
			solver.render2d_temperature();
		}
		
		window.swap_buffers();
		};

	window.newsletters->on_window_resolution_events.subscribe([&](const glm::vec2& new_resolution) {
		primitive_renderer::set_viewport_size(new_resolution);
		});

	window.newsletters->on_window_refresh_events.subscribe([&]() {
		double deltatime = window.get_and_reset_deltatime();
		update_function(deltatime);
		});

	while (true) {
		double deltatime = window.handle_events(false);
		update_function(deltatime);
	}
}