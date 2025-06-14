#include "GraphicsCortex.h"
#include "LBM/LBM.h"

#include "Demos/LBMDemo2D.h"
#include "Demos/LBMDemo3D.h"

int main() {

	std::function<void(LBM&)> init_scenario = demo2d::von_karman_street_periodic;

	WindowDescription desc;
	desc.w_scale_framebuffer_size = false;
	desc.w_scale_window_size = false;
	desc.f_swap_interval = 0;
	desc.w_resolution = glm::ivec2(1);
	Window window(desc);

	LBM solver;
	init_scenario(solver);

	window.set_window_resolution(solver.get_dimentionality() == 2 ? solver.get_resolution() : glm::ivec2(1024, 1024));
	primitive_renderer::set_viewport_size(window.get_window_resolution());

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
			case Window::Key::TAB:
				pause = !pause;
				break;
			case Window::Key::R:
				solver = LBM();
				init_scenario(solver);
				break;
			}
		}
		});

	window.newsletters->on_should_close_events.subscribe([&]() {
		exit(0);
		});

	Camera camera_3d;
	camera_3d.screen_width = window.get_window_resolution().x;
	camera_3d.screen_height = window.get_window_resolution().y;

	auto last_ticks_print = std::chrono::system_clock::now();
	auto update_function = [&](double deltatime) {
		if (!pause) {
			solver.iterate_time(50);

			if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_ticks_print).count() >= 1000) {
				last_ticks_print = std::chrono::system_clock::now();
				std::cout << "[LBM Info] total ticks elapsed : " << solver.get_total_ticks_elapsed() << std::endl;
			}
		}

		primitive_renderer::clear(0, 0, 0, 0);
		camera_3d.handle_movements((GLFWwindow*)window.get_handle(), deltatime);

		switch (display_mode) {
		case 1:
			if (solver.get_dimentionality() == 3)	solver.render3d_density(camera_3d);
			else									solver.render2d_density();
			break;
		case 2:
			if (solver.get_dimentionality() == 3)	solver.render3d_boundries(camera_3d);
			else									solver.render2d_boundries();
			break;
		case 3:
			if (solver.get_dimentionality() == 3)	solver.render3d_velocity(camera_3d);
			else									solver.render2d_velocity();
			break;
		case 4:
			if (solver.get_dimentionality() == 3)	solver.render3d_forces(camera_3d);
			else									solver.render2d_forces();
			break;
		case 5:
			if (solver.get_dimentionality() == 3)	solver.render3d_temperature(camera_3d);
			else									solver.render2d_temperature();
			break;
		}

		window.swap_buffers();
		};

	window.newsletters->on_window_resolution_events.subscribe([&](const glm::vec2& new_resolution) {
		primitive_renderer::set_viewport_size(new_resolution);
		camera_3d.screen_width = new_resolution.x;
		camera_3d.screen_height = new_resolution.y;
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