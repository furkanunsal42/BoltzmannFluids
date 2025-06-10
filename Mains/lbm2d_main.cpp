#include "GraphicsCortex.h"
#include "LBM2D/LBM2D.h"

void init_poiseuille_flow(LBM2D& solver) {
	glm::ivec2 simulation_resolution(1024, 128);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(1, 0, 0) / 16.0f);
	solver.set_boundry_properties(2, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {

			properties.boundry_id = false;

			if (coordinate.y == 10 && coordinate.x > 10 && coordinate.x < solver.get_resolution().x - 10)
				properties.boundry_id = 2;
			if (coordinate.y == solver.get_resolution().y - 11 && coordinate.x > 10 && coordinate.x < solver.get_resolution().x - 10)
				properties.boundry_id = 1;

		},
		simulation_resolution,
		0.53f,
		false,
		true,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

void init_von_karman_street_set_velocity(LBM2D& solver) {
	glm::ivec2 simulation_resolution(1024, 1024);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {

			properties.boundry_id = false;
			if (glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 1 / 4.0, simulation_resolution.y / 2)) < 32) {
				properties.boundry_id = 1;
			}

			if (coordinate.x == 0)
				properties.velocity = glm::vec3(1, 0, 0) / 16.0f;
			if (coordinate.x == solver.get_resolution().x-1)
				properties.velocity = glm::vec3(1, 0, 0) / 16.0f;
			
		},
		simulation_resolution,
		0.51f,
		false,
		true,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

void init_von_karman_street_periodic(LBM2D& solver) {
	glm::ivec2 simulation_resolution(512, 512);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {

			properties.velocity = glm::vec3(1, 0, 0) / 16.0f;

			properties.boundry_id = false;
			if (glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 1 / 4.0, simulation_resolution.y / 2)) < 32) {
				properties.boundry_id = 1;
			}
		},
		simulation_resolution,
		0.51f,
		true,
		true,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

void init_von_karman_street_inlet_boundry(LBM2D& solver) {
	glm::ivec2 simulation_resolution(1024, 1024);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);
	solver.set_boundry_properties(2, glm::vec3(1, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {

			properties.boundry_id = false;
			if (glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 1 / 4.0, simulation_resolution.y / 2)) < 32) {
				properties.boundry_id = 1;
			}

			if (coordinate.x == 0)
				properties.boundry_id = 2;
			if (coordinate.x == solver.get_resolution().x - 1)
				properties.boundry_id = 2;
		},
		simulation_resolution,
		0.60f,
		false,
		true,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

void init_von_karman_street_thin_jet(LBM2D& solver) {
	glm::ivec2 simulation_resolution(1024, 128);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {

			properties.boundry_id = false;
			if (glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 1 / 4.0, simulation_resolution.y / 2)) < 38) {
				properties.boundry_id = 1;
			}

			if (coordinate.x == 0)
				properties.velocity = glm::vec3(1, 0, 0) / 16.0f;
			if (coordinate.x == solver.get_resolution().x - 1)
				properties.velocity = glm::vec3(1, 0, 0) / 16.0f;

			if (coordinate.y == 0)
				properties.boundry_id = 1;
			if (coordinate.y == solver.get_resolution().y - 1)
				properties.boundry_id = 1;
		},
		simulation_resolution,
		0.515f,
		false,
		false,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

void init_von_karman_street_set_velocity_with_gravity(LBM2D& solver) {
	glm::ivec2 simulation_resolution(1024, 1024);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {

			properties.force = glm::vec3(0, -2, 0) / 128000.0f;

			properties.boundry_id = false;
			if (glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 1 / 4.0, simulation_resolution.y / 2)) < 32) {
				properties.boundry_id = 1;
			}

			if (coordinate.x == 0)
				properties.velocity = glm::vec3(1, 0, 0) / 16.0f;
			if (coordinate.x == solver.get_resolution().x - 1)
				properties.velocity = glm::vec3(1, 0, 0) / 16.0f;

		},
		simulation_resolution,
		0.51f,
		false,
		false,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

void init_rayleigh_benard_convection(LBM2D& solver) {
	glm::ivec2 simulation_resolution(1024, 128);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, 1.75);
	solver.set_boundry_properties(2, 0.25);
	solver.set_boundry_properties(3, 1.0f);

	solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {

			properties.force = glm::vec3(0, -4, 0) / 128000.0f;

			if (coordinate.x == 0)
				properties.boundry_id = 3;
			if (coordinate.x == solver.get_resolution().x - 1)
				properties.boundry_id = 3;
			if (coordinate.y == 0)
				properties.boundry_id = 1;
			if (coordinate.y == solver.get_resolution().y - 1)
				properties.boundry_id = 2;
		},
		simulation_resolution,
		0.515f,
		false,
		false,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

void init_thermal_convection_tall(LBM2D& solver) {
	glm::ivec2 simulation_resolution(512, 1024);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, 1.75);
	solver.set_boundry_properties(2, 0.1);
	solver.set_boundry_properties(3, 0.6f);

	solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {

			properties.force = glm::vec3(0, -2, 0) / 128000.0f;

			if (coordinate.x == 0)
				properties.boundry_id = 3;
			if (coordinate.x == solver.get_resolution().x - 1)
				properties.boundry_id = 3;
			if (coordinate.y == 0)
				properties.boundry_id = 1;
			if (coordinate.y == solver.get_resolution().y - 1)
				properties.boundry_id = 2;
		},
		simulation_resolution,
		0.515f,
		false,
		false,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

void init_thermal_convection_square(LBM2D& solver) {
	glm::ivec2 simulation_resolution(512, 512);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, 1.75);
	solver.set_boundry_properties(2, 0.1);
	solver.set_boundry_properties(3, 1.f);

	solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {

			properties.force = glm::vec3(0, -2, 0) / 128000.0f;
			properties.density = 1.4;

			if (coordinate.x == 0)
				properties.boundry_id = 3;
			if (coordinate.x == solver.get_resolution().x - 1)
				properties.boundry_id = 3;
			if (coordinate.y == 0)
				properties.boundry_id = 1;
			if (coordinate.y == solver.get_resolution().y - 1)
				properties.boundry_id = 2;
		},
		simulation_resolution,
		0.515f,
		false,
		false,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		true
	);
}

void init_multiphase_droplet(LBM2D& solver) {
	glm::ivec2 simulation_resolution(1024, 1024);
	solver.clear_boundry_properties();

	solver.initialize_fields(
		[&](glm::ivec2 coordinate, LBM2D::FluidProperties& properties) {

			properties.force = glm::vec3(0, -1, 0) / 128000.0f;

			//properties.density = 0.056;
			properties.density = 0.156;
			//properties.density = 0.20;

			//if (glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 1 / 4.0, simulation_resolution.y / 2)) < 32 ) {
			//	properties.density = 2.659;
			//	//properties.velocity = glm::vec3(1, 0, 0) / 16.0f;
			//}

			//if (coordinate.x == 0)
			//	properties.boundry_id = 1;
			//if (coordinate.x == solver.get_resolution().x - 1)
			//	properties.boundry_id = 1;
			//if (coordinate.y == 0)
			//	properties.boundry_id = 1;
			//if (coordinate.y == solver.get_resolution().y - 1)
			//	properties.boundry_id = 1;
			if (coordinate.y == 0 && glm::abs(coordinate.x - 512) < 300)
				properties.boundry_id = 1;
			//if (glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 2.1 / 4.0, simulation_resolution.y / 2)) < 16) {
			//	//properties.density = 1.85;
			//	properties.density = 1.89;
			//	properties.velocity = glm::vec3(-1, 0, 0) / 16.0f;
			//}
		},
		glm::ivec2(simulation_resolution),
		0.9,
		true,
		true,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		true
	);
}

int main() {

	WindowDescription desc;
	desc.w_scale_framebuffer_size = false;
	desc.w_scale_window_size = false;
	desc.f_swap_interval = 0;
	desc.w_resolution = glm::ivec2(1);
	Window window(desc);

	LBM2D lbm2d_solver;
	init_multiphase_droplet(lbm2d_solver);
	window.set_window_resolution(lbm2d_solver.get_resolution());

	Texture2D texture_1c(window.get_window_resolution().x, window.get_window_resolution().y, Texture2D::ColorTextureFormat::R32F, 1, 0, 0);
	Texture2D texture_4c(window.get_window_resolution().x, window.get_window_resolution().y, Texture2D::ColorTextureFormat::RGBA32F, 1, 0, 0);

	Framebuffer fb;
	fb.attach_color(0, texture_1c, 0);
	fb.activate_draw_buffer(0);

	uint32_t display_mode = 3;
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

	auto update_function = [&](double deltatime) {
		if(!pause)
			lbm2d_solver.iterate_time(std::chrono::duration<double, std::milli>(deltatime * 100));

		if (display_mode == 1) {
			Texture2D& texture_target = texture_1c;
			lbm2d_solver.copy_to_texture_density(texture_target);
			fb.attach_color(0, texture_target, 0);
		}
		else if (display_mode == 2) {
			Texture2D& texture_target = texture_1c;
			lbm2d_solver.copy_to_texture_velocity_magnetude(texture_target);
			fb.attach_color(0, texture_target, 0);
		}
		else if (display_mode == 3) {
			Texture2D& texture_target = texture_4c;
			lbm2d_solver.copy_to_texture_velocity_vector(texture_target);
			fb.attach_color(0, texture_target, 0);
		}
		else if (display_mode == 4) {
			Texture2D& texture_target = texture_4c;
			lbm2d_solver.copy_to_texture_boundries(texture_target);
			fb.attach_color(0, texture_target, 0);
		}
		else if (display_mode == 5) {
			Texture2D& texture_target = texture_4c;
			lbm2d_solver.copy_to_texture_force_vector(texture_target);
			fb.attach_color(0, texture_target, 0);
		}
		else if (display_mode == 6) {
			Texture2D& texture_target = texture_1c;
			lbm2d_solver.copy_to_texture_temperature(texture_target);
			fb.attach_color(0, texture_target, 0);
		}

		fb.blit_to_screen(window.get_window_resolution(), window.get_framebuffer_resolution(), Framebuffer::Channel::COLOR, Framebuffer::Filter::NEAREST);
		window.swap_buffers();
		};

	window.newsletters->on_window_refresh_events.subscribe([&]() {
		double deltatime = window.get_and_reset_deltatime();
		update_function(deltatime);
		});

	while (true) {
		double deltatime = window.handle_events(true);
		update_function(deltatime);
	}
}