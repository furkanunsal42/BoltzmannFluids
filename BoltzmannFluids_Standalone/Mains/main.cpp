#include "GraphicsCortex.h"
#include "LBM/LBM.h"
#include "Rendering/CameraController.h"

#include "Demos/LBMDemo2D.h"
#include "Demos/LBMDemo3D.h"

#include "simulationcontroller.h"
#include "Viewport3D.h"
#include "application.h"

#include "MeshRasterizer3D/MeshRasterizer3D.h"
#include "Application/ProgramSourcePaths.h"

/*
int main() {
	lbm_shader_directory = "BoltzmannFluids/Source/GLSL/LBM/";
	renderer2d_shader_directory = "BoltzmannFluids/Source/GLSL/Renderer2D/";
	renderer3d_shader_directory = "BoltzmannFluids/Source/GLSL/Renderer3D/";
	marching_cubes_shader_directory = "BoltzmannFluids/Source/GLSL/MarchingCubes/";

	auto& BoltzmannFluids = Application::get();
	SimulationController& simulation_controller = BoltzmannFluids.simulation;
	Window& window = *simulation_controller.simulation_context;
	LBM& solver = *simulation_controller.lbm_solver;

	simulation_controller.add_object(
		"object1",
		SimulationController::Sphere
	);

	Viewport3D viewport;
	viewport.initializeGL();


	std::function<void(LBM&)> init_scenario = demo3d::multiphase_droplet_collision;

	init_scenario(*simulation_controller.lbm_solver);
	
	uint32_t display_mode = 1;
	bool pause = true;

	//window.set_window_resolution(solver.get_dimentionality() == 2 ? solver.get_resolution() : glm::ivec2(1024, 1024));
	primitive_renderer::set_viewport_size(simulation_controller.simulation_context->get_window_resolution());

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

	CameraController camera_controller;
	camera_controller.camera.screen_width = window.get_window_resolution().x;
	camera_controller.camera.screen_height = window.get_window_resolution().y;

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
		camera_controller.handle_movements(window, deltatime);
			
		viewport.paintGL();

		//switch (display_mode) {
		//case 1:
		//	if (solver.get_dimentionality() == 3)	solver.render3d_density(camera_controller.camera);
		//	else									solver.render2d_density();
		//	break;
		//case 2:
		//	if (solver.get_dimentionality() == 3)	solver.render3d_boundries(camera_controller.camera);
		//	else									solver.render2d_boundries();
		//	break;
		//case 3:
		//	if (solver.get_dimentionality() == 3)	solver.render3d_velocity(camera_controller.camera);
		//	else									solver.render2d_velocity();
		//	break;
		//case 4:
		//	if (solver.get_dimentionality() == 3)	solver.render3d_forces(camera_controller.camera);
		//	else									solver.render2d_forces();
		//	break;
		//case 5:
		//	if (solver.get_dimentionality() == 3)	solver.render3d_temperature(camera_controller.camera);
		//	else									solver.render2d_temperature();
		//	break;
		//}

		window.swap_buffers();
		};

	window.newsletters->on_window_resolution_events.subscribe([&](const glm::vec2& new_resolution) {
		primitive_renderer::set_viewport_size(new_resolution);
		camera_controller.camera.screen_width = new_resolution.x;
		camera_controller.camera.screen_height = new_resolution.y;
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
/**/


void esoteric_demo(LBM& solver) {
	glm::ivec3 simulation_resolution(128, 128, 1);
	solver.clear_boundry_properties();

	solver.is_lattice_texture3d = true;
	solver.is_collide_esoteric = true;

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

		},
		simulation_resolution,
		0.51f,
		true,
		true,
		true,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

int main() {

	lbm_shader_directory = "BoltzmannFluids/Source/GLSL/LBM/";
	renderer2d_shader_directory = "BoltzmannFluids/Source/GLSL/Renderer2D/";
	renderer3d_shader_directory = "BoltzmannFluids/Source/GLSL/Renderer3D/";
	marching_cubes_shader_directory = "BoltzmannFluids/Source/GLSL/MarchingCubes/";

	std::function<void(LBM&)> init_scenario = demo3d::multiphase_droplet_collision;

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

	CameraController camera_controller;
	camera_controller.camera.screen_width = window.get_window_resolution().x;
	camera_controller.camera.screen_height = window.get_window_resolution().y;

	auto last_ticks_print = std::chrono::system_clock::now();
	auto update_function = [&](double deltatime) {
		
		//if (!pause) {
		//	solver.iterate_time(0);
		//	
		//	auto image = solver.lattice0_tex->get_image(Texture3D::ColorFormat::RED, Texture3D::Type::FLOAT, 0);
		//	float* raw_data = (float*)image->get_image_data();
		//	
		//	std::cout << solver.get_total_ticks_elapsed() << std::endl;;
		//	for (int y = 32; y < 48; y++) {
		//		for (int x = 32; x < 48; x++) {
		//			int32_t population_id = 1;
		//			glm::ivec3 pixel_coord(x, y, 0);
		//
		//			glm::ivec3 tex3_address = glm::ivec3(solver.resolution.x * population_id, 0, 0) + pixel_coord;
		//			std::cout << std::fixed << std::setprecision(2) << raw_data[tex3_address.y * (solver.resolution.x * 9) + tex3_address.x] << " ";
		//		}
		//		std::cout << std::endl;
		//	}
		//	std::cout << std::endl;
		//
		//	for (int y = 32; y < 48; y++) {
		//		for (int x = 32; x < 48; x++) {
		//			int32_t population_id = 2;
		//			glm::ivec3 pixel_coord(x, y, 0);
		//			
		//			glm::ivec3 tex3_address = glm::ivec3(solver.resolution.x * population_id, 0, 0) + pixel_coord;
		//			std::cout << std::fixed << std::setprecision(2) << raw_data[tex3_address.y * (solver.resolution.x*9) + tex3_address.x] << " ";
		//		}
		//		std::cout << std::endl;
		//	}
		//	std::cout << std::endl;
		//	std::cout << std::endl;
		//	std::cout << std::endl;
		//	pause = true;
		//}


		
		if (!pause) {
			solver.iterate_time(0);
		
			if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_ticks_print).count() >= 1000) {
				last_ticks_print = std::chrono::system_clock::now();
				std::cout << "[LBM Info] total ticks elapsed : " << solver.get_total_ticks_elapsed() << std::endl;
			}
		}

		primitive_renderer::clear(0, 0, 0, 0);
		camera_controller.handle_movements(window, deltatime);
		
		switch (display_mode) {
		case 1:
			if (solver.get_dimentionality() == 3)	solver.render3d_density(camera_controller.camera);
			else									solver.render2d_density();
			break;
		case 2:
			if (solver.get_dimentionality() == 3)	solver.render3d_boundries(camera_controller.camera);
			else									solver.render2d_boundries();
			break;
		case 3:
			if (solver.get_dimentionality() == 3)	solver.render3d_velocity(camera_controller.camera);
			else									solver.render2d_velocity();
			break;
		case 4:
			if (solver.get_dimentionality() == 3)	solver.render3d_forces(camera_controller.camera);
			else									solver.render2d_forces();
			break;
		case 5:
			if (solver.get_dimentionality() == 3)	solver.render3d_temperature(camera_controller.camera);
			else									solver.render2d_temperature();
			break;
		}

		window.swap_buffers();
		};

	window.newsletters->on_window_resolution_events.subscribe([&](const glm::vec2& new_resolution) {
		primitive_renderer::set_viewport_size(new_resolution);
		camera_controller.camera.screen_width = new_resolution.x;
		camera_controller.camera.screen_height = new_resolution.y;
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
/**/

/*
int main() {

	std::function<void(LBM&)> init_scenario = demo3d::multiphase_droplet_collision;

	WindowDescription desc;
	desc.w_scale_framebuffer_size = false;
	desc.w_scale_window_size = false;
	desc.f_swap_interval = 2;
	desc.w_resolution = glm::ivec2(1024);
	Window window(desc);

	std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>(
		Asset("C:/dev/GraphicsCortex/GraphicsCortex/Models/teducar/source/teduCar.fbx").load_mesh()
	);
	
	Framebuffer fb;
	Texture3D texture(256, 256, 256, Texture3D::ColorTextureFormat::RGBA8, 1, 0);
	MeshRasterizer3D rasterizer;

	rasterizer.rasterize(*mesh, texture, glm::scale(glm::identity<glm::mat4>(), glm::vec3(0.005)));

	int32_t z = 0;

	while (!window.should_close()) {
		double deltatime = window.handle_events(false);
		primitive_renderer::clear(0, 0, 0, 1);

		fb.attach_color(0, texture, z, 0);
		fb.set_read_buffer(0);
		
		fb.blit_to_screen(
			glm::ivec2(texture.get_size().x, texture.get_size().y),
			window.get_window_resolution(),
			Framebuffer::Channel::COLOR,
			Framebuffer::Filter::LINEAR
		);

		z = (z + 1) % texture.get_size().z;
		window.swap_buffers();
	}
}
/**/