#include "GraphicsCortex.h"
#include "LBM2D/LBM2D.h"

int main() {

	WindowDescription desc;
	desc.w_resolution = glm::ivec2(512, 512);
	Window window(desc);

	LBM2D lbm2d_solver;

	lbm2d_solver.set_velocity_set(VelocitySet::D2Q9);
	lbm2d_solver.set_floating_point_accuracy(FloatingPointAccuracy::fp32);
	lbm2d_solver.compile_shaders();
	
	lbm2d_solver.generate_lattice(glm::ivec2(512, 512), glm::vec2(1));
	lbm2d_solver.iterate_time(16.6);

	Texture2D texture(512, 512, Texture2D::ColorTextureFormat::R32F, 1, 0, 0);
	Framebuffer fb;
	fb.attach_color(0, texture, 0);
	fb.activate_draw_buffer(0);

	lbm2d_solver.copy_to_texture_velocity_index(texture, 2);

	while (!window.should_close()) {
		double deltatime = window.handle_events(true);

		fb.blit_to_screen(glm::ivec2(512, 512), glm::ivec2(512, 512), Framebuffer::Channel::COLOR, Framebuffer::Filter::LINEAR);

		window.swap_buffers();
	}

}