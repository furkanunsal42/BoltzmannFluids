#include "LBMDemo3D.h"

void demo3d::poiseuille_flow(LBM& solver)
{
	glm::ivec3 simulation_resolution(256, 128, 32);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);
	solver.set_boundry_properties(2, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

			properties.boundry_id = false;

			properties.velocity = glm::vec3(1, 0, 0) / 1024.0f;

			//properties.force = glm::vec3(.1, 0, 0) / 128000.0f;

			if (coordinate.y == 0)
				properties.boundry_id = 2;
			if (coordinate.y == solver.get_resolution().y - 1)
				properties.boundry_id = 1;

		},
		simulation_resolution,
		0.53f,
		true,
		false,
		VelocitySet::D3Q19,
		FloatingPointAccuracy::fp32,
		false
	);
}

void demo3d::multiphase_droplet_collision(LBM& solver)
{
	glm::ivec3 simulation_resolution(256, 256, 256);
	solver.clear_boundry_properties();

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

			properties.density = 0.056;

			if (glm::distance(glm::vec3(coordinate), glm::vec3(simulation_resolution.x * 2 / 4.0, simulation_resolution.y / 2, simulation_resolution.z / 2)) < 32) {
				properties.density = 2.659;
				//properties.velocity = glm::vec3(16, 0, 0) / 16.0f;
			}

			//if (glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 2.5 / 4.0, simulation_resolution.y / 2)) < 32) {
			//	properties.density = 2.659;
			//	properties.velocity = glm::vec3(-16, 0, 0) / 16.0f;
			//}

		},
		simulation_resolution,
		0.55,
		true,
		true,
		VelocitySet::D3Q27,
		FloatingPointAccuracy::fp32,
		true
	);
}
