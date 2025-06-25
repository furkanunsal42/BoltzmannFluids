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

			properties.velocity = glm::vec3(1, 0, 0) / 16.0f;

			//properties.force = glm::vec3(.1, 0, 0) / 128000.0f;

			if (coordinate.y == 0)
				properties.boundry_id = 2;
			if (coordinate.y == solver.get_resolution().y - 1)
				properties.boundry_id = 1;

		},
		simulation_resolution,
		0.53f,
		true,
		true,
		true,
		VelocitySet::D3Q19,
		FloatingPointAccuracy::fp32,
		false
	);
}

void demo3d::rayleigh_benard_convection(LBM& solver)
{
	glm::ivec3 simulation_resolution(128, 128, 32);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, 1.75);
	solver.set_boundry_properties(2, 0.25);
	solver.set_boundry_properties(3, 0.25f);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

			properties.velocity = glm::vec3(2, 0, 0) / 16.0f;
			properties.force = glm::vec3(0, -1, 0) / 128000.0f;

			if (glm::distance(glm::vec3(coordinate), glm::vec3(simulation_resolution.x * 1 / 4.0, simulation_resolution.y / 2, simulation_resolution.z / 2)) < 8) {
				properties.boundry_id = 1;
			}
		},
		simulation_resolution,
		0.515f,
		true,
		true,
		true,
		VelocitySet::D3Q15,
		FloatingPointAccuracy::fp32,
		false
	);
}

void demo3d::multiphase_humid_platform(LBM& solver)
{
	glm::ivec3 simulation_resolution(192, 384, 192);
	solver.clear_boundry_properties();

	solver.set_boundry_properties(1, LBM::referance_temperature, 4);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

			properties.force = glm::vec3(0, -8, 0) / 128000.0f;

			//properties.density = 0.056;
			properties.density = 0.356;

			if (coordinate.y == 100 && glm::abs(glm::distance(glm::vec2(coordinate.x, coordinate.z), glm::vec2(simulation_resolution.x, simulation_resolution.z) / 2.0f)) < simulation_resolution.x/10)
				properties.boundry_id = 1;
		},
		simulation_resolution,
		0.51,
		false,
		false,
		false,
		VelocitySet::D3Q15,
		FloatingPointAccuracy::fp32,
		true
	);
}

void demo3d::multiphase_droplet_collision(LBM& solver)
{
    glm::ivec3 simulation_resolution(256, 256, 256);
    solver.clear_boundry_properties();

    solver.is_lattice_texture3d = true;
    solver.velocity_limit = 0.28;
    solver.velocity_limit_extreme = 0.30;

    solver.initialize_fields(
        [&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

            properties.density = 0.056;

            if (glm::distance(glm::vec3(coordinate), glm::vec3(simulation_resolution.x * 1.2 / 4.0, simulation_resolution.y / 2, simulation_resolution.z / 2 + 10)) < 24) {
                properties.density = 2.659;
                properties.velocity = glm::vec3(24, 0, 0) / 16.0f;
            }

            if (glm::distance(glm::vec3(coordinate), glm::vec3(simulation_resolution.x * 2.8 / 4.0, simulation_resolution.y / 2, simulation_resolution.z / 2 - 10)) < 24) {
                properties.density = 2.659;
                properties.velocity = glm::vec3(-24, 0, 0) / 16.0f;
            }

        },
        simulation_resolution,
        0.51,
        true,
        true,
        true,
        VelocitySet::D3Q19,
        FloatingPointAccuracy::fp32,
        true
    );
}

void demo3d::multiphase_raindrop(LBM& solver)
{
	glm::ivec3 simulation_resolution(256, 256, 256);
	solver.clear_boundry_properties();

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

			properties.force = glm::vec3(0, -1, 0) / 1280.0f;

			properties.density = 0.056;

			if (glm::distance(glm::vec3(coordinate), glm::vec3(simulation_resolution.x * 1.2 / 4.0, simulation_resolution.y / 2, simulation_resolution.z / 2 + 10)) < 16) {
				properties.density = 2.659;
				properties.velocity = glm::vec3(16, -24, 0) / 16.0f;
			}

			//if (glm::distance(glm::vec3(coordinate), glm::vec3(simulation_resolution.x * 2.8 / 4.0, simulation_resolution.y / 2, simulation_resolution.z / 2 - 10)) < 64) {
			//	properties.density = 2.659;
			//	properties.velocity = glm::vec3(-24, 0, 0) / 16.0f;
			//}

			if (coordinate.y < solver.get_resolution().y / 4) {
				properties.density = 2.659;
			}

			//if (coordinate.y == 0) {
			//	properties.boundry_id = 1;
			//}

		},
		simulation_resolution,
		0.51,
		true,
		true,
		true,
		VelocitySet::D3Q15,
		FloatingPointAccuracy::fp32,
		true
    );
}
