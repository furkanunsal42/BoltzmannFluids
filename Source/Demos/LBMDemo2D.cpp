#include "LBMDemo2D.h"

void demo2d::poiseuille_flow(LBM& solver) {
	glm::ivec3 simulation_resolution(1024, 128, 32);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(1, 0, 0) / 16.0f);
	solver.set_boundry_properties(2, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

			properties.boundry_id = false;

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
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

void demo2d::von_karman_street_set_velocity(LBM& solver) {
	glm::ivec3 simulation_resolution(1024, 1024, 1);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

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
		true,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		false
	);
}

void demo2d::von_karman_street_periodic(LBM& solver) {
	glm::ivec3 simulation_resolution(512, 512, 1);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

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

void demo2d::von_karman_street_inlet_boundry(LBM& solver) {
	glm::ivec3 simulation_resolution(1024, 1024, 1);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);
	solver.set_boundry_properties(2, glm::vec3(1, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

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

void demo2d::von_karman_street_thin_jet(LBM& solver) {
	glm::ivec3 simulation_resolution(1024, 128, 1);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

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

void demo2d::von_karman_street_set_velocity_with_gravity(LBM& solver) {
	glm::ivec3 simulation_resolution(1024, 1024, 1);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, glm::vec3(0, 0, 0) / 16.0f);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

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

void demo2d::rayleigh_benard_convection(LBM& solver) {
	glm::ivec3 simulation_resolution(1024, 128, 1);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, 1.75);
	solver.set_boundry_properties(2, 0.25);
	solver.set_boundry_properties(3, 1.0f);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

			properties.force = glm::vec3(0, -8, 0) / 128000.0f;

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

void demo2d::thermal_convection_tall(LBM& solver) {
	glm::ivec3 simulation_resolution(512, 1024, 1);
	solver.clear_boundry_properties();
	solver.set_boundry_properties(1, 1.75);
	solver.set_boundry_properties(2, 0.1);
	solver.set_boundry_properties(3, 0.6f);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

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

void demo2d::multiphase_thermal_boiling(LBM& solver) {
	glm::ivec3 simulation_resolution(512, 512, 1);
	solver.clear_boundry_properties();

	solver.set_boundry_properties(1, 1, 2.6);
	solver.set_boundry_properties(2, 1, 2.6);
	solver.set_boundry_properties(3, 1, 2.6);
	solver.set_intermolecular_interaction_strength(-6);
	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

			properties.force = glm::vec3(0, -16, 0) / 128000.0f;

			//properties.temperature = 1;
			//if (abs(coordinate.x - solver.get_resolution().x / 2) < 100 && abs(coordinate.y - solver.get_resolution().y / 2) < 100)
			//	properties.density = 2.659;
			properties.density = 0.056;
			properties.temperature = 0.0056;
			if (coordinate.y < 100) {
				properties.density = 2.659;
				properties.temperature = 0.2659;
			}

			if (coordinate.x == 0)
				properties.boundry_id = 3;
			if (coordinate.x == solver.get_resolution().x - 1)
				properties.boundry_id = 3;
			if (coordinate.y == 0 && abs(coordinate.x - solver.get_resolution().x / 2) >= 0)
				properties.boundry_id = 1;
			if (coordinate.y == solver.get_resolution().y - 1 && abs(coordinate.x - solver.get_resolution().x / 2) >= 0)
				properties.boundry_id = 2;
		},
		simulation_resolution,
		0.60,
		false,
		false,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		true
	);
}

void demo2d::multiphase_humid_platform(LBM& solver) {
	glm::ivec3 simulation_resolution(1024, 1024, 1);
	solver.clear_boundry_properties();

	solver.set_boundry_properties(1, LBM::referance_temperature, 4);

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

			properties.force = glm::vec3(0, -8, 0) / 128000.0f;

			//properties.density = 0.056;
			properties.density = 0.356;

			if (coordinate.y == 100 && glm::abs(coordinate.x - 512) < 300)
				properties.boundry_id = 1;
		},
		simulation_resolution,
		0.51,
		false,
		false,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		true
	);
}

void demo2d::multiphase_droplet_collision(LBM& solver) {
	glm::ivec3 simulation_resolution(1024, 1024, 1);
	solver.clear_boundry_properties();

	solver.initialize_fields(
		[&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

			properties.density = 0.056;

			if (glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 1.5 / 4.0, simulation_resolution.y / 2)) < 32) {
				properties.density = 2.659;
				properties.velocity = glm::vec3(16, 0, 0) / 16.0f;
			}

			if (glm::distance(glm::vec2(coordinate), glm::vec2(simulation_resolution.x * 2.5 / 4.0, simulation_resolution.y / 2)) < 32) {
				properties.density = 2.659;
				properties.velocity = glm::vec3(-16, 0, 0) / 16.0f;
			}

		},
		simulation_resolution,
		0.52,
		true,
		true,
		VelocitySet::D2Q9,
		FloatingPointAccuracy::fp32,
		true
	);
}