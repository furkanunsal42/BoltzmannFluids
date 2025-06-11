#pragma once

#include "LBM/LBM.h"

namespace demo2d {
	void poiseuille_flow(LBM& solver);

	void von_karman_street_set_velocity(LBM& solver);
	void von_karman_street_periodic(LBM& solver);
	void von_karman_street_inlet_boundry(LBM& solver);
	void von_karman_street_thin_jet(LBM& solver);
	void von_karman_street_set_velocity_with_gravity(LBM& solver);

	void rayleigh_benard_convection(LBM& solver);
	void thermal_convection_tall(LBM& solver);

	void multiphase_thermal_boiling(LBM& solver);
	void multiphase_humid_platform(LBM& solver);
	void multiphase_droplet_collision(LBM& solver);
}
