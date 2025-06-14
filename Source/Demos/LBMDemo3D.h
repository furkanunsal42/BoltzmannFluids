#pragma once

#include "LBM/LBM.h"

namespace demo3d {
	void poiseuille_flow(LBM& solver);

	void rayleigh_benard_convection(LBM& solver);


	void multiphase_humid_platform(LBM& solver);
	void multiphase_droplet_collision(LBM& solver);
	void multiphase_raindrop(LBM& solver);



}