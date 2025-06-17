#pragma once
#include "PhysicsContext.h"
#include "PhysicsLink.h"
#include <glm.hpp>

#include <vector>

class PhysicsLink;

class PhysicsObject {
public:
	physx::PxMaterial* material;
	physx::PxShape* shape;
	physx::PxTransform transform;
	physx::PxRigidActor* actor;
	bool exclusive_shape = false;

	unsigned int type;

	std::vector<PhysicsLink*> links;

	enum type {
		DYNAMIC = 0,
		KINEMATIC,
		STATIC,
	};

	PhysicsObject(const physx::PxGeometry& geometry, enum type type = type::DYNAMIC, bool exclusive_shape = false);
	PhysicsObject(const physx::PxPlane& plane, bool exclusive_shape = false);

	void set_type(unsigned int new_type);
	void create_shape(const physx::PxGeometry& geometry, const physx::PxMaterial& material, bool exclusive_shape = false);
	void set_position(float x, float y, float z);
	
	template<typename T>
	std::enable_if_t<std::is_same<T, glm::vec3>::value || std::is_same<T, physx::PxVec3>::value, void>
	set_position(T rotation_vector) {
		set_position(rotation_vector.x, rotation_vector.y, rotation_vector.z);
	}
	
	void set_rotation(float x, float y, float z);

	template<typename T>
	std::enable_if_t<std::is_same<T, glm::vec3>::value || std::is_same<T, physx::PxVec3>::value, void>
	set_rotation(T rotation_vector) {
		set_rotation(rotation_vector.x, rotation_vector.y, rotation_vector.z);
	}

	void update_transform();
	PhysicsLink* add_link(PhysicsObject& other, unsigned int link_type = 0); // PhysicsLink::type insted of unsigned int
	void remove_link(PhysicsLink* link);

	physx::PxVec3 get_position();
	physx::PxQuat get_rotation();

	void set_gravity(bool enable_gravity);

	void make_drivable();
private:
	glm::vec3 position;
	glm::vec3 rotation;
};