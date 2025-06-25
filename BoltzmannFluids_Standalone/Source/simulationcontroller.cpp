#include "simulationcontroller.h"
#include "programsourcepaths.h"

SimulationController::SimulationController()
{
    simulation_context = std::make_shared<Window>(glm::ivec2(1024, 1024), "GraphicsCortexGL");
    simulation_context->context_make_current();
    lbm_solver = std::make_shared<LBM>();

    add_mesh("BoltzmannFluids_UI/Models/UnitCube.fbx");
    add_mesh("BoltzmannFluids_UI/Models/UnitSphere.fbx");
    add_mesh("BoltzmannFluids_UI/Models/UnitCylinder.fbx");
}

SimulationController::~SimulationController()
{
    exit(0); // temp, it is to not close the program with a crash due to improper OpenGL context release sequence
    lbm_solver = nullptr;
    simulation_context = nullptr;
}

int32_t SimulationController::generate_next_object_id(){
    int32_t id = next_object_id;
    next_object_id++;
    return id;
}

int32_t SimulationController::generate_next_mesh_id(){
    int32_t id = next_mesh_id;
    next_mesh_id++;
    return id;
}

int32_t SimulationController::add_mesh(std::shared_ptr<Mesh> mesh){
    int32_t id = generate_next_mesh_id();
    imported_meshes[id] = mesh;
    return id;
}

int32_t SimulationController::add_mesh(const std::filesystem::path& mesh_path){
    Asset asset(mesh_path);
    std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>(asset.load_mesh());
    return add_mesh(mesh);
}

int32_t SimulationController::add_object(
    std::string name,
    BasicObject basic_object,
    glm::mat4 transform,
    glm::vec3 velocity_translational,
    glm::vec3 velocity_angular,
    glm::vec3 center_of_mass,
    float temperature,
    float effective_density
    )
{
    Object object;

    object.id = generate_next_object_id();
    object.name = name;
    object.is_object_basic = true;
    object.basic_object_type = basic_object;
    object.mesh_id = (int32_t)basic_object;

    object.transform = transform;
    object.velocity_translational = velocity_translational;
    object.velocity_angular = velocity_angular;
    object.center_of_mass = center_of_mass;
    object.temperature = temperature;
    object.effective_density = effective_density;

    objects[object.id] = object;
    return object.id;
}

int32_t SimulationController::add_object(
    std::string name,
    int32_t mesh_id,
    glm::mat4 transform,
    glm::vec3 velocity_translational,
    glm::vec3 velocity_angular,
    glm::vec3 center_of_mass,
    float temperature,
    float effective_density
    )
{
    Object object;

    object.id = generate_next_object_id();
    object.name = name;
    object.is_object_basic = false;
    object.basic_object_type = Cube;
    object.mesh_id = mesh_id;

    object.transform = transform;
    object.velocity_translational = velocity_translational;
    object.velocity_angular = velocity_angular;
    object.center_of_mass = center_of_mass;
    object.temperature = temperature;
    object.effective_density = effective_density;

    objects[object.id] = object;
    return object.id;
}
