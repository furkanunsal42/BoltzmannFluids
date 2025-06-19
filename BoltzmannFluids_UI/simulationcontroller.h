#ifndef SIMULATIONCONTROLLER_H
#define SIMULATIONCONTROLLER_H

#include "LBM/LBM.h"
#include "GraphicsCortex.h"

class SimulationController
{
public:
    SimulationController();
    ~SimulationController();

    constexpr static int32_t not_an_object = 0;
    constexpr static int32_t not_a_mesh = 0;

    // solvers
    std::shared_ptr<LBM> lbm_solver = nullptr;
    std::shared_ptr<Window> gl_context = nullptr;

    // simulation modes
    enum SimulationMode {
        PreSimulation,
        RealtimeSimulation,
        SavingSimulation,
        PlaybackSimulation,
    };

    SimulationMode simulation_mode = PreSimulation;

    // objects
    enum BasicObject {
        Cube        = 1,
        Sphere      = 2,
        Cylinder    = 3,
    };

    int32_t add_mesh(std::shared_ptr<Mesh> mesh);
    int32_t add_mesh(const std::filesystem::path& mesh_path);

    int32_t add_object(
        std::string name = "Object",
        BasicObject basic_object = Cube,
        glm::mat4 transform = glm::identity<glm::mat4>(),
        glm::vec3 velocity_translational = glm::vec3(0),
        glm::vec3 velocity_angular = glm::vec3(0),
        glm::vec3 center_of_mass = glm::vec3(0),
        float temperature = LBM::referance_temperature,
        float effective_density = LBM::referance_boundry_density
        );

    int32_t add_object(
        std::string name,
        int32_t mesh_id,
        glm::mat4 transform = glm::identity<glm::mat4>(),
        glm::vec3 velocity_translational = glm::vec3(0),
        glm::vec3 velocity_angular = glm::vec3(0),
        glm::vec3 center_of_mass = glm::vec3(0),
        float temperature = LBM::referance_temperature,
        float effective_density = LBM::referance_boundry_density
        );

    struct Object {
        int32_t id = 0;
        std::string name = "Object";
        bool is_object_basic = true;
        BasicObject basic_object_type = Cube;
        int32_t mesh_id = not_a_mesh;
        glm::mat4 transform;

        glm::vec3 velocity_translational = glm::vec3(0);
        glm::vec3 velocity_angular = glm::vec3(0);
        glm::vec3 center_of_mass = glm::vec3(0);
        float temperature = LBM::referance_temperature;
        float effective_density = LBM::referance_boundry_density;
    };

    std::unordered_map<int32_t, Object> objects;
    std::unordered_map<int32_t, std::shared_ptr<Mesh>> imported_meshes;

private:

    void initialize();

    int32_t next_object_id = 1;
    int32_t next_mesh_id = 1;
    int32_t generate_next_object_id();
    int32_t generate_next_mesh_id();

    std::shared_ptr<Mesh> mesh_cube = nullptr;
    std::shared_ptr<Mesh> mesh_sphere = nullptr;
    std::shared_ptr<Mesh> mesh_cylinder = nullptr;
};

#endif // SIMULATIONCONTROLLER_H
