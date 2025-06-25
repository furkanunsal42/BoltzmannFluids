#include "LBM/LBM.h"
#include "Demos/LBMDemo3D.h"
#include "Demos/LBMDemo2D.h"

#include "SimulationController.h"
#include "ProgramSourcePaths.h"
#include "Application.h"
#include "GUI/InitialConditionsBox.h"
#include "GUI/SmartDoubleSpinBox.h"

#include <QComboBox>
#include <QTimer>
#include <QCheckBox>

SimulationController::SimulationController()
{
    gl_context = Window::create_from_current();

    add_mesh("BoltzmannFluids_UI/Models/UnitCube.fbx");
    add_mesh("BoltzmannFluids_UI/Models/UnitSphere.fbx");
    add_mesh("BoltzmannFluids_UI/Models/UnitCylinder.fbx");
}

SimulationController::~SimulationController()
{
    //exit(0); // temp, it is to not close the program with a crash due to improper OpenGL context release sequence
    lbm_solver = nullptr;
    gl_context = nullptr;
}

void SimulationController::start_simulation()
{
    if (lbm_solver == nullptr){
        lbm_solver = std::make_shared<LBM>();


        //demo2d::poiseuille_flow(*lbm_solver);
        //demo2d::von_karman_street_periodic(*lbm_solver);
        //demo2d::rayleigh_benard_convection(*lbm_solver);

        demo3d::multiphase_droplet_collision(*lbm_solver);

        Application& BoltzmannFluids = Application::get();
    }
}

void SimulationController::iterate_time(float target_tick_per_seconds)
{
    if (lbm_solver != nullptr){
        lbm_solver->iterate_time(target_tick_per_seconds);
        if (lbm_solver->get_total_ticks_elapsed() % 100 == 0)
            std::cout << "[LBM Info] total ticks elapsed : " << lbm_solver->get_total_ticks_elapsed() << std::endl;

        Application& BoltzmannFluids = Application::get();
        BoltzmannFluids.main_window.update_timeline(lbm_solver->get_total_ticks_elapsed());

        if (!is_first_iteration_happened){            
            if (BoltzmannFluids.main_window.viewport->is_edit_happening()){
                BoltzmannFluids.main_window.viewport->edit_cancel();
            }
            BoltzmannFluids.main_window.viewport->can_edit = false;
        }
        is_first_iteration_happened = true;
    }
}

void SimulationController::initialize_lbm_from_panel(LBM &lbm)
{
    if (lbm_solver == nullptr)
        return;

    Application& BoltzmannFluids = Application::get();
    auto panel = BoltzmannFluids.main_window.initial_conditions;

    VelocitySet velocity_set =
        panel->velocity_set->currentText() == "D2Q9"  ? VelocitySet::D2Q9 :
        panel->velocity_set->currentText() == "D3Q15" ? VelocitySet::D3Q15 :
        panel->velocity_set->currentText() == "D3Q19" ? VelocitySet::D3Q19 :
        panel->velocity_set->currentText() == "D3Q27" ? VelocitySet::D3Q27 : VelocitySet::D2Q9;

    FloatingPointAccuracy floating_point_accuracy =
        panel->floating_point_accuracy->currentText() == "16-Bit" ? FloatingPointAccuracy::fp16 :
        panel->floating_point_accuracy->currentText() == "32-Bit" ? FloatingPointAccuracy::fp32 : FloatingPointAccuracy::fp16;

    glm::ivec3 resolution = glm::ivec3(
        panel->resolution_X_value->value(),
        panel->resolution_Y_value->value(),
        panel->resolution_Z_value->value()
        );

    glm::vec3 gravity = glm::vec3(
        panel->gravity_X_value->value(),
        panel->gravity_Y_value->value(),
        panel->gravity_Z_value->value()
        );

    glm::vec3 initial_velocity = glm::vec3(
        panel->initial_velocity_X_value->value(),
        panel->initial_velocity_Y_value->value(),
        panel->initial_velocity_Z_value->value()
        );

    float relaxation_time = panel->relaxation_time->value();
    float initial_temperature = panel->initial_temprature_value->value();
    float thermal_relaxation_time = panel->thermal_relaxation_time->value();
    float thermal_extension_coefficient = panel->thermal_extension_coefficient->value();

    bool is_multiphase = panel->multiphase_checkbox->isChecked();
    float intermolacular_interaction_stength = panel->intermoleculer_interaction_strength->value();

    bool periodic_x = panel->boundary_X->currentText() == "Periodic Wall";
    bool periodic_y = panel->boundary_Y->currentText() == "Periodic Wall";
    bool periodic_z = panel->boundary_Z->currentText() == "Periodic Wall";

    bool wall_x = panel->boundary_X->currentText() == "Solid Wall";
    bool wall_y = panel->boundary_Y->currentText() == "Solid Wall";
    bool wall_z = panel->boundary_Z->currentText() == "Solid Wall";

    float boundry_x_temperature = panel->boundary_X_temprature->value();
    float boundry_x_effecive_density  = panel->boundary_X_effective_density->value();
    glm::vec3 boundry_x_translational_velocity =
        glm::vec3(
        panel->boundary_X_velocity_translation_X_box->value(),
        panel->boundary_X_velocity_translation_Y_box->value(),
        panel->boundary_X_velocity_translation_Z_box->value()
        );

    float boundry_y_temperature = panel->boundary_Y_temprature->value();
    float boundry_y_effecive_density  = panel->boundary_Y_effective_density->value();
    glm::vec3 boundry_y_translational_velocity =
        glm::vec3(
            panel->boundary_Y_velocity_translation_X_box->value(),
            panel->boundary_Y_velocity_translation_Y_box->value(),
            panel->boundary_Y_velocity_translation_Z_box->value()
            );

    float boundry_z_temperature = panel->boundary_Z_temprature->value();
    float boundry_z_effecive_density  = panel->boundary_Z_effective_density->value();
    glm::vec3 boundry_z_translational_velocity =
        glm::vec3(
            panel->boundary_Z_velocity_translation_X_box->value(),
            panel->boundary_Z_velocity_translation_Y_box->value(),
            panel->boundary_Z_velocity_translation_Z_box->value()
            );

    lbm_solver->clear_boundry_properties();

    lbm_solver->set_boundry_properties(1, boundry_x_translational_velocity, boundry_x_temperature, boundry_x_effecive_density);
    lbm_solver->set_boundry_properties(2, boundry_y_translational_velocity, boundry_y_temperature, boundry_y_effecive_density);
    lbm_solver->set_boundry_properties(3, boundry_z_translational_velocity, boundry_z_temperature, boundry_z_effecive_density);

    lbm_solver->initialize_fields(
        [&](glm::ivec3 coordinate, LBM::FluidProperties& properties) {

            properties.force = gravity;
            properties.density = 1;
            properties.density = 1;

            properties.density = 0.056;

            if (glm::distance(glm::vec3(coordinate), glm::vec3(resolution.x * 1.2 / 4.0, resolution.y / 2, resolution.z / 2 + 10)) < 24) {
                properties.density = 2.659;
                properties.velocity = glm::vec3(24, 0, 0) / 16.0f;
            }

            if (glm::distance(glm::vec3(coordinate), glm::vec3(resolution.x * 2.8 / 4.0, resolution.y / 2, resolution.z / 2 - 10)) < 24) {
                properties.density = 2.659;
                properties.velocity = glm::vec3(-24, 0, 0) / 16.0f;
            }

            if (coordinate.x == 0 && wall_x)
                properties.boundry_id = 1;
            if (coordinate.x == lbm_solver->get_resolution().x - 1 && wall_x)
                properties.boundry_id = 1;

            if (coordinate.y == 0 && wall_y)
                properties.boundry_id = 2;
            if (coordinate.y == lbm_solver->get_resolution().y - 1 && wall_y)
                properties.boundry_id = 2;

            if (coordinate.z == 0 && wall_z)
                properties.boundry_id = 3;
            if (coordinate.z == lbm_solver->get_resolution().z - 1 && wall_z)
                properties.boundry_id = 3;

        },
        resolution,
        relaxation_time,
        periodic_x,
        periodic_y,
        periodic_z,
        velocity_set,
        floating_point_accuracy,
        is_multiphase
        );
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
