#include "PrimitiveRenderer.h"
#include "Window.h"

#include "Viewport3D.h"
#include "application.h"
#include "programsourcepaths.h"

Viewport3D::Viewport3D() {}

void Viewport3D::initializeGL()
{

}

void Viewport3D::resizeGL(int w, int h)
{
    primitive_renderer::set_viewport_size(glm::ivec2(w, h));
}

void Viewport3D::paintGL()
{
    primitive_renderer::clear(1, 1, 1, 1);

    camera_controller.camera.screen_width = 1024;
    camera_controller.camera.screen_height = 1024;
    camera_controller.camera.fov = 45;
    camera_controller.camera.position = glm::vec3(0, 0, 4);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    render_objects(camera_controller.camera);
    //render_simulation(camera_controller.camera);

    static int32_t i = 0;
    std::cout << "Hello from Viewport3D " << i++ <<  std::endl;
}

void Viewport3D::render_objects(Camera& camera){
    compile_shaders();

    auto& BoltzmannFluids = Application::get();
    Program& renderer = *renderer_flatcolor;

    camera.update_matrixes();
    camera.update_default_uniforms(renderer);

    renderer.update_uniform("color", glm::vec4(1, 0, 0, 1));

    auto& meshes = BoltzmannFluids.simulation.imported_meshes;
    for (auto& object : BoltzmannFluids.simulation.objects){


        if (meshes.find(object.second.mesh_id) == meshes.end()){
            std::cout << "[BoltzmannFluidsUI Error] Viewport3D::render_objects() object mesh was null" << std::endl;
            ASSERT(false);
        }

        meshes[object.second.mesh_id]->traverse([&](Mesh::Node& node, glm::mat4& matrix){
            renderer.update_uniform("model", object.second.transform);

            for (auto& submesh : node.get_submeshes())
                primitive_renderer::render(
                    renderer,
                    *meshes[object.second.mesh_id]->get_mesh(submesh),
                    RenderParameters(),
                    1,
                    0
                    );
            });
    }
}

void Viewport3D::render_simulation(Camera& camera)
{
    auto& BoltzmannFluids = Application::get();



    BoltzmannFluids.simulation.lbm_solver->render3d_density(camera);
}

void Viewport3D::compile_shaders(){
    if(is_shaders_compiled)
        return;

    renderer_flatcolor = std::make_shared<Program>(Shader(boltzmann_fluids_ui_shaders/"basic.vert", boltzmann_fluids_ui_shaders/"flatcolor.frag"));

    is_shaders_compiled = true;
}
