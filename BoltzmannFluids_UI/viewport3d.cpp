#include "PrimitiveRenderer.h"
#include "Window.h"
#include "Debuger.h"
#include "Demos/LBMDemo3D.h"

#include "Viewport3D.h"
#include "application.h"
#include "programsourcepaths.h"

Viewport3D::EditModeProperty Viewport3D::get_EditMode_axis(EditMode edit_mode){
    switch(edit_mode){
    case TranslateX:        return X;
    case TranslateY:        return Y;
    case TranslateZ:        return Z;
    case TranslateXYZ:      return XYZ;
    case RotateX:           return X;
    case RotateY:           return Y;
    case RotateZ:           return Z;
    case RotateXYZ:         return XYZ;
    case ScaleX:            return X;
    case ScaleY:            return Y;
    case ScaleZ:            return Z;
    case ScaleXYZ:          return XYZ;
    }

    return (EditModeProperty)-1;
}

Viewport3D::EditModeProperty Viewport3D::get_EditMode_type(EditMode edit_mode){
    switch(edit_mode){
    case TranslateX:        return Translate;
    case TranslateY:        return Translate;
    case TranslateZ:        return Translate;
    case TranslateXYZ:      return Translate;
    case RotateX:           return Rotate;
    case RotateY:           return Rotate;
    case RotateZ:           return Rotate;
    case RotateXYZ:         return Rotate;
    case ScaleX:            return Scale;
    case ScaleY:            return Scale;
    case ScaleZ:            return Scale;
    case ScaleXYZ:          return Scale;
    }

    return (EditModeProperty)-1;
}

void Viewport3D::edit_cancel()
{
    is_editing = false;
    active_edit_mode = TranslateX;
    editing_amount = glm::vec2(0);
    selected_object = SimulationController::not_an_object;
    cursor_position_when_edit_begin = glm::ivec2(-128);
}

bool Viewport3D::is_edit_happening()
{
    return is_editing;
}

void Viewport3D::edit_begin(EditMode edit_mode)
{
    if (selected_object == SimulationController::not_an_object || !can_edit)
        return;
    is_editing = true;
    active_edit_mode = edit_mode;
    editing_amount = glm::vec2(0);
    cursor_position_when_edit_begin = glm::ivec2(-128);
}

void Viewport3D::edit_apply()
{
    if (selected_object == SimulationController::not_an_object || !can_edit)
        return;
    if (!is_edit_happening())
        return;

    auto& BoltzmannFluids = Application::get();
    auto& object = BoltzmannFluids.simulation->objects[selected_object];
    glm::mat4 editing_matrix = edit_compute_matrix();
    glm::mat4 composed_matrix = get_EditMode_type(active_edit_mode) == Translate ?
                                    editing_matrix * object.transform :
                                    object.transform * editing_matrix;

    object.transform = composed_matrix;

    edit_cancel();
}

glm::mat4 Viewport3D::edit_compute_matrix()
{
    glm::mat4 editing_matrix = glm::identity<glm::mat4>();
    if (is_edit_happening()){
        glm::vec3 forward = glm::normalize(get_camera_forward());
        glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        glm::vec3 up = glm::normalize(glm::cross(right, forward));

        glm::vec3 translation(0);
        glm::vec3 scale(0);
        float angle = 0;
        switch(active_edit_mode){
        case TranslateX:
            translation.x = editing_amount.x * glm::dot(right, glm::vec3(1, 0, 0)) + editing_amount.y * glm::dot(forward, glm::vec3(1, 0, 0));
            editing_matrix = glm::translate(editing_matrix, translation);
            break;
        case TranslateY:
            translation.y = editing_amount.y;
            editing_matrix = glm::translate(editing_matrix, translation);
            break;
        case TranslateZ:
            translation.z = editing_amount.x * glm::dot(right, glm::vec3(0, 0, 1)) + editing_amount.y * glm::dot(forward, glm::vec3(0, 0, 1));
            editing_matrix = glm::translate(editing_matrix, translation);
            break;
        case TranslateXYZ:
            translation.x = editing_amount.x * right.x + editing_amount.y * up.x;
            translation.y = editing_amount.y * up.y;
            translation.z = editing_amount.x * right.z + editing_amount.y * up.z;
            editing_matrix = glm::translate(editing_matrix, translation);
            break;
        case RotateX:
            angle           = -editing_amount.y * glm::dot(right, glm::vec3(1, 0, 0)) + editing_amount.x * glm::dot(forward, glm::vec3(1, 0, 0));
            editing_matrix  = glm::rotate(editing_matrix, angle, glm::vec3(1, 0, 0));
            break;
        case RotateY:
            angle           = editing_amount.x;
            editing_matrix  = glm::rotate(editing_matrix, angle, glm::vec3(0, 1, 0));
            break;
        case RotateZ:
            angle           = -editing_amount.y * glm::dot(right, glm::vec3(0, 0, 1)) + editing_amount.x * glm::dot(forward, glm::vec3(0, 0, 1));
            editing_matrix  = glm::rotate(editing_matrix, angle, glm::vec3(0, 0, 1));
            break;
        case RotateXYZ:
            angle           = -editing_amount.x + editing_amount.y;
            editing_matrix  = glm::rotate(editing_matrix, angle, glm::normalize(get_camera_position()));
            break;
        case ScaleX:
            scale.x         = editing_amount.x * glm::dot(right, glm::vec3(1, 0, 0)) + editing_amount.y * glm::dot(forward, glm::vec3(1, 0, 0));
            editing_matrix  = glm::scale(editing_matrix, glm::vec3(1) + scale);
            break;
        case ScaleY:
            scale.y         = editing_amount.y;
            editing_matrix  = glm::scale(editing_matrix, glm::vec3(1) + scale);
            break;
        case ScaleZ:
            scale.z         = editing_amount.x * glm::dot(right, glm::vec3(0, 0, 1)) + editing_amount.y * glm::dot(forward, glm::vec3(0, 0, 1));
            editing_matrix  = glm::scale(editing_matrix, glm::vec3(1) + scale);
            break;
        case ScaleXYZ:
            editing_matrix = glm::scale(editing_matrix, (1 + editing_amount.x + editing_amount.y) * glm::vec3(1));
            break;
        }
    }
    return editing_matrix;
}

Viewport3D::Viewport3D(QWidget* parent, Qt::WindowFlags f)
 : QOpenGLWidget(parent, f) {

    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setVersion(4, 6);
    format.setSamples(16);
    setFormat(format);

    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
}

void Viewport3D::initializeGL()
{
    makeCurrent();

    auto& BoltzmannFluids = Application::get();
    BoltzmannFluids.simulation = std::make_shared<SimulationController>();

    BoltzmannFluids.simulation->add_object(
        "object",
        SimulationController::Cube,
        glm::rotate(glm::identity<glm::mat4>(), 0.0f, glm::vec3(1, 0, 0))
        );

    BoltzmannFluids.simulation->lbm_solver = std::make_shared<LBM>();
    demo3d::multiphase_droplet_collision(*BoltzmannFluids.simulation->lbm_solver);

    camera.mouse_sensitivity = 0.2;

    framebuffer = std::make_shared<Framebuffer>();
    object_id_texture = std::make_shared<Texture2D>(512, 512, object_id_texture_format, 1, 0, 0);
}

void Viewport3D::resizeGL(int w, int h)
{
    primitive_renderer::set_viewport_size(glm::ivec2(w, h));
}

void Viewport3D::paintGL()
{
    primitive_renderer::clear(0.1, 0.1, 0.1, 1);

    camera.screen_width = width();
    camera.screen_height = height();
    camera.fov = 45;
    camera.position = glm::vec3(0, 0, 4);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    render_objects(camera, false);
    render_objects(camera, true);
    render_simulation(camera);

}

void Viewport3D::render_objects(Camera& camera, bool render_to_object_texture){
    compile_shaders();
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    auto& BoltzmannFluids = Application::get();

    Program& renderer = render_to_object_texture ? *renderer_flatcolor : *renderer_uv_pattern;

    int32_t bound_framebuffer;
    glm::vec4 old_viewport_dim = primitive_renderer::get_viewport_position_size();

    if (render_to_object_texture){
        glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &bound_framebuffer);
        framebuffer->attach_color(0, *object_id_texture, 0);
        framebuffer->activate_draw_buffer(0);
        framebuffer->bind_draw();
        primitive_renderer::set_viewport(glm::ivec4(0, 0, object_id_texture->get_size().x, object_id_texture->get_size().y));
        primitive_renderer::clear(0, 0, 0, 0);
        glDisable(GL_BLEND);
    }

    camera.position = get_camera_position();
    camera.update_matrixes();
    camera.update_default_uniforms(renderer);

    auto& meshes = BoltzmannFluids.simulation->imported_meshes;
    for (auto& object : BoltzmannFluids.simulation->objects){

        if (render_to_object_texture)
            renderer.update_uniform("color", glm::vec4(object.second.id));
        else{
            glm::vec4 blend_color = selected_object != object.second.id ?
                                        glm::vec4(0.3, 0.3, 0.3, 1) :
                                        glm::vec4(0.9, 0.6, 0.6, 1);

            renderer.update_uniform("blend_color", blend_color);
        }

        if (meshes.find(object.second.mesh_id) == meshes.end()){
            std::cout << "[BoltzmannFluidsUI Error] Viewport3D::render_objects() object mesh was null" << std::endl;
            ASSERT(false);
        }

        glm::mat4 editing_matrix = edit_compute_matrix();
        glm::mat4 composed_matrix = get_EditMode_type(active_edit_mode) == Translate ?
                                        editing_matrix * object.second.transform :
                                        object.second.transform * editing_matrix;

        meshes[object.second.mesh_id]->traverse([&](Mesh::Node& node, glm::mat4& matrix){
            renderer.update_uniform("model", composed_matrix);

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

    if (render_to_object_texture){
        glBindFramebuffer(GL_FRAMEBUFFER, bound_framebuffer);
        primitive_renderer::set_viewport(old_viewport_dim);
    }

}

void Viewport3D::render_simulation(Camera& camera)
{
    auto& BoltzmannFluids = Application::get();
    if (BoltzmannFluids.simulation->lbm_solver != nullptr)
        BoltzmannFluids.simulation->lbm_solver->render3d_density(camera);
}

void Viewport3D::render_axes(Camera &camera)
{

}

void Viewport3D::compile_shaders(){
    if(is_shaders_compiled)
        return;

    renderer_uv_pattern  = std::make_shared<Program>(Shader(boltzmann_fluids_ui_shaders/"basic.vert", boltzmann_fluids_ui_shaders/"texcoord_pattern.frag"));
    renderer_flatcolor  = std::make_shared<Program>(Shader(boltzmann_fluids_ui_shaders/"basic.vert", boltzmann_fluids_ui_shaders/"flatcolor.frag"));

    is_shaders_compiled = true;
}

void Viewport3D::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::RightButton) {
        if (!movement_focus)
            cursor_position_when_movement_begin = glm::ivec2(event->position().x(), event->position().y());
        movement_focus = true;

        if (is_edit_happening() && can_edit)
            edit_cancel();
    }

    if (event->button() == Qt::LeftButton && object_id_texture != nullptr) {
        if (!is_edit_happening() && can_edit){
            glm::ivec2 widget_size = glm::ivec2(size().width(), size().height());
            glm::ivec2 cursor_position = glm::ivec2(event->position().x(), event->position().y());
            cursor_position.y = widget_size.y - cursor_position.y;
            glm::ivec2 texture_size = object_id_texture->get_size();
            glm::ivec2 texture_position = glm::vec2(cursor_position) / glm::vec2(widget_size) * glm::vec2(texture_size);

            auto image = *object_id_texture->get_image(Texture2D::ColorFormat::RED, Texture2D::Type::FLOAT, 0, texture_position.x, texture_position.y, 1, 1);
            float data = *(float*)image.get_image_data();
            selected_object = data;
        }
        if (is_edit_happening() && can_edit)
            edit_apply();
    }
}

void Viewport3D::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::RightButton) {
        movement_focus = false;
    }
}

void Viewport3D::mouseDoubleClickEvent(QMouseEvent *event)
{

}

void Viewport3D::mouseMoveEvent(QMouseEvent *event)
{
    if (movement_focus) {

        glm::dvec2 position = glm::ivec2(event->position().x(), event->position().y());
        float input_rotation_x = glm::radians((float)(camera.mouse_sensitivity * -(position.y - cursor_position_when_movement_begin.y)));
        float input_rotation_y = glm::radians((float)(camera.mouse_sensitivity * -(position.x - cursor_position_when_movement_begin.x)));

        cursor_position_when_movement_begin = glm::ivec2(event->position().x(), event->position().y());;

        rotation_x += input_rotation_x;
        rotation_y += input_rotation_y;

        camera.rotation_quat = glm::quat(glm::vec3(rotation_x, 0, 0));
        camera.rotation_quat = glm::quat(glm::vec3(0, rotation_y, 0)) * camera.rotation_quat;
    }

    glm::vec3 forward_vector = (camera.rotation_quat * glm::vec3(0, 0, -1));
    camera.position = camera_origin - forward_vector * camera_distance;

    camera.update_matrixes();

    glm::ivec2 current_position = glm::ivec2(event->position().x(), event->position().y());
    glm::ivec2 widget_size = glm::ivec2(width(), height());
    if (is_edit_happening()){
        if (cursor_position_when_edit_begin.x <= 0 || cursor_position_when_edit_begin.y <= 0){
            std::cout << "here" << std::endl;
            cursor_position_when_edit_begin = current_position;
        }

        editing_amount += edit_sensitivity * glm::vec2(current_position - cursor_position_when_edit_begin) / glm::vec2(widget_size) * glm::vec2(1, -1);

        cursor_position_when_edit_begin = current_position;
        std::cout << editing_amount.x << " " << editing_amount.y << std::endl;
    }
}

void Viewport3D::leaveEvent(QEvent* event){
    if (is_edit_happening()){
        cursor_position_when_edit_begin = glm::ivec2(-128);
    }
}

void Viewport3D::wheelEvent(QWheelEvent *event)
{
    camera_distance += -event->angleDelta().y() / 120.0f * scroll_sensitivity;
}

void Viewport3D::keyPressEvent(QKeyEvent *event)
{
    switch(event->key()){
    case Qt::Key_G:
        std::cout << "G" << std::endl;
        if (is_editing && can_edit)
            edit_cancel();
        if (!is_editing && can_edit)
            edit_begin(TranslateXYZ);
        break;
    case Qt::Key_S:
        if (is_editing && can_edit)
            edit_cancel();
        if (!is_editing && can_edit)
            edit_begin(ScaleXYZ);
        break;
    case Qt::Key_R:
        if (is_editing && can_edit)
            edit_cancel();
        if (!is_editing && can_edit)
            edit_begin(RotateXYZ);
        break;
    case Qt::Key_X:
        if (is_editing && get_EditMode_type(active_edit_mode) == Translate)
            edit_begin(TranslateX);
        if (is_editing && get_EditMode_type(active_edit_mode) == Rotate)
            edit_begin(RotateX);
        if (is_editing && get_EditMode_type(active_edit_mode) == Scale)
            edit_begin(ScaleX);
        break;
    case Qt::Key_Y:
        if (is_editing && get_EditMode_type(active_edit_mode) == Translate)
            edit_begin(TranslateY);
        if (is_editing && get_EditMode_type(active_edit_mode) == Rotate)
            edit_begin(RotateY);
        if (is_editing && get_EditMode_type(active_edit_mode) == Scale)
            edit_begin(ScaleY);
        break;
    case Qt::Key_Z:
        if (is_editing && get_EditMode_type(active_edit_mode) == Translate)
            edit_begin(TranslateZ);
        if (is_editing && get_EditMode_type(active_edit_mode) == Rotate)
            edit_begin(RotateZ);
        if (is_editing && get_EditMode_type(active_edit_mode) == Scale)
            edit_begin(ScaleZ);
        break;
    case Qt::Key_Enter:
    case Qt::Key_Return:
        if (is_edit_happening() && can_edit)
            edit_apply();
        break;
    case Qt::Key_Escape:
        edit_cancel();
        break;
    }
}

glm::vec3 Viewport3D::get_camera_position()
{
    camera.position = camera_origin - get_camera_forward() * camera_distance;
    return camera.position;
}

glm::vec3 Viewport3D::get_camera_forward()
{
    camera.rotation_quat = glm::quat(glm::vec3(rotation_x, 0, 0));
    camera.rotation_quat = glm::quat(glm::vec3(0, rotation_y, 0)) * camera.rotation_quat;
    glm::vec3 forward_vector = (camera.rotation_quat * glm::vec3(0, 0, -1));
    return forward_vector;
}
