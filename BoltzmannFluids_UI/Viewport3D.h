#ifndef VIEWPORT3D_H
#define VIEWPORT3D_H

#include "Rendering/CameraController.h"
#include "Texture2D.h"
#include "Mesh.h"

#include <QOpenGLWidget>
#include <QMouseEvent>

class Window;

class Viewport3D : public QOpenGLWidget
{
public:
    Viewport3D(QWidget* parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());

protected:
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();

    void render_objects(Camera& camera, bool render_to_object_texture = false);
    void render_simulation(Camera& camera);
    void render_axes(Camera& camera);

    void compile_shaders();
    bool is_shaders_compiled = false;
    std::shared_ptr<Program> renderer_uv_pattern = nullptr;
    std::shared_ptr<Program> renderer_flatcolor = nullptr;
    std::shared_ptr<Mesh> mesh_line = nullptr;

    // object editing
    enum EditMode {
        TranslateX,
        TranslateY,
        TranslateZ,
        TranslateXYZ,
        RotateX,
        RotateY,
        RotateZ,
        RotateXYZ,
        ScaleX,
        ScaleY,
        ScaleZ,
        ScaleXYZ,
    };

    enum EditModeProperty{
        X = 1,
        Y = 2,
        Z = 3,
        XYZ = 4,
        Translate,
        Rotate,
        Scale,
    };

    static EditModeProperty get_EditMode_axis(EditMode edit_mode);
    static EditModeProperty get_EditMode_type(EditMode edit_mode);

    bool can_edit = true;
    bool is_editing = false;
    EditMode active_edit_mode = TranslateX;
    glm::vec2 editing_amount = glm::vec2(0);
    float edit_sensitivity = 3.5f;
    glm::ivec2 cursor_position_when_edit_begin = glm::ivec2(-128);
    void edit_cancel();
    bool is_edit_happening();
    void edit_begin(EditMode edit_mode);
    void edit_apply();
    glm::mat4 edit_compute_matrix();

    // object selecting
    std::shared_ptr<Framebuffer> framebuffer = nullptr;
    std::shared_ptr<Texture2D> object_id_texture = nullptr;
    Texture2D::ColorTextureFormat object_id_texture_format = Texture2D::ColorTextureFormat::R16F;
    int32_t selected_object = 0;

    // camera controller
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void mouseDoubleClickEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void leaveEvent(QEvent* event);

    virtual void wheelEvent(QWheelEvent* event);

    virtual void keyPressEvent(QKeyEvent *event);


    glm::vec3 camera_origin = glm::vec3(0);
    float camera_distance = 4;
    Camera camera;

    float scroll_sensitivity = 0.1;
    glm::vec3 get_camera_position();
    glm::vec3 get_camera_forward();

private:
    float rotation_x = 0;
    float rotation_y = 0;
    bool first_handle_movement = true;
    bool movement_focus = false;
    glm::dvec2 cursor_position_when_movement_begin;
};

#endif // VIEWPORT3D_H
