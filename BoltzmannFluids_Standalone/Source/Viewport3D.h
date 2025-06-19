#ifndef VIEWPORT3D_H
#define VIEWPORT3D_H

#include "Rendering/CameraController.h"

class Window;

class Viewport3D
{
public:
    Viewport3D();

public:
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();

    void render_objects(Camera& camera);
    void render_simulation(Camera& camera);

    CameraController camera_controller;

    void compile_shaders();
    bool is_shaders_compiled = false;
    std::shared_ptr<Program> renderer_flatcolor = nullptr;
};

#endif // VIEWPORT3D_H
