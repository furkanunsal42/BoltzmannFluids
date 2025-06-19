#include "Application/ProgramSourcePaths.h"

#include "Demos/LBMDemo3D.h"

#include "application.h"
#include <QApplication>
#include <QTimer>

int main(int argc, char *argv[])
{
    lbm_shader_directory = "BoltzmannFluids/Source/GLSL/LBM/";
    renderer2d_shader_directory = "BoltzmannFluids/Source/GLSL/Renderer2D/";
    renderer3d_shader_directory = "BoltzmannFluids/Source/GLSL/Renderer3D/";
    marching_cubes_shader_directory = "BoltzmannFluids/Source/GLSL/MarchingCubes/";

    QApplication a(argc, argv);
    Application& BoltzmannFluids = Application::get();
    BoltzmannFluids.main_window.show();

    //demo3d::multiphase_droplet_collision(*BoltzmannFluids.simulation.lbm_solver);

    auto timer = new QTimer();
    int frame = 0;

    QTimer::singleShot(0, &BoltzmannFluids.main_window, [&]() {
        timer->setInterval(1000 / 60*2);
        timer->start();

        QObject::connect(timer, &QTimer::timeout, [&] {
            frame+= 1;
            BoltzmannFluids.main_window.update_timeline(frame);
        });
    });

    auto viewport_timer = new QTimer();
    QTimer::singleShot(0, &BoltzmannFluids.main_window, [&]() {
        viewport_timer->setInterval(1000 / 60.0);
        viewport_timer->start();

        QObject::connect(viewport_timer, &QTimer::timeout, [&] {
            if (BoltzmannFluids.main_window.viewport != nullptr){
                BoltzmannFluids.main_window.viewport->update();
            }



        });
    });

    return a.exec();
}
