#ifndef APPLICATION_H
#define APPLICATION_H

#include "Window.h"
#include "simulationcontroller.h"

class Application
{
public:

    static Application& get();

    SimulationController simulation;
    //MainWindow main_window;

private:
    Application() = default;
    ~Application() = default;
};

#endif // APPLICATION_H
