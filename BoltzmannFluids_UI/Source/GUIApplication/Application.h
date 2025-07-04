#ifndef APPLICATION_H
#define APPLICATION_H

#include "Window.h"
#include "SimulationController.h"
#include "GUI/MainWindow.h"

class Application
{
public:

    static Application& get();

    std::shared_ptr<SimulationController> simulation = nullptr;
    MainWindow main_window;

private:
    Application() = default;
    ~Application() = default;
};

#endif // APPLICATION_H
