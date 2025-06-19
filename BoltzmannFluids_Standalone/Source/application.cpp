#include "application.h"

Application& Application::get()
{
    static Application app;
    return app;
}
