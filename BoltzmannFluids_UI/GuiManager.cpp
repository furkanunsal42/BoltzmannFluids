#include "GuiManager.h"

GuiManager::GuiManager(int& argc, char** argv)
    : app(argc, argv)
{
}

void GuiManager::run()
{
    mainWindow.show();
    app.exec();
}
