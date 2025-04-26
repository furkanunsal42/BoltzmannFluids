#include "GuiManager.h"

GuiManager::GuiManager(int& argc, char** argv)
    : app(argc, argv), mainWindow() {
}

void GuiManager::run() {
    mainWindow.show();
    app.exec();
}