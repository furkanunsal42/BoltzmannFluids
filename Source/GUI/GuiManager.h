#ifndef GUIMANAGER_H
#define GUIMANAGER_H

#include <QApplication>

#include "MainWindow.h"

class GuiManager
{
public:
    GuiManager(int& argc, char** argv);

    void run();

private:
    QApplication app;
    MainWindow mainWindow;
};

#endif // GUIMANAGER_H
