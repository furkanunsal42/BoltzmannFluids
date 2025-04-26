#pragma once

#include <QtWidgets/qapplication.h>
#include "MainWindow.h"

class GuiManager {
private:
	QApplication app;
	MainWindow mainWindow;

public:
	GuiManager(int& argc, char** argv);

	void run();
};