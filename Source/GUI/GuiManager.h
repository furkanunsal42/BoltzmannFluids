#pragma once

#include <QtWidgets/qapplication>
#include <QtWidgets/qpushbutton.h>

#include <QtCore/qstringliteral.h>
#include <QtCore/qstring.h>


#include "MainWindow.h"

class GuiManager {
public:
	GuiManager(int& argc, char** argv);

	void run();

private:
	QApplication app;
	MainWindow mainWindow;
};


