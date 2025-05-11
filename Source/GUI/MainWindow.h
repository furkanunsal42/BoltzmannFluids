#pragma once

#include <QtWidgets/qmainwindow.h>

class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(QWidget* parent = nullptr);
};