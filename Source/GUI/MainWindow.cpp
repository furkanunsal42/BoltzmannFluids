#include "MainWindow.h"

#include "RightSidePanel.h"

#include <QtWidgets/qmainwindow.h>
#include <QtWidgets/qgridlayout.h>
#include <QtGui/qpainter.h>

MainWindow::MainWindow(QWidget* parent)
	: QMainWindow(parent) 
{
	//setWindowTitle("BoltzmannFluids");
	//resize(1280, 720);


	auto centralLayout = new QHBoxLayout();
	//centralLayout->addWidget(renderingWidget);
	auto rightSidePanel = new RightSidePanel();
	centralLayout->addWidget(rightSidePanel);

	auto central = new QWidget();
	central->setLayout(centralLayout);
	setCentralWidget(central);

}