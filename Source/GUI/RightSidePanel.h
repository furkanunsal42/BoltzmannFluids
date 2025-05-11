#pragma once
#include <QtWidgets/qboxlayout.h>
#include <QtWidgets/qgroupbox.h>

class RightSidePanel : public QWidget{
	Q_OBJECT

public:
	explicit RightSidePanel(QWidget* parent = nullptr);

private:
	QVBoxLayout* mainLayout;

	QGroupBox* createInitialConditionsGroup();
};