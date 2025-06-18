#ifndef R_GHTSIDEPANEL_H
#define R_GHTSIDEPANEL_H

#include <QFrame>

class QGroupBox;
class SmartDoubleSpinBox;
class QCheckBox;

class InitialConditionsBox : public QFrame
{
    Q_OBJECT
public:

    explicit InitialConditionsBox(QWidget* parent = nullptr);

private:

    QGroupBox* createInitialConditionsGroup();

    SmartDoubleSpinBox* resolution_X_value;
    SmartDoubleSpinBox* resolution_Y_value;
    SmartDoubleSpinBox* resolution_Z_value;

    SmartDoubleSpinBox* gravity_X_value;
    SmartDoubleSpinBox* gravity_Y_value;
    SmartDoubleSpinBox* gravity_Z_value;

    SmartDoubleSpinBox* initial_velocity_X_value;
    SmartDoubleSpinBox* initial_velocity_Y_value;
    SmartDoubleSpinBox* initial_velocity_Z_value;

    QCheckBox* singlephase_checkbox;
    QCheckBox* multiphase_checkbox;

    SmartDoubleSpinBox* initial_temprature_value;
};

#endif // R_GHTSIDEPANEL_H
