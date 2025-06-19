#ifndef R_GHTSIDEPANEL_H
#define R_GHTSIDEPANEL_H

#include <QFrame>

class QGroupBox;
class SmartDoubleSpinBox;
class QCheckBox;
class QComboBox;

class InitialConditionsBox : public QFrame
{
    Q_OBJECT
public:

    explicit InitialConditionsBox(QWidget* parent = nullptr);

private:

    QGroupBox* createInitialConditionsGroup();

    QComboBox* velocity_set;
    QComboBox* floating_point_accuracy;

    SmartDoubleSpinBox* resolution_X_value;
    SmartDoubleSpinBox* resolution_Y_value;
    SmartDoubleSpinBox* resolution_Z_value;

    SmartDoubleSpinBox* relaxation_time;

    SmartDoubleSpinBox* gravity_X_value;
    SmartDoubleSpinBox* gravity_Y_value;
    SmartDoubleSpinBox* gravity_Z_value;

    SmartDoubleSpinBox* initial_velocity_X_value;
    SmartDoubleSpinBox* initial_velocity_Y_value;
    SmartDoubleSpinBox* initial_velocity_Z_value;

    SmartDoubleSpinBox* initial_temprature_value;
    SmartDoubleSpinBox* thermal_relaxation_time;
    SmartDoubleSpinBox* thermal_extension_coefficient;

    QCheckBox* singlephase_checkbox;
    QCheckBox* multiphase_checkbox;

    SmartDoubleSpinBox* intermoleculer_interaction_strength;


    QComboBox* boundary_X;
    QComboBox* boundary_Y;
    QComboBox* boundary_Z;

    SmartDoubleSpinBox* boundary_X_temprature;
    SmartDoubleSpinBox* boundary_X_effective_density;
    SmartDoubleSpinBox* boundary_X_velocity_translation_X_box;
    SmartDoubleSpinBox* boundary_X_velocity_translation_Y_box;
    SmartDoubleSpinBox* boundary_X_velocity_translation_Z_box;

    SmartDoubleSpinBox* boundary_Y_temprature;
    SmartDoubleSpinBox* boundary_Y_effective_density;
    SmartDoubleSpinBox* boundary_Y_velocity_translation_X_box;
    SmartDoubleSpinBox* boundary_Y_velocity_translation_Y_box;
    SmartDoubleSpinBox* boundary_Y_velocity_translation_Z_box;

    SmartDoubleSpinBox* boundary_Z_temprature;
    SmartDoubleSpinBox* boundary_Z_effective_density;
    SmartDoubleSpinBox* boundary_Z_velocity_translation_X_box;
    SmartDoubleSpinBox* boundary_Z_velocity_translation_Y_box;
    SmartDoubleSpinBox* boundary_Z_velocity_translation_Z_box;

};

#endif // R_GHTSIDEPANEL_H
