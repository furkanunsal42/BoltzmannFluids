#include "InitialConditionsBox.h"
#include "SmartDoubleSpinBox.h"

#include <QLabel>
#include <QCheckBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QComboBox>

void enable_widget(QWidget* widget, QLabel* label);
void disable_widget(QWidget* widget, QLabel* label);

InitialConditionsBox::InitialConditionsBox(QWidget* parent)
    : QFrame(parent)
{
    auto main_layout = new QVBoxLayout(this);
    main_layout->setAlignment(Qt::AlignTop);
    main_layout->addWidget(createInitialConditionsGroup());
    main_layout->setContentsMargins(0,0,0,0);
    main_layout->setSpacing(0);
    setLayout(main_layout);

    setStyleSheet(
    "InitialConditionsBox {"
        "background-color: rgb(65, 66, 67);"
        "color: rgb(245, 246, 247);"
    "}"
    "InitialConditionsBox QDoubleSpinBox {"
        "color: rgb(225, 226, 227);"
        "border: 1px solid rgb(80, 81, 82);"
        "background-color: rgb(85, 86, 87);"
    "}"
    "InitialConditionsBox QDoubleSpinBox:hover {"
        "background-color: rgb(105, 106, 107);"
    "}"
    "InitialConditionsBox QLabel {"
        "font-weight: bold;"
        "background-color: rgb(65, 66, 67);"
        "color: rgb(225, 226, 227);"
    "}"
    "InitialConditionsBox QGroupBox {"
        "border: 0;"
        "background-color: rgb(65, 66, 67);"
    "}"
    "InitialConditionsBox QCheckBox {"
        "background-color: rgb(65, 66, 67);"
    "}"
    "InitialConditionsBox QComboBox {"
        "color: rgb(225, 226, 227);"
        "border: 1px solid rgb(80, 81, 82);"
        "background-color: rgb(85, 86, 87);"
    "}"
    "InitialConditionsBox QComboBox:hover {"
        "color: rgb(225, 226, 227);"
        "border: 1px solid rgb(80, 81, 82);"
        "background-color: rgb(105, 106, 107);"
    "}"
    //"InitialConditionsBox QAbstractItemView {"
    //    "background-color: rgb(85, 86, 87);"
    //"}"
        );

}

QGroupBox* InitialConditionsBox::createInitialConditionsGroup()
{
    auto group = new QGroupBox(this);

    auto layout = new QVBoxLayout(group);
    group->setLayout(layout);
    layout->setContentsMargins(0,0,0,0);
    layout->setSpacing(10);

    //// Box Label - Initial Conditions
    //auto box_label_initial_conditions = new QLabel("Initial Conditions");
    //layout->addWidget(box_label_initial_conditions);


    // Velocity Set
    auto velocity_set_vertical = new QHBoxLayout();
    layout->addLayout(velocity_set_vertical);
    velocity_set_vertical->setContentsMargins(0, 0, 0, 0);
    velocity_set_vertical->setSpacing(0);

    /// Label
    auto velocity_set_label = new QLabel("Velocity Set");
    velocity_set_vertical->addWidget(velocity_set_label);

    /// Value
    velocity_set_vertical->addSpacing(126);
    velocity_set = new QComboBox();
    velocity_set->addItems({"D2Q9", "D3Q15", "D3Q19", "D3Q27"});
    velocity_set_vertical->addWidget(velocity_set);
    velocity_set->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    //velocity_set_vertical->addStretch();
    velocity_set_vertical->addSpacing(0);


    // Floating Point accuracy
    auto floating_point_accuracy_vertical = new QHBoxLayout();
    layout->addLayout(floating_point_accuracy_vertical);
    floating_point_accuracy_vertical->setContentsMargins(0, 0, 0, 0);
    floating_point_accuracy_vertical->setSpacing(0);

    /// Label
    auto floating_point_accuracy_label = new QLabel("Floating Point Accuracy");
    floating_point_accuracy_vertical->addWidget(floating_point_accuracy_label);

    /// Value
    floating_point_accuracy_vertical->addSpacing(62);
    floating_point_accuracy = new QComboBox();
    floating_point_accuracy->addItems({"16-Bit", "32-Bit"});
    floating_point_accuracy_vertical->addWidget(floating_point_accuracy);
    floating_point_accuracy->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    //floating_point_accuracy_vertical->addStretch();
    floating_point_accuracy_vertical->addSpacing(0);


    //Resolution
    auto resolution = new QVBoxLayout();
    resolution->setContentsMargins(0, 0, 0, 0);
    resolution->setSpacing(0);
    layout->addLayout(resolution);

    /// Label
    auto resolution_label = new QLabel("Resolution");
    resolution->addWidget(resolution_label);

    auto resolution_horizontal = new QHBoxLayout();
    resolution->addLayout(resolution_horizontal);
    /// X
    resolution_horizontal->addSpacing(10);
    auto resolution_X_txt = new QLabel("X:");
    resolution_horizontal->addWidget(resolution_X_txt);
    resolution_X_value = new SmartDoubleSpinBox();
    resolution_horizontal->addWidget(resolution_X_value);

    /// Y
    resolution_horizontal->addSpacing(5);
    auto resolution_Y_txt = new QLabel("Y:");
    resolution_horizontal->addWidget(resolution_Y_txt);
    resolution_Y_value = new SmartDoubleSpinBox();
    resolution_horizontal->addWidget(resolution_Y_value);

    /// Z
    resolution_horizontal->addSpacing(5);
    auto resolution_Z_txt = new QLabel("Z:");
    resolution_horizontal->addWidget(resolution_Z_txt);
    resolution_Z_value = new SmartDoubleSpinBox();
    resolution_horizontal->addWidget(resolution_Z_value);


    // Gravity
    auto gravity_vertical = new QVBoxLayout();
    gravity_vertical->setContentsMargins(0, 0, 0, 0);
    gravity_vertical->setSpacing(0);
    layout->addLayout(gravity_vertical);

    /// Label
    auto gravity_label = new QLabel("Gravity");
    gravity_vertical->addWidget(gravity_label);

    auto gravity_horizontal = new QHBoxLayout();
    gravity_vertical->addLayout(gravity_horizontal);
    /// X
    gravity_horizontal->addSpacing(10);
    auto gravity_X_txt = new QLabel("X:");
    gravity_horizontal->addWidget(gravity_X_txt);
    gravity_X_value = new SmartDoubleSpinBox();
    gravity_horizontal->addWidget(gravity_X_value);

    /// Y
    gravity_horizontal->addSpacing(5);
    auto gravity_Y_txt = new QLabel("Y:");
    gravity_horizontal->addWidget(gravity_Y_txt);
    gravity_Y_value = new SmartDoubleSpinBox();
    gravity_horizontal->addWidget(gravity_Y_value);

    /// Z
    gravity_horizontal->addSpacing(5);
    auto gravity_Z_txt = new QLabel("Z:");
    gravity_horizontal->addWidget(gravity_Z_txt);
    gravity_Z_value = new SmartDoubleSpinBox();
    gravity_horizontal->addWidget(gravity_Z_value);



    // Initial Velocities
    auto initial_velocity_vertical = new QVBoxLayout();
    initial_velocity_vertical->setContentsMargins(0, 0, 0, 0);
    initial_velocity_vertical->setSpacing(0);
    layout->addLayout(initial_velocity_vertical);

    /// Label
    auto initial_velocity_label = new QLabel("Initial Velocity");
    initial_velocity_vertical->addWidget(initial_velocity_label);

    auto initial_velocity_horizontal = new QHBoxLayout();
    initial_velocity_vertical->addLayout(initial_velocity_horizontal);

    /// X
    initial_velocity_horizontal->addSpacing(10);
    auto initial_velocity_X_txt = new QLabel("X:");
    initial_velocity_horizontal->addWidget(initial_velocity_X_txt);
    initial_velocity_X_value = new SmartDoubleSpinBox();
    initial_velocity_horizontal->addWidget(initial_velocity_X_value);
    /// Y
    initial_velocity_horizontal->addSpacing(5);
    auto initial_velocity_Y_txt = new QLabel("Y:");
    initial_velocity_horizontal->addWidget(initial_velocity_Y_txt);
    initial_velocity_Y_value = new SmartDoubleSpinBox();
    initial_velocity_horizontal->addWidget(initial_velocity_Y_value);
    /// Z
    initial_velocity_horizontal->addSpacing(5);
    auto initial_velocity_Z_txt = new QLabel("Z:");
    initial_velocity_horizontal->addWidget(initial_velocity_Z_txt);
    initial_velocity_Z_value = new SmartDoubleSpinBox();
    initial_velocity_horizontal->addWidget(initial_velocity_Z_value);


    // Relaxation Time
    auto relaxation_time_vertical = new QHBoxLayout();
    layout->addLayout(relaxation_time_vertical);
    relaxation_time_vertical->setContentsMargins(0, 0, 0, 0);
    relaxation_time_vertical->setSpacing(0);

    /// Label
    auto relaxation_time_label = new QLabel("Relaxation Time");
    relaxation_time_vertical->addWidget(relaxation_time_label);

    /// Value
    relaxation_time_vertical->addSpacing(102);
    relaxation_time = new SmartDoubleSpinBox();
    relaxation_time->setValue(1.00);
    relaxation_time_vertical->addWidget(relaxation_time);
    //    relaxation_time_vertical->addStretch();
    relaxation_time_vertical->addSpacing(0);


    // Initial temperature
    auto initial_temprature_vertical = new QHBoxLayout();
    layout->addLayout(initial_temprature_vertical);
    initial_temprature_vertical->setContentsMargins(0, 0, 0, 0);
    initial_temprature_vertical->setSpacing(0);
    /// Label
    auto initial_temprature_label = new QLabel("Initial Temperature");
    initial_temprature_vertical->addWidget(initial_temprature_label);

    /// Value
    initial_temprature_vertical->addSpacing(85);
    initial_temprature_value = new SmartDoubleSpinBox();
    initial_temprature_value->setValue(1.00);
    initial_temprature_vertical->addWidget(initial_temprature_value);
    initial_temprature_vertical->addSpacing(0);
//    initial_temprature_vertical->addStretch();


    // Thermal Relaxation Time
    auto thermal_relaxation_time_vertical = new QHBoxLayout();
    layout->addLayout(thermal_relaxation_time_vertical);
    thermal_relaxation_time_vertical->setContentsMargins(0, 0, 0, 0);
    thermal_relaxation_time_vertical->setSpacing(0);

    /// Label
    auto thermal_relaxation_time_label = new QLabel("Thermal Relaxation Time");
    thermal_relaxation_time_vertical->addWidget(thermal_relaxation_time_label);

    /// Value
    thermal_relaxation_time_vertical->addSpacing(53);
    thermal_relaxation_time = new SmartDoubleSpinBox();
    thermal_relaxation_time_vertical->addWidget(thermal_relaxation_time);
    //thermal_relaxation_time_vertical->addStretch();
    thermal_relaxation_time_vertical->addSpacing(0);


    // Thermal Extension Coefficient
    auto thermal_extension_coefficient_vertical = new QHBoxLayout();
    layout->addLayout(thermal_extension_coefficient_vertical);
    thermal_extension_coefficient_vertical->setContentsMargins(0, 0, 0, 0);
    thermal_extension_coefficient_vertical->setSpacing(0);

    /// Label
    auto thermal_extension_coefficient_label = new QLabel("Thermal Relaxation Coefficient");
    thermal_extension_coefficient_vertical->addWidget(thermal_extension_coefficient_label);

    /// Value
    thermal_extension_coefficient_vertical->addSpacing(19);
    thermal_extension_coefficient = new SmartDoubleSpinBox();
    thermal_extension_coefficient_vertical->addWidget(thermal_extension_coefficient);
    //thermal_extension_coefficient_vertical->addStretch();
    thermal_extension_coefficient_vertical->addSpacing(0);


    // Phase Selector
    auto phase_selector_vertical = new QVBoxLayout();
    layout->addLayout(phase_selector_vertical);
    phase_selector_vertical->setContentsMargins(0, 0, 0, 0);
    phase_selector_vertical->setSpacing(0);

    /// Phase Selector Label
    auto phase_selsector_label = new QLabel("Phase Selector");
    phase_selector_vertical->addWidget(phase_selsector_label);

    /// Singlephase
    auto singlephase_horizontal = new QHBoxLayout();
    singlephase_horizontal->setContentsMargins(20, 0, 0, 0);
    singlephase_horizontal->setSpacing(0);
    phase_selector_vertical->addLayout(singlephase_horizontal);

    /// Singlephase Checkbox
    singlephase_checkbox = new QCheckBox();
    singlephase_checkbox->setChecked(true);
    singlephase_horizontal->addWidget(singlephase_checkbox);

    /// SinglephasePhase Label
    auto singlephase_label = new QLabel("Single Phase");
    singlephase_horizontal->addWidget(singlephase_label);
    singlephase_horizontal->addStretch();

    /// Multiphase
    auto multiphase_horizontal = new QHBoxLayout();
    multiphase_horizontal->setContentsMargins(20, 0, 0, 0);
    multiphase_horizontal->setSpacing(0);
    phase_selector_vertical->addLayout(multiphase_horizontal);

    /// Multiphase Checkbox
    multiphase_checkbox = new QCheckBox();
    multiphase_horizontal->addWidget(multiphase_checkbox);

    /// Multiphase Label
    auto multiphase_label = new QLabel("Multi Phase");
    multiphase_horizontal->addWidget(multiphase_label);
    multiphase_horizontal->addStretch();


    // Intermoleculer Interaction Strength
    auto intermoleculer_interaction_strength_vertical = new QHBoxLayout();
    layout->addLayout(intermoleculer_interaction_strength_vertical);
    intermoleculer_interaction_strength_vertical->setContentsMargins(0, 0, 0, 0);
    intermoleculer_interaction_strength_vertical->setSpacing(0);

    /// Label
    auto intermoleculer_interaction_strength_label = new QLabel("Intermoleculer Interaction Strength");
    intermoleculer_interaction_strength_vertical->addWidget(intermoleculer_interaction_strength_label);

    /// Value
    intermoleculer_interaction_strength_vertical->addSpacing(0);
    intermoleculer_interaction_strength = new SmartDoubleSpinBox();
    disable_widget(intermoleculer_interaction_strength, intermoleculer_interaction_strength_label);
    intermoleculer_interaction_strength_vertical->addWidget(intermoleculer_interaction_strength);
    //intermoleculer_interaction_strength_vertical->addStretch();
    intermoleculer_interaction_strength_vertical->setSpacing(0);

    /// Phase settings
    QObject::connect(singlephase_checkbox, &QCheckBox::toggled, this, [=](bool checked) {
        if (checked) multiphase_checkbox->setChecked(false);
        if (!checked && !multiphase_checkbox->isChecked()) singlephase_checkbox->setChecked(true);

        // Disable Intermoleculer Interaction Strength
        disable_widget(intermoleculer_interaction_strength, intermoleculer_interaction_strength_label);

    });
    QObject::connect(multiphase_checkbox, &QCheckBox::toggled, this, [=](bool checked) {
        if (checked) singlephase_checkbox->setChecked(false);
        if (!checked && !singlephase_checkbox->isChecked()) multiphase_checkbox->setChecked(true);

        // Enable Intermoleculer Interaction Strength
        enable_widget(intermoleculer_interaction_strength, intermoleculer_interaction_strength_label);

    });

    {
    // *** Boundary Conditions ***
    // Boundary X
    auto boundary_X_vertical = new QHBoxLayout();
    layout->addLayout(boundary_X_vertical);
    boundary_X_vertical->setContentsMargins(0, 0, 0, 0);
    boundary_X_vertical->setSpacing(0);

    /// Label
    auto boundary_X_label = new QLabel("Boundary X");
    boundary_X_vertical->addWidget(boundary_X_label);

    /// Value
    boundary_X_vertical->addSpacing(75);
    boundary_X = new QComboBox();
    boundary_X->addItems({"Periodic Wall", "Solid Wall", "Open Boundary"});
    boundary_X_vertical->addWidget(boundary_X);
    boundary_X->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    //floating_point_accuracy_vertical->addStretch();
    boundary_X_vertical->addSpacing(0);

    // Boundary Y
    auto boundary_Y_vertical = new QHBoxLayout();
    layout->addLayout(boundary_Y_vertical);
    boundary_Y_vertical->setContentsMargins(0, 0, 0, 0);
    boundary_Y_vertical->setSpacing(0);

    /// Label
    auto boundary_Y_label = new QLabel("Boundary Y");
    boundary_Y_vertical->addWidget(boundary_Y_label);

    /// Value
    boundary_Y_vertical->addSpacing(75);
    boundary_Y = new QComboBox();
    boundary_Y->addItems({"Periodic Wall", "Solid Wall", "Open Boundary"});
    boundary_Y_vertical->addWidget(boundary_Y);
    boundary_Y->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    //floating_point_accuracy_vertical->addStretch();
    boundary_Y_vertical->addSpacing(0);

    // Boundary Z
    auto boundary_Z_vertical = new QHBoxLayout();
    layout->addLayout(boundary_Z_vertical);
    boundary_Z_vertical->setContentsMargins(0, 0, 0, 0);
    boundary_Z_vertical->setSpacing(0);

    /// Label
    auto boundary_Z_label = new QLabel("Boundary Z");
    boundary_Z_vertical->addWidget(boundary_Z_label);

    /// Value
    boundary_Z_vertical->addSpacing(75);
    boundary_Z = new QComboBox();
    boundary_Z->addItems({"Periodic Wall", "Solid Wall", "Open Boundary"});
    boundary_Z_vertical->addWidget(boundary_Z);
    boundary_Z->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    //floating_point_accuracy_vertical->addStretch();
    boundary_Z_vertical->addSpacing(0);

    }


    return group;
}

void enable_widget(QWidget* widget, QLabel* label)
{
    widget->setEnabled(true);

    label->setStyleSheet(
        "font-weight: bold;"
        "background-color: rgb(65, 66, 67);"
        "color: rgb(225, 226, 227);"
        );

    QString style =
        "color: rgb(225, 226, 227);"
        "border: 1px solid rgb(80, 81, 82);"
        "background-color: rgb(85, 86, 87);";

    if (qobject_cast<QDoubleSpinBox*>(widget)) {
        style = "QDoubleSpinBox { " + style + " }"
                "QDoubleSpinBox:hover { background-color: rgb(105, 106, 107); }";
    }

    widget->setStyleSheet(style);
}


void disable_widget(QWidget* widget, QLabel* label)
{
    widget->setEnabled(false);
    label->setStyleSheet(
            "font-weight: bold;"
            "background-color: rgb(65, 66, 67);"
            "color: rgb(165, 166, 167);"
        );
    widget->setStyleSheet(
        "color: rgb(165, 166, 167);"
        "border: 1px solid rgb(60, 61, 62);"
        "background-color: rgb(65, 66, 67);"
        );
}
