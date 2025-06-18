#include "InitialConditionsBox.h"
#include "SmartDoubleSpinBox.h"

#include <QLabel>
#include <QCheckBox>
#include <QGroupBox>
#include <QVBoxLayout>

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
        "background-color: rgb(110, 111, 112);"
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
        );

}

QGroupBox* InitialConditionsBox::createInitialConditionsGroup()
{
    auto group = new QGroupBox(this);

    auto layout = new QVBoxLayout(group);
    layout->setContentsMargins(0,0,0,0);
    layout->setSpacing(10);

    //// Box Label - Initial Conditions
    //auto box_label_initial_conditions = new QLabel("Initial Conditions");
    //layout->addWidget(box_label_initial_conditions);

    //Resolution

    // Gravity
    {
        auto gravity_vertical = new QVBoxLayout();
        gravity_vertical->setContentsMargins(0, 0, 0, 0);
        gravity_vertical->setSpacing(0);
        layout->addLayout(gravity_vertical);

        // Label
        auto gravity_label = new QLabel("Gravity");
        gravity_vertical->addWidget(gravity_label);

        {   //X-Y-Z
            auto gravity_horizontal = new QHBoxLayout();
            /// X
            gravity_horizontal->addSpacing(5);
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

            gravity_vertical->addLayout(gravity_horizontal);
        }
    }


    // Initial Velocities
    {
        auto initial_velocity_vertical = new QVBoxLayout();
        initial_velocity_vertical->setContentsMargins(0, 0, 0, 0);
        initial_velocity_vertical->setSpacing(0);
        layout->addLayout(initial_velocity_vertical);

        // Label
        auto initial_velocity_label = new QLabel("Initial Velocity");
        initial_velocity_vertical->addWidget(initial_velocity_label);

        {   // X-Y-Z
            auto initial_velocity_horizontal = new QHBoxLayout();
            //initial_velocity_horizontal->addStretch();

            /// X
            initial_velocity_horizontal->addSpacing(5);
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

            initial_velocity_vertical->addLayout(initial_velocity_horizontal);
        }
    }

    // Phase Selector
    {
        auto phase_selector_vertical = new QVBoxLayout();
        phase_selector_vertical->setContentsMargins(0, 0, 0, 0);
        phase_selector_vertical->setSpacing(0);

        // Phase Selector Label
        auto phase_selsector_label = new QLabel("Phase Selector");
        phase_selector_vertical->addWidget(phase_selsector_label);

        // Singlephase
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

        // Multiphase
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

        // Mutual Exclusiveness
        QObject::connect(singlephase_checkbox, &QCheckBox::toggled, this, [=](bool checked) {
            if (checked) multiphase_checkbox->setChecked(false);
            if (!checked && !multiphase_checkbox->isChecked()) singlephase_checkbox->setChecked(true);
        });
        QObject::connect(multiphase_checkbox, &QCheckBox::toggled, this, [=](bool checked) {
            if (checked) singlephase_checkbox->setChecked(false);
            if (!checked && !singlephase_checkbox->isChecked()) multiphase_checkbox->setChecked(true);
        });

        layout->addLayout(phase_selector_vertical);
    }

    // Initial temperature
    {
        auto temprature_vertical = new QHBoxLayout();
        temprature_vertical->setContentsMargins(0, 0, 0, 0);
        temprature_vertical->setSpacing(0);

        // Initial Temprature Label
        auto initial_temprature_label = new QLabel("Initial Temperature");
        temprature_vertical->addWidget(initial_temprature_label);

        //auto initial_temprature_layout = new QHBoxLayout();
        //temprature_vertical->addLayout(initial_temprature_layout);

        // Initial Temprature Value
        temprature_vertical->addSpacing(10);
        initial_temprature_value = new SmartDoubleSpinBox();
        initial_temprature_value->setValue(1.00);
        temprature_vertical->addWidget(initial_temprature_value);
        temprature_vertical->addStretch();

        layout->addLayout(temprature_vertical);
    }

    group->setLayout(layout);
    return group;
}
