#include "InitialConditionsBox.h"
#include "UI_Config.h"

#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>
#include <QGroupBox>
#include <QToolButton>
#include <QIcon>
#include <QStyle>
#include <QPixmap>
#include <QFrame>
#include <QVBoxLayout>
#include <QScrollBar>

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
    layout->setSpacing(9);

    //// Box Label - Initial Conditions
    //auto box_label_initial_conditions = new QLabel("Initial Conditions");
    //layout->addWidget(box_label_initial_conditions);

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
            gravity_horizontal->addStretch();
            /// X
            auto gravity_X_txt = new QLabel("X:");
            gravity_horizontal->addWidget(gravity_X_txt);
            auto gravity_X_box = new QDoubleSpinBox();
            gravity_X_box->setDecimals(DECIMAL_COUNT);
            gravity_X_box->setRange(GRAVITY_MIN, GRAVITY_MAX);
            gravity_X_box->setSingleStep(0.01);
            gravity_horizontal->addWidget(gravity_X_box);

            /// Y
            auto gravity_Y_txt = new QLabel("Y:");
            gravity_horizontal->addWidget(gravity_Y_txt);
            auto gravity_Y_box = new QDoubleSpinBox();
            gravity_Y_box->setDecimals(DECIMAL_COUNT);
            gravity_Y_box->setRange(GRAVITY_MIN, GRAVITY_MAX);
            gravity_Y_box->setSingleStep(0.01);
            gravity_horizontal->addWidget(gravity_Y_box);

            /// Z
            auto gravity_Z_txt = new QLabel("Z:");
            gravity_horizontal->addWidget(gravity_Z_txt);
            auto gravity_Z_box = new QDoubleSpinBox();
            gravity_Z_box->setDecimals(DECIMAL_COUNT);
            gravity_Z_box->setRange(GRAVITY_MIN, GRAVITY_MAX);
            gravity_Z_box->setSingleStep(0.01);
            gravity_horizontal->addWidget(gravity_Z_box);

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
            initial_velocity_horizontal->addStretch();

            /// X
            auto initial_velocity_X_txt = new QLabel("x:");
            initial_velocity_horizontal->addWidget(initial_velocity_X_txt);
            auto initial_velocity_X_box = new QDoubleSpinBox();
            initial_velocity_X_box->setDecimals(DECIMAL_COUNT);
            initial_velocity_X_box->setRange(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX);
            initial_velocity_X_box->setSingleStep(0.01);
            initial_velocity_horizontal->addWidget(initial_velocity_X_box);
            /// Y
            auto initial_velocity_Y_txt = new QLabel("y:");
            initial_velocity_horizontal->addWidget(initial_velocity_Y_txt);
            auto initial_velocity_Y_box = new QDoubleSpinBox();
            initial_velocity_Y_box->setDecimals(DECIMAL_COUNT);
            initial_velocity_Y_box->setRange(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX);
            initial_velocity_Y_box->setSingleStep(0.01);
            initial_velocity_horizontal->addWidget(initial_velocity_Y_box);
            /// Z
            auto initial_velocity_Z_txt = new QLabel("z:");
            initial_velocity_horizontal->addWidget(initial_velocity_Z_txt);
            auto initial_velocity_Z_box = new QDoubleSpinBox();
            initial_velocity_Z_box->setDecimals(DECIMAL_COUNT);
            initial_velocity_Z_box->setRange(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX);
            initial_velocity_Z_box->setSingleStep(0.01);
            initial_velocity_horizontal->addWidget(initial_velocity_Z_box);

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
        auto singlephase_checkbox = new QCheckBox();
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
        auto multiphase_checkbox = new QCheckBox();
        multiphase_horizontal->addWidget(multiphase_checkbox);

        /// Multiphase Label
        auto multiphase_label = new QLabel("Multi Phase");
        multiphase_horizontal->addWidget(multiphase_label);
        multiphase_horizontal->addStretch();

        // Mutual Exclusiveness
        QObject::connect(singlephase_checkbox, &QCheckBox::toggled, [=](bool checked) {
            if (checked) multiphase_checkbox->setChecked(false);
            if (!checked && !multiphase_checkbox->isChecked()) singlephase_checkbox->setChecked(true);
        });
        QObject::connect(multiphase_checkbox, &QCheckBox::toggled, [=](bool checked) {
            if (checked) singlephase_checkbox->setChecked(false);
            if (!checked && !singlephase_checkbox->isChecked()) multiphase_checkbox->setChecked(true);
        });

        layout->addLayout(phase_selector_vertical);
    }

    // Initial temperature
    {
        auto temprature_vertical = new QVBoxLayout();
        temprature_vertical->setContentsMargins(0, 0, 0, 0);
        temprature_vertical->setSpacing(0);

        // Initial Temprature Label
        auto initial_temprature_label = new QLabel("Initial Temperature");
        temprature_vertical->addWidget(initial_temprature_label);

        auto initial_temprature_layout = new QHBoxLayout();
        temprature_vertical->addLayout(initial_temprature_layout);

        // Initial Temprature Value
        initial_temprature_layout->addStretch();
        auto initial_temprature_value = new QDoubleSpinBox();
        initial_temprature_value->setDecimals(DECIMAL_COUNT);
        initial_temprature_value->setRange(INITIAL_TEMPRATURE_MIN, INITIAL_TEMPRATURE_MAX); // Example range for temperature
        initial_temprature_value->setSingleStep(0.1);
        initial_temprature_layout->addWidget(initial_temprature_value);

        layout->addLayout(temprature_vertical);
    }

    group->setLayout(layout);
    return group;
}
