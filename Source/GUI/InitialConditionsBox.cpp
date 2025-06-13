#include "InitialConditionsBox.h"
#include "UI_Config.h"

#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>

#include <QToolButton>
#include <QIcon>
#include <QStyle>
#include <QPixmap>
#include <QFrame>

#include <QScrollBar>

InitialConditionsBox::InitialConditionsBox(QWidget* parent)
    : QWidget(parent)
{
    main_layout = new QVBoxLayout(this);
    main_layout->setAlignment(Qt::AlignTop);
    main_layout->addWidget(createInitialConditionsGroup());

    setStyleSheet(
    "InitialConditionsBox {"
        "background-color: rgb(60, 61, 62);"
        "color: rgb(245, 246, 247);"
    "}"
    "InitialConditionsBox QDoubleSpinBox {"
        "color: rgb(225, 226, 227);"
        "border: 1px solid rgb(80, 81, 82);"
        "background-color: rgb(85, 86, 87);"
    "}"
        "InitialConditionsBox QLabel {"
        "font-weight: bold;"
        "background-color: rgb(60, 61, 62);"
        "color: rgb(225, 226, 227);"
    "}"
    "InitialConditionsBox QGroupBox {"
        "border: 2px solid rgb(75, 76, 77);"
        "background-color: rgb(60, 61, 62);"
    "}"
    "InitialConditionsBox QCheckBox {"
        "background-color: rgb(60, 61, 62);"
    "}"
        );

}

QGroupBox* InitialConditionsBox::createInitialConditionsGroup()
{
    auto group = new QGroupBox();

    auto layout = new QVBoxLayout();
    layout->setSpacing(10);


    // Box Label - Initial Conditions
    auto box_label_initial_conditions = new QLabel("Initial Conditions");
    layout->addWidget(box_label_initial_conditions);

    // Gravity
    {
        // Label
        auto gravity_label = new QLabel("Gravity");
        layout->addWidget(gravity_label);

        auto gravity_layout = new QHBoxLayout();

        // X
        auto gravity_X_txt = new QLabel("x:");
        gravity_layout->addWidget(gravity_X_txt);
        auto gravity_X_box = new QDoubleSpinBox();
        gravity_X_box->setRange(GRAVITY_MIN, GRAVITY_MAX);
        gravity_X_box->setSingleStep(0.01);
        gravity_layout->addWidget(gravity_X_box);

        // Y
        auto gravity_Y_txt = new QLabel("y:");
        gravity_layout->addWidget(gravity_Y_txt);
        auto gravity_Y_box = new QDoubleSpinBox();
        gravity_Y_box->setRange(GRAVITY_MIN, GRAVITY_MAX);
        gravity_Y_box->setSingleStep(0.01);
        gravity_layout->addWidget(gravity_Y_box);

        // Z
        auto gravity_Z_txt = new QLabel("z:");
        gravity_layout->addWidget(gravity_Z_txt);
        auto gravity_Z_box = new QDoubleSpinBox();
        gravity_Z_box->setRange(GRAVITY_MIN, GRAVITY_MAX);
        gravity_Z_box->setSingleStep(0.01);
        gravity_layout->addWidget(gravity_Z_box);

        gravity_layout->addStretch();
        layout->addLayout(gravity_layout);
    }


    // Initial Velocities
    {
        // Label
        auto initial_velocity_label = new QLabel("Initial Velocity");
        layout->addWidget(initial_velocity_label);

        auto initial_velocity_layout = new QHBoxLayout();
        // X
        auto initial_velocity_X_txt = new QLabel("x:");
        initial_velocity_layout->addWidget(initial_velocity_X_txt);
        auto initial_velocity_X_box = new QDoubleSpinBox();
        initial_velocity_X_box->setRange(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX);
        initial_velocity_X_box->setSingleStep(0.01);
        initial_velocity_layout->addWidget(initial_velocity_X_box);
        // Y
        auto initial_velocity_Y_txt = new QLabel("y:");
        initial_velocity_layout->addWidget(initial_velocity_Y_txt);
        auto initial_velocity_Y_box = new QDoubleSpinBox();
        initial_velocity_Y_box->setRange(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX);
        initial_velocity_Y_box->setSingleStep(0.01);
        initial_velocity_layout->addWidget(initial_velocity_Y_box);
        // Z
        auto initial_velocity_Z_txt = new QLabel("z:");
        initial_velocity_layout->addWidget(initial_velocity_Z_txt);
        auto initial_velocity_Z_box = new QDoubleSpinBox();
        initial_velocity_Z_box->setRange(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX);
        initial_velocity_Z_box->setSingleStep(0.01);
        initial_velocity_layout->addWidget(initial_velocity_Z_box);

        initial_velocity_layout->addStretch();
        layout->addLayout(initial_velocity_layout);
    }

    // Densities
    {
        auto density_layout = new QHBoxLayout();

        // Label - Single Phase
        auto density_label_single = new QLabel("Single Phase");
        density_layout->addWidget(density_label_single);

        // Single Phase Checkbox
        auto density_single_phase_checkbox = new QCheckBox();
        density_single_phase_checkbox->setChecked(true);
        density_layout->addWidget(density_single_phase_checkbox);

        // Label - Multi Phase
        auto density_label_multi = new QLabel("Multi Phase");
        density_layout->addWidget(density_label_multi);

        // Multi Phase Checkbox
        auto density_multi_phase_checkbox = new QCheckBox();
        density_layout->addWidget(density_multi_phase_checkbox);

        // Mutual Exclusiveness
        QObject::connect(density_single_phase_checkbox, &QCheckBox::toggled, [=](bool checked) {
            if (checked) density_multi_phase_checkbox->setChecked(false);
            if (!checked && !density_multi_phase_checkbox->isChecked()) density_single_phase_checkbox->setChecked(true);
        });
        QObject::connect(density_multi_phase_checkbox, &QCheckBox::toggled, [=](bool checked) {
            if (checked) density_single_phase_checkbox->setChecked(false);
            if (!checked && !density_single_phase_checkbox->isChecked()) density_multi_phase_checkbox->setChecked(true);
        });

        density_layout->addStretch();
        layout->addLayout(density_layout);
    }

    // Initial temperature
    {
        auto initial_temprature_layout = new QHBoxLayout();

        // Initial Temprature Label
        auto initial_temprature_label = new QLabel("Initial Temperature");
        initial_temprature_layout->addWidget(initial_temprature_label);

        // Initial Temprature Value
        auto initial_temprature_value = new QDoubleSpinBox();
        initial_temprature_value->setRange(INITIAL_TEMPRATURE_MIN, INITIAL_TEMPRATURE_MAX); // Example range for temperature
        initial_temprature_value->setSingleStep(0.1);
        initial_temprature_layout->addWidget(initial_temprature_value);

        initial_temprature_layout->addStretch();
        layout->addLayout(initial_temprature_layout);
    }


    group->setLayout(layout);
    return group;
}
