#include "RightSidePanel.h"
#include "UI_Config.h"

#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>

RightSidePanel::RightSidePanel(QWidget* parent)
    : QWidget(parent)
{
    setStyleSheet("background-color: rgb(45, 45, 45); color: white;");

    mainLayout = new QVBoxLayout(this);
    mainLayout->setAlignment(Qt::AlignTop);
    mainLayout->addWidget(createInitialConditionsGroup());

}

QGroupBox* RightSidePanel::createInitialConditionsGroup()
{
    auto group = new QGroupBox("Initial Conditions");
    group->setStyleSheet("QGroupBox { font-weight: bold; color: rgb(225, 225, 225); border: 1px solid gray; margin-top: 10px; padding-top: 20px; }");
    group->setMinimumWidth(280);
    auto layout = new QVBoxLayout();


    // Gravity
    {
        // Label
        auto gravity_label = new QLabel("Gravity");
        gravity_label->setStyleSheet("QLabel { font-weight: bold; color: rgb(208, 208, 208);}");
        layout->addWidget(gravity_label); // Spanning the label across 6 columns

        auto gravity_layout = new QHBoxLayout();

        // X
        auto gravity_X_txt = new QLabel("x:");
        gravity_layout->addWidget(gravity_X_txt);
        auto gravity_X_box = new QDoubleSpinBox();
        gravity_X_box->setRange(GRAVITY_MIN, GRAVITY_MAX);
        gravity_X_box->setSingleStep(0.01);
        gravity_X_box->setFixedSize(59,22);

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
        initial_velocity_label->setStyleSheet("QLabel { font-weight: bold; color: rgb(208, 208, 208);}");
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
        density_label_single->setStyleSheet("QLabel { font-weight: bold; color: rgb(208, 208, 208);}");
        density_layout->addWidget(density_label_single);

        // Single Phase Checkbox
        auto density_single_phase_checkbox = new QCheckBox();
        density_single_phase_checkbox->setChecked(true);
        density_layout->addWidget(density_single_phase_checkbox);

        // Label - Multi Phase
        auto density_label_multi = new QLabel("Multi Phase");
        density_label_multi->setStyleSheet("QLabel { font-weight: bold; color: rgb(208, 208, 208);}");
        density_layout->addWidget(density_label_multi);

        // Multi Phase Checkbox
        auto density_multi_phase_checkbox = new QCheckBox();
        density_layout->addWidget(density_multi_phase_checkbox);

        // Mutual Exclusiveness
        QObject::connect(density_single_phase_checkbox, &QCheckBox::toggled, [=](bool checked) {
            if (checked) density_multi_phase_checkbox->setChecked(false);
        });
        QObject::connect(density_multi_phase_checkbox, &QCheckBox::toggled, [=](bool checked) {
            if (checked) density_single_phase_checkbox->setChecked(false);
        });

        density_layout->addStretch();
        layout->addLayout(density_layout);
    }

    // Initial temperature
    {
        auto initial_temprature_layout = new QHBoxLayout();

        // Initial Temprature Label
        auto initial_temprature_label = new QLabel("Initial Temperature");
        initial_temprature_label->setStyleSheet("QLabel { font-weight: bold; color: rgb(208, 208, 208);}");
        initial_temprature_layout->addWidget(initial_temprature_label);

        // Initial Temprature Value
        auto initial_temprature_value = new QDoubleSpinBox();
        initial_temprature_value->setRange(INITIAL_TEMPRATURE_MIN, INITIAL_TEMPRATURE_MAX); // Example range for temperature
        initial_temprature_value->setSingleStep(0.1);
        initial_temprature_layout->addWidget(initial_temprature_value);

        // "°C"
        auto degree_label = new QLabel("Celsius Degree");
        degree_label->setStyleSheet("QLabel { color: rgb(200, 200, 200); }");
        initial_temprature_layout->addWidget(degree_label);

        initial_temprature_layout->addStretch();
        layout->addLayout(initial_temprature_layout);
    }


    group->setLayout(layout);
    return group;
}
