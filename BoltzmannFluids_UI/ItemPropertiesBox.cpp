#include "ItemPropertiesBox.h"
#include <glm.hpp>
#include <gtc/quaternion.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtx/transform.hpp>

#include <gtx/matrix_decompose.hpp>


#include "SmartDoubleSpinBox.h"
#include "AddItemsBox.h"
#include "application.h"

#include <QGroupBox>
#include <QVBoxLayout>
#include <QLabel>

void enum_to_qstring(Type& type, QLabel* _string);

ItemPropertiesBox::ItemPropertiesBox(QWidget *parent)
    :QWidget(parent)
{
    auto main_layout = new QVBoxLayout(this);
    main_layout->setAlignment(Qt::AlignTop);
    main_layout->setContentsMargins(0,0,0,0);
    main_layout->setSpacing(10);
    setLayout(main_layout);


    {   /// Name & Type
        auto name_vertical = new QVBoxLayout();
        name_vertical->setContentsMargins(0, 0, 0, 0);
        name_vertical->setSpacing(0);
        main_layout->addLayout(name_vertical);

        name_label = new QLabel("BoltzmannFluids");
        name_vertical->addWidget(name_label);         // TODO: use when add scene collection

        //  type_label = new QLabel("Object Type");
        //  name_vertical->addWidget(type_label);
    }

    {   /// Veocity Translation
        auto velocity_translation_vertical = new QVBoxLayout();
        velocity_translation_vertical->setContentsMargins(0, 0, 0, 0);
        velocity_translation_vertical->setSpacing(0);
        main_layout->addLayout(velocity_translation_vertical);

        // Label
        auto velocity_translation_label = new QLabel("Velocity Translation");
        velocity_translation_vertical->addWidget(velocity_translation_label);

        auto velocity_translation_horizontal = new QHBoxLayout();
        //        velocity_translation_horizontal->addStretch();

        /// X
        velocity_translation_horizontal->addSpacing(5);
        auto velocity_translation_X_txt = new QLabel("X:");
        velocity_translation_horizontal->addWidget(velocity_translation_X_txt);
        velocity_translation_X_box = new SmartDoubleSpinBox();
        velocity_translation_horizontal->addWidget(velocity_translation_X_box);
        /// Y
        velocity_translation_horizontal->addSpacing(5);
        auto velocity_translation_Y_txt = new QLabel("Y:");
        velocity_translation_horizontal->addWidget(velocity_translation_Y_txt);
        velocity_translation_Y_box = new SmartDoubleSpinBox();
        velocity_translation_horizontal->addWidget(velocity_translation_Y_box);
        /// Z
        velocity_translation_horizontal->addSpacing(5);
        auto velocity_translation_Z_txt = new QLabel("Z:");
        velocity_translation_horizontal->addWidget(velocity_translation_Z_txt);
        velocity_translation_Z_box = new SmartDoubleSpinBox();
        velocity_translation_horizontal->addWidget(velocity_translation_Z_box);

        velocity_translation_vertical->addLayout(velocity_translation_horizontal);
    }

    { /// Velocity Angular
        auto velocity_angular_vertical = new QVBoxLayout();
        velocity_angular_vertical->setContentsMargins(0, 0, 0, 0);
        velocity_angular_vertical->setSpacing(0);
        main_layout->addLayout(velocity_angular_vertical);

        // Label
        auto velocity_angular_label = new QLabel("Velocity Angular");
        velocity_angular_vertical->addWidget(velocity_angular_label);

        auto velocity_angular_horizontal = new QHBoxLayout();

        /// X
        velocity_angular_horizontal->addSpacing(5);
        auto velocity_angular_X_txt = new QLabel("X:");
        velocity_angular_horizontal->addWidget(velocity_angular_X_txt);
        velocity_angular_X_box = new SmartDoubleSpinBox();
        velocity_angular_X_box->setSingleStep(0.01);
        velocity_angular_horizontal->addWidget(velocity_angular_X_box);
        /// Y
        velocity_angular_horizontal->addSpacing(5);
        auto velocity_angular_Y_txt = new QLabel("Y:");
        velocity_angular_horizontal->addWidget(velocity_angular_Y_txt);
        velocity_angular_Y_box = new SmartDoubleSpinBox();
        velocity_angular_horizontal->addWidget(velocity_angular_Y_box);
        /// Z
        velocity_angular_horizontal->addSpacing(5);
        auto velocity_angular_Z_txt = new QLabel("Z:");
        velocity_angular_horizontal->addWidget(velocity_angular_Z_txt);
        velocity_angular_Z_box = new SmartDoubleSpinBox();
        velocity_angular_horizontal->addWidget(velocity_angular_Z_box);

        velocity_angular_vertical->addLayout(velocity_angular_horizontal);
    }

    /// Center of Mass
    {
        auto center_of_mass__vertical = new QVBoxLayout();
        center_of_mass__vertical->setContentsMargins(0, 0, 0, 0);
        center_of_mass__vertical->setSpacing(0);
        main_layout->addLayout(center_of_mass__vertical);

        // Label
        auto center_of_mass_label = new QLabel("Center of Mass");
        center_of_mass__vertical->addWidget(center_of_mass_label);

        auto center_of_mass_horizontal = new QHBoxLayout();

        /// X
        center_of_mass_horizontal->addSpacing(5);
        auto center_of_mass_X_txt = new QLabel("X:");
        center_of_mass_horizontal->addWidget(center_of_mass_X_txt);
        center_of_mass_X_box = new SmartDoubleSpinBox();
        center_of_mass_horizontal->addWidget(center_of_mass_X_box);
        /// Y
        center_of_mass_horizontal->addSpacing(5);
        auto center_of_mass_Y_txt = new QLabel("Y:");
        center_of_mass_horizontal->addWidget(center_of_mass_Y_txt);
        center_of_mass_Y_box = new SmartDoubleSpinBox();
        center_of_mass_horizontal->addWidget(center_of_mass_Y_box);
        /// Z
        center_of_mass_horizontal->addSpacing(5);
        auto center_of_mass_Z_txt = new QLabel("Z:");
        center_of_mass_horizontal->addWidget(center_of_mass_Z_txt);
        center_of_mass_Z_box = new SmartDoubleSpinBox();
        center_of_mass_horizontal->addWidget(center_of_mass_Z_box);

        center_of_mass__vertical->addLayout(center_of_mass_horizontal);
    }

    /// Position
    {
        auto position_vertical = new QVBoxLayout();
        position_vertical->setContentsMargins(0, 0, 0, 0);
        position_vertical->setSpacing(0);
        main_layout->addLayout(position_vertical);

        // Label
        auto position_label = new QLabel("Position");
        position_vertical->addWidget(position_label);

        auto position_horizontal = new QHBoxLayout();

        /// X
        position_horizontal->addSpacing(5);
        auto position_X_txt = new QLabel("X:");
        position_horizontal->addWidget(position_X_txt);
        position_X_box = new SmartDoubleSpinBox();
        position_horizontal->addWidget(position_X_box);
        /// Y
        position_horizontal->addSpacing(5);
        auto position_Y_txt = new QLabel("Y:");
        position_horizontal->addWidget(position_Y_txt);
        position_Y_box = new SmartDoubleSpinBox();
        position_horizontal->addWidget(position_Y_box);
        /// Z
        position_horizontal->addSpacing(5);
        auto position_Z_txt = new QLabel("Z:");
        position_horizontal->addWidget(position_Z_txt);
        position_Z_box = new SmartDoubleSpinBox();
        position_horizontal->addWidget(position_Z_box);

        position_vertical->addLayout(position_horizontal);
    }

    { /// Rotation
        auto rotation_vertical = new QVBoxLayout();
        rotation_vertical->setContentsMargins(0, 0, 0, 0);
        rotation_vertical->setSpacing(0);
        main_layout->addLayout(rotation_vertical);

        // Label
        auto rotation_label = new QLabel("Rotation");
        rotation_vertical->addWidget(rotation_label);

        auto rotation_horizontal = new QHBoxLayout();

        /// X
        rotation_horizontal->addSpacing(5);
        auto rotation_X_txt = new QLabel("X:");
        rotation_horizontal->addWidget(rotation_X_txt);
        rotation_X_box = new SmartDoubleSpinBox();
        rotation_horizontal->addWidget(rotation_X_box);
        /// Y
        rotation_horizontal->addSpacing(5);
        auto rotation_Y_txt = new QLabel("Y:");
        rotation_horizontal->addWidget(rotation_Y_txt);
        rotation_Y_box = new SmartDoubleSpinBox();
        rotation_horizontal->addWidget(rotation_Y_box);
        /// Z
        rotation_horizontal->addSpacing(5);
        auto rotation_Z_txt = new QLabel("Z:");
        rotation_horizontal->addWidget(rotation_Z_txt);
        rotation_Z_box = new SmartDoubleSpinBox();
        rotation_horizontal->addWidget(rotation_Z_box);

        rotation_vertical->addLayout(rotation_horizontal);
    }

    /// Size
    {
        auto size_vertical = new QVBoxLayout();
        size_vertical->setContentsMargins(0, 0, 0, 0);
        size_vertical->setSpacing(0);
        main_layout->addLayout(size_vertical);

        // Label
        auto size_label = new QLabel("Size");
        size_vertical->addWidget(size_label);

        auto size_horizontal = new QHBoxLayout();

        /// X
        size_horizontal->addSpacing(5);
        auto size_X_txt = new QLabel("X:");
        size_horizontal->addWidget(size_X_txt);
        size_X_box = new SmartDoubleSpinBox();
        size_horizontal->addWidget(size_X_box);
        /// Y
        size_horizontal->addSpacing(5);
        auto size_Y_txt = new QLabel("Y:");
        size_horizontal->addWidget(size_Y_txt);
        size_Y_box = new SmartDoubleSpinBox();
        size_horizontal->addWidget(size_Y_box);
        /// Z
        size_horizontal->addSpacing(5);
        auto size_Z_txt = new QLabel("Z:");
        size_horizontal->addWidget(size_Z_txt);
        size_Z_box = new SmartDoubleSpinBox();
        size_horizontal->addWidget(size_Z_box);

        size_vertical->addLayout(size_horizontal);
    }
    {
        // Item temperature
        auto item_temprature_vertical = new QHBoxLayout();
        main_layout->addLayout(item_temprature_vertical);
        item_temprature_vertical->setContentsMargins(0, 0, 0, 0);
        item_temprature_vertical->setSpacing(0);

        // Label
        auto item_temprature_label = new QLabel("Item Temperature");
        item_temprature_vertical->addWidget(item_temprature_label);

        //auto item_temprature_layout = new QHBoxLayout();
        //item_temprature_layout->addSpacing(18);
        //item_temprature_vertical->addLayout(item_temprature_layout);

        // Value
        item_temprature_vertical->addSpacing(10);
        auto item_temprature_value = new SmartDoubleSpinBox();
        item_temprature_value->setValue(1.00);
        item_temprature_vertical->addWidget(item_temprature_value);
        item_temprature_vertical->addSpacing(0);
    }

    {
        // Effective density
        auto effective_density_vertical = new QHBoxLayout();
        main_layout->addLayout(effective_density_vertical);
        effective_density_vertical->setContentsMargins(0, 0, 0, 0);
        effective_density_vertical->setSpacing(0);

        // Label
        auto effective_density_label = new QLabel("Effective Density");
        effective_density_vertical->addWidget(effective_density_label);

        //auto effective_density_layout = new QHBoxLayout();
        //effective_density_layout->addSpacing(22);
        //effective_densityvertical->addLayout(effective_density_layout);

        // Value
        effective_density_vertical->addSpacing(17);
        auto effective_density_value = new SmartDoubleSpinBox();
        effective_density_value->setValue(1.00);
        effective_density_vertical->addWidget(effective_density_value);
        effective_density_vertical->addSpacing(0);

    }

    main_layout->update();
}

void ItemPropertiesBox::set_selected_item(int32_t selected_object_id)
{
    update_styles();
}

void ItemPropertiesBox::reset_selected_item()
{
    update_styles();
    std::cout << "reset" << std::endl;
}

void ItemPropertiesBox::edit_applying(glm::mat4 matrix)
{
    update_property_fields(matrix);
}

void ItemPropertiesBox::update_styles()
{
    auto& BoltzmannFluids = Application::get();
    auto simulation = BoltzmannFluids.simulation;
    auto& viewport = BoltzmannFluids.main_window.viewport;

    if (viewport->selected_object != SimulationController::not_an_object) {
        setStyleSheet(
            "ItemPropertiesBox QLabel {"
            "font-weight: bold;"
            "background-color: rgb(65, 66, 67);"
            "color: rgb(225, 226, 227);"
            "}"
            "ItemPropertiesBox QDoubleSpinBox {"
            "color: rgb(225, 226, 227);"
            "border: 1px solid rgb(80, 81, 82);"
            "background-color: rgb(85, 86, 87);"
            "}"
            "ItemPropertiesBox QDoubleSpinBox:hover {"
            "background-color: rgb(105, 106, 107);"
            "}"
            );
        velocity_translation_X_box->setEnabled(true);
        velocity_translation_Y_box->setEnabled(true);
        velocity_translation_Z_box->setEnabled(true);
        velocity_angular_X_box->setEnabled(true);
        velocity_angular_Y_box->setEnabled(true);
        velocity_angular_Z_box->setEnabled(true);
        center_of_mass_X_box->setEnabled(true);
        center_of_mass_Y_box->setEnabled(true);
        center_of_mass_Z_box->setEnabled(true);

        position_X_box->setEnabled(true);
        position_Y_box->setEnabled(true);
        position_Z_box->setEnabled(true);
        rotation_X_box->setEnabled(true);
        rotation_Y_box->setEnabled(true);
        rotation_Z_box->setEnabled(true);
        size_X_box->setEnabled(true);
        size_Y_box->setEnabled(true);
        size_Z_box->setEnabled(true);

        update_property_fields(simulation->objects[viewport->selected_object].transform);
    }
    else {
        setStyleSheet(
            "ItemPropertiesBox QLabel {"
            "font-weight: bold;"
            "background-color: rgb(65, 66, 67);"
            "color: rgb(165, 166, 167);"
            "}"
            "ItemPropertiesBox QDoubleSpinBox {"
            "color: rgb(165, 166, 167);"
            "border: 1px solid rgb(60, 61, 62);"
            "background-color: rgb(65, 66, 67);"
            "}"
            "ItemPropertiesBox QDoubleSpinBox:hover {"
            "background-color: rgb(105, 106, 107);"
            "}"
            );
        velocity_translation_X_box->setEnabled(false);
        velocity_translation_Y_box->setEnabled(false);
        velocity_translation_Z_box->setEnabled(false);
        velocity_angular_X_box->setEnabled(false);
        velocity_angular_Y_box->setEnabled(false);
        velocity_angular_Z_box->setEnabled(false);
        center_of_mass_X_box->setEnabled(false);
        center_of_mass_Y_box->setEnabled(false);
        center_of_mass_Z_box->setEnabled(false);


        position_X_box->setEnabled(false);
        position_Y_box->setEnabled(false);
        position_Z_box->setEnabled(false);
        rotation_X_box->setEnabled(false);
        rotation_Y_box->setEnabled(false);
        rotation_Z_box->setEnabled(false);
        size_X_box->setEnabled(false);
        size_Y_box->setEnabled(false);
        size_Z_box->setEnabled(false);

        update_property_fields(glm::identity<glm::mat4>());
    }

}

void ItemPropertiesBox::update_property_fields(glm::mat4 matrix)
{
    auto& BoltzmannFluids = Application::get();
    auto simulation = BoltzmannFluids.simulation;
    auto& viewport = BoltzmannFluids.main_window.viewport;

    if (simulation == nullptr)
        return;

    if(simulation->objects.find(viewport->selected_object) == simulation->objects.end()){
        name_label->setText(QString(QString::fromStdString("")));
    }else{
        int32_t selected_object = viewport->selected_object;
        name_label->setText(QString(QString::fromStdString(simulation->objects[selected_object].name)));

    }


    //enum_to_qstring(item->type, type_label);

    glm::mat4 transform = matrix;
    glm::vec3 scale;
    glm::quat rotation;
    glm::vec3 translation;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(transform, scale, rotation, translation, skew, perspective);
    glm::vec3 rotation_euler = glm::eulerAngles(rotation);

    position_X_box->setValue(translation.x);
    position_Y_box->setValue(translation.y);
    position_Z_box->setValue(translation.z);

    rotation_X_box->setValue(rotation_euler.x);
    rotation_Y_box->setValue(rotation_euler.y);
    rotation_Z_box->setValue(rotation_euler.z);

    size_X_box->setValue(scale.x);
    size_Y_box->setValue(scale.y);
    size_Z_box->setValue(scale.z);
}


void enum_to_qstring(Type& type, QLabel* _string) {
    switch (type) {
    case Type::CUBE:
        _string->setText("Cube");
        break;
    case Type::SPHERE:
        _string->setText("Sphere");
        break;
    case Type::CYLINDER:
        _string->setText("Cylinder");
        break;
    }
}


void ItemPropertiesBox::create_connections()
{
    auto update_transform = [&]() {

        auto& BoltzmannFluids = Application::get();
        auto simulation = BoltzmannFluids.simulation;
        auto& viewport = BoltzmannFluids.main_window.viewport;

        if (viewport->selected_object == 0 || viewport->selected_object > simulation->objects.size()) {
            qDebug() << "selected_object: " << viewport->selected_object;
            return;
        }

        if (viewport->is_edit_happening())
            return;

        glm::quat rotation_new(glm::vec3(rotation_X_box->value(), rotation_Y_box->value(), rotation_Z_box->value()));
        glm::vec3 translation_new(position_X_box->value(), position_Y_box->value(), position_Z_box->value());
        glm::vec3 scale_new(glm::vec3(size_X_box->value(), size_Y_box->value(), size_Z_box->value()));

        std::cout <<"translation new: " <<translation_new.x << std::endl;

        // glm::mat4 transform = recompose(scale, rotation, translation, skew, perspective);

        glm::mat4 transform = glm::scale((glm::translate(glm::identity<glm::mat4>(), translation_new)), scale_new) * glm::mat4_cast(rotation_new);


        simulation->objects[viewport->selected_object].transform = transform;
    };

    std::vector<QDoubleSpinBox*> boxes = {
        position_X_box, position_Y_box, position_Z_box,
        rotation_X_box, rotation_Y_box, rotation_Z_box,
        size_X_box, size_Y_box, size_Z_box
    };

    for (auto* box : boxes) {
        QObject::connect(box, &SmartDoubleSpinBox::valueChanged,
                         this, [=](double) { update_transform(); });
    }

}
