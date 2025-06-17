#include "ItemPropertiesBox.h"
#include "UI_Config.h"

#include <QGroupBox>
#include <QVBoxLayout>
#include <cfloat>
#include <qlineedit.h>

void enum_to_qstring(Type& type, QLabel* _string) ;

ItemPropertiesBox::ItemPropertiesBox(QWidget *parent)
    :QWidget(parent)
{
    auto main_layout = new QVBoxLayout(this);
    main_layout->setAlignment(Qt::AlignTop);
    main_layout->setContentsMargins(0,0,0,0);
    main_layout->setSpacing(9);
    setLayout(main_layout);

    QString item_properties_qss;

    /// Initialize AddableItem "item"
    item = new AddableItem(QString("Object"), Type::CUBE, QIcon());

    {   /// Name & Type
        auto name_vertical = new QVBoxLayout();
        name_vertical->setContentsMargins(0, 0, 0, 0);
        name_vertical->setSpacing(0);
        main_layout->addLayout(name_vertical);

        name_label = new QLabel("BoltzmannFluids");
        name_vertical->addWidget(name_label);

        type_label = new QLabel("Object");
        name_vertical->addWidget(type_label);
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
        velocity_translation_horizontal->addStretch();

        /// X
        auto velocity_translation_X_txt = new QLabel("X:");
        velocity_translation_horizontal->addWidget(velocity_translation_X_txt);
        velocity_translation_X_box = new QDoubleSpinBox();
        velocity_translation_X_box->setMinimumWidth(50);
        velocity_translation_X_box->setDecimals(DECIMAL_COUNT);
        velocity_translation_X_box->setRange(POSITION_MIN, POSITION_MAX);
        velocity_translation_X_box->setSingleStep(0.01);
        velocity_translation_horizontal->addWidget(velocity_translation_X_box);
        /// Y
        auto velocity_translation_Y_txt = new QLabel("Y:");
        velocity_translation_horizontal->addWidget(velocity_translation_Y_txt);
        velocity_translation_Y_box = new QDoubleSpinBox();
        velocity_translation_Y_box->setMinimumWidth(50);
        velocity_translation_Y_box->setDecimals(DECIMAL_COUNT);
        velocity_translation_Y_box->setRange(POSITION_MIN, POSITION_MAX);
        velocity_translation_Y_box->setSingleStep(0.01);
        velocity_translation_horizontal->addWidget(velocity_translation_Y_box);
        /// Z
        auto velocity_translation_Z_txt = new QLabel("Z:");
        velocity_translation_horizontal->addWidget(velocity_translation_Z_txt);
        velocity_translation_Z_box = new QDoubleSpinBox();
        velocity_translation_Z_box->setMinimumWidth(50);
        velocity_translation_Z_box->setDecimals(DECIMAL_COUNT);
        velocity_translation_Z_box->setRange(POSITION_MIN, POSITION_MAX);
        velocity_translation_Z_box->setSingleStep(0.01);
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
        velocity_angular_horizontal->addStretch();

        /// X
        auto velocity_angular_X_txt = new QLabel("X:");
        velocity_angular_horizontal->addWidget(velocity_angular_X_txt);
        velocity_angular_X_box = new QDoubleSpinBox();
        velocity_angular_X_box->setMinimumWidth(50);
        velocity_angular_X_box->setDecimals(DECIMAL_COUNT);
        velocity_angular_X_box->setRange(POSITION_MIN, POSITION_MAX);
        velocity_angular_X_box->setSingleStep(0.01);
        velocity_angular_horizontal->addWidget(velocity_angular_X_box);
        /// Y
        auto velocity_angular_Y_txt = new QLabel("Y:");
        velocity_angular_horizontal->addWidget(velocity_angular_Y_txt);
        velocity_angular_Y_box = new QDoubleSpinBox();
        velocity_angular_Y_box->setMinimumWidth(50);
        velocity_angular_Y_box->setDecimals(DECIMAL_COUNT);
        velocity_angular_Y_box->setRange(POSITION_MIN, POSITION_MAX);
        velocity_angular_Y_box->setSingleStep(0.01);
        velocity_angular_horizontal->addWidget(velocity_angular_Y_box);
        /// Z
        auto velocity_angular_Z_txt = new QLabel("Z:");
        velocity_angular_horizontal->addWidget(velocity_angular_Z_txt);
        velocity_angular_Z_box = new QDoubleSpinBox();
        velocity_angular_Z_box->setMinimumWidth(50);
        velocity_angular_Z_box->setDecimals(DECIMAL_COUNT);
        velocity_angular_Z_box->setRange(POSITION_MIN, POSITION_MAX);
        velocity_angular_Z_box->setSingleStep(0.01);
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
        center_of_mass_horizontal->addStretch();

        /// X
        auto center_of_mass_X_txt = new QLabel("X:");
        center_of_mass_horizontal->addWidget(center_of_mass_X_txt);
        center_of_mass_X_box = new QDoubleSpinBox();
        center_of_mass_X_box->setMinimumWidth(50);
        center_of_mass_X_box->setDecimals(DECIMAL_COUNT);
        center_of_mass_X_box->setRange(POSITION_MIN, POSITION_MAX);
        center_of_mass_X_box->setSingleStep(0.01);
        center_of_mass_horizontal->addWidget(center_of_mass_X_box);
        /// Y
        auto center_of_mass_Y_txt = new QLabel("Y:");
        center_of_mass_horizontal->addWidget(center_of_mass_Y_txt);
        center_of_mass_Y_box = new QDoubleSpinBox();
        center_of_mass_Y_box->setMinimumWidth(50);
        center_of_mass_Y_box->setDecimals(DECIMAL_COUNT);
        center_of_mass_Y_box->setRange(POSITION_MIN, POSITION_MAX);
        center_of_mass_Y_box->setSingleStep(0.01);
        center_of_mass_horizontal->addWidget(center_of_mass_Y_box);
        /// Z
        auto center_of_mass_Z_txt = new QLabel("Z:");
        center_of_mass_horizontal->addWidget(center_of_mass_Z_txt);
        center_of_mass_Z_box = new QDoubleSpinBox();
        center_of_mass_Z_box->setMinimumWidth(50);
        center_of_mass_Z_box->setDecimals(DECIMAL_COUNT);
        center_of_mass_Z_box->setRange(POSITION_MIN, POSITION_MAX);
        center_of_mass_Z_box->setSingleStep(0.01);
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
        position_horizontal->addStretch();

        /// X
        auto position_X_txt = new QLabel("X:");
        position_horizontal->addWidget(position_X_txt);
        position_X_box = new QDoubleSpinBox();
        position_X_box->setMinimumWidth(50);
        position_X_box->setDecimals(DECIMAL_COUNT);
        position_X_box->setRange(POSITION_MIN, POSITION_MAX);
        position_X_box->setSingleStep(0.01);
        position_horizontal->addWidget(position_X_box);
        /// Y
        auto position_Y_txt = new QLabel("Y:");
        position_horizontal->addWidget(position_Y_txt);
        position_Y_box = new QDoubleSpinBox();
        position_Y_box->setMinimumWidth(50);
        position_Y_box->setDecimals(DECIMAL_COUNT);
        position_Y_box->setRange(POSITION_MIN, POSITION_MAX);
        position_Y_box->setSingleStep(0.01);
        position_horizontal->addWidget(position_Y_box);
        /// Z
        auto position_Z_txt = new QLabel("Z:");
        position_horizontal->addWidget(position_Z_txt);
        position_Z_box = new QDoubleSpinBox();
        position_Z_box->setMinimumWidth(50);
        position_Z_box->setDecimals(DECIMAL_COUNT);
        position_Z_box->setRange(POSITION_MIN, POSITION_MAX);
        position_Z_box->setSingleStep(0.01);
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
        rotation_horizontal->addStretch();

        /// X
        auto rotation_X_txt = new QLabel("X:");
        rotation_horizontal->addWidget(rotation_X_txt);
        rotation_X_box = new QDoubleSpinBox();
        rotation_X_box->setMinimumWidth(50);
        rotation_X_box->setDecimals(DECIMAL_COUNT);
        rotation_X_box->setRange(POSITION_MIN, POSITION_MAX);
        rotation_X_box->setSingleStep(0.01);
        rotation_horizontal->addWidget(rotation_X_box);
        /// Y
        auto rotation_Y_txt = new QLabel("Y:");
        rotation_horizontal->addWidget(rotation_Y_txt);
        rotation_Y_box = new QDoubleSpinBox();
        rotation_Y_box->setMinimumWidth(50);
        rotation_Y_box->setDecimals(DECIMAL_COUNT);
        rotation_Y_box->setRange(POSITION_MIN, POSITION_MAX);
        rotation_Y_box->setSingleStep(0.01);
        rotation_horizontal->addWidget(rotation_Y_box);
        /// Z
        auto rotation_Z_txt = new QLabel("Z:");
        rotation_horizontal->addWidget(rotation_Z_txt);
        rotation_Z_box = new QDoubleSpinBox();
        rotation_Z_box->setMinimumWidth(50);
        rotation_Z_box->setDecimals(DECIMAL_COUNT);
        rotation_Z_box->setRange(POSITION_MIN, POSITION_MAX);
        rotation_Z_box->setSingleStep(0.01);
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
        size_horizontal->addStretch();

        /// X
        auto size_X_txt = new QLabel("X:");
        size_horizontal->addWidget(size_X_txt);
        size_X_box = new QDoubleSpinBox();
        size_X_box->setMinimumWidth(50);
        size_X_box->setDecimals(DECIMAL_COUNT);
        size_X_box->setRange(POSITION_MIN, POSITION_MAX);
        size_X_box->setSingleStep(0.01);
        size_horizontal->addWidget(size_X_box);
        /// Y
        auto size_Y_txt = new QLabel("Y:");
        size_horizontal->addWidget(size_Y_txt);
        size_Y_box = new QDoubleSpinBox();
        size_Y_box->setMinimumWidth(50);
        size_Y_box->setDecimals(DECIMAL_COUNT);
        size_Y_box->setRange(POSITION_MIN, POSITION_MAX);
        size_Y_box->setSingleStep(0.01);
        size_horizontal->addWidget(size_Y_box);
        /// Z
        auto size_Z_txt = new QLabel("Z:");
        size_horizontal->addWidget(size_Z_txt);
        size_Z_box = new QDoubleSpinBox();
        size_Z_box->setMinimumWidth(50);
        size_Z_box->setDecimals(DECIMAL_COUNT);
        size_Z_box->setRange(POSITION_MIN, POSITION_MAX);
        size_Z_box->setSingleStep(0.01);
        size_horizontal->addWidget(size_Z_box);

        size_vertical->addLayout(size_horizontal);
    }

    main_layout->update();

    update_styles();
    //setStyleSheet(item_properties_qss);
}

void ItemPropertiesBox::set_selected_item(AddableItem& new_item)
{
    qDebug() << "[ItemPropertiesBox] set_selected_item called for" << new_item.name;

    item = &new_item;
    is_item_selected = true;
    update_property_fields();
    update_styles();
}

void ItemPropertiesBox::reset_selected_item()
{
    qDebug() << "[ItemPropertiesBox] reset_selected_item called";

    is_item_selected = false;
    update_property_fields();
    update_styles();
}

void ItemPropertiesBox::update_styles()
{
    if (is_item_selected) {
        setStyleSheet(
            "ItemPropertiesBox QWidget {"
                "color: rgb(225, 226, 227);"
            "}"
            "ItemPropertiesBox QDoubleSpinBox {"
                "color: rgb(225, 226, 227);"
                "border: 1px solid rgb(80, 81, 82);"
                "background-color: rgb(85, 86, 87);"
            "}"
            "ItemPropertiesBox QDoubleSpinBox:hover {"
                "background-color: rgb(110, 111, 112);"
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
    }
    else {
        setStyleSheet(
            "ItemPropertiesBox QDoubleSpinBox {"
                "color: rgb(165, 166, 167);"
                "border: 1px solid rgb(60, 61, 62);"
                "background-color: rgb(65, 66, 67);"
            "}"
            "ItemPropertiesBox QDoubleSpinBox:hover {"
                "background-color: rgb(110, 111, 112);"
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

    }
}

void ItemPropertiesBox::update_property_fields()
{
    if (!item) {
        qDebug() << "(ItemPropertiesBox::update_property_fields) \"item\" is nullptr";
        return;
    }


    name_label->setText(item->name);
    enum_to_qstring(item->type, type_label);

    position_X_box->setValue(item->position.x());
    position_Y_box->setValue(item->position.y());
    position_Z_box->setValue(item->position.z());

    rotation_X_box->setValue(item->rotation.x());
    rotation_Y_box->setValue(item->rotation.y());
    rotation_Z_box->setValue(item->rotation.z());

    size_X_box->setValue(item->size.x());
    size_Y_box->setValue(item->size.y());
    size_Z_box->setValue(item->size.z());
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
