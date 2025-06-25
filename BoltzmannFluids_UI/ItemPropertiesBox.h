#ifndef ITEMPROPERTIESBOX_H
#define ITEMPROPERTIESBOX_H

#include <glm.hpp>
#include <QWidget>

class QLabel;
class QCheckBox;
class AddableItem;
class QGroupBox;
class SmartDoubleSpinBox;

class ItemPropertiesBox : public QWidget
{
    Q_OBJECT

public:

    explicit ItemPropertiesBox(QWidget* parent = nullptr);

    void create_connections();


//private:

    QLabel* name_label;
    QLabel* type_label;

    QCheckBox* is_boundary;

    SmartDoubleSpinBox* force_X;
    SmartDoubleSpinBox* force_Y;
    SmartDoubleSpinBox* force_Z;

    SmartDoubleSpinBox* velocity_translation_X_box;
    SmartDoubleSpinBox* velocity_translation_Y_box;
    SmartDoubleSpinBox* velocity_translation_Z_box;

    SmartDoubleSpinBox* velocity_angular_X_box;
    SmartDoubleSpinBox* velocity_angular_Y_box;
    SmartDoubleSpinBox* velocity_angular_Z_box;

    SmartDoubleSpinBox* center_of_mass_X_box;
    SmartDoubleSpinBox* center_of_mass_Y_box;
    SmartDoubleSpinBox* center_of_mass_Z_box;

    SmartDoubleSpinBox* position_X_box;
    SmartDoubleSpinBox* position_Y_box;
    SmartDoubleSpinBox* position_Z_box;

    SmartDoubleSpinBox* rotation_X_box;
    SmartDoubleSpinBox* rotation_Y_box;
    SmartDoubleSpinBox* rotation_Z_box;

    SmartDoubleSpinBox* size_X_box;
    SmartDoubleSpinBox* size_Y_box;
    SmartDoubleSpinBox* size_Z_box;

    SmartDoubleSpinBox* item_temprature_value;
    SmartDoubleSpinBox* effective_density_value;

    QGroupBox* createInitialConditionsGroup();

    void update_styles();
    void update_property_fields(glm::mat4 matrix);

public slots:

    void set_selected_item(int32_t selected_object_id);
    void reset_selected_item();

    void edit_applying(glm::mat4 _composed_matrix);

private:
    void set_all_widgets(bool value);
    void set_default_values();

};

#endif // ITEMPROPERTIESBOX_H
