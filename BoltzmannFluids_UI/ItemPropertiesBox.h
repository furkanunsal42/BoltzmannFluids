#ifndef ITEMPROPERTIESBOX_H
#define ITEMPROPERTIESBOX_H

#include <QWidget>

class QLabel;
class AddableItem;
class QGroupBox;
class SmartDoubleSpinBox;

class ItemPropertiesBox : public QWidget
{
    Q_OBJECT

public:

    explicit ItemPropertiesBox(QWidget* parent = nullptr);

private:
    AddableItem* item;
    QLabel* name_label;
    QLabel* type_label;

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

    SmartDoubleSpinBox* temperature;
    SmartDoubleSpinBox* effective_density;

    QGroupBox* createInitialConditionsGroup();

    bool is_item_selected = false;

    void update_styles();
    void update_property_fields();

public slots:
    void set_selected_item(AddableItem& new_item);
    void reset_selected_item();

};


#endif // ITEMPROPERTIESBOX_H
