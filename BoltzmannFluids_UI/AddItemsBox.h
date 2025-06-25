#ifndef ADDITEMSBOX_H
#define ADDITEMSBOX_H

#include <qvectornd.h>
#include <QListWidget>

class QPushButton;

enum class Type {
    CUBE,
    SPHERE,
    CYLINDER,
};

struct AddableItem {
    AddableItem(
        QString name,
        Type type,
        QIcon icon,
        QVector3D position  = QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D rotation  = QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D size      = QVector3D(1.0f, 1.0f, 1.0f)
        );

    QString name;
    Type type;
    QIcon icon;
    QVector3D position;
    QVector3D rotation;
    QVector3D size;
};

class AddableItemModel : public QAbstractListModel {
    Q_OBJECT

public:
    AddableItemModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;

    void set_items(const std::vector<AddableItem>& items);
    AddableItem& item_at(int index);

private:
    std::vector<AddableItem> _items;

};


class AddItemsBox : public QWidget {
    Q_OBJECT

public:
    explicit AddItemsBox(QWidget* parent = nullptr);

    signals:
        void add_item_request(const AddableItem& item);
        void delete_item_request(const AddableItem& item);

        void item_selected(AddableItem& item);
        void item_deselected();

private:
    QListView* _items_list;
    QPushButton* _add_button;
    QPushButton* _delete_button;
    AddableItemModel* _model;

};

#endif // ADDITEMSBOX_H
