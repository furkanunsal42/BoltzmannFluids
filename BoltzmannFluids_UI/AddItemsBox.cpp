#include "AddItemsBox.h"

#include <QVBoxLayout>

AddableItem::AddableItem(QString name, Type type, QIcon icon, QVector3D position,  QVector3D rotation, QVector3D size)
    : name(std::move(name)), type(type), icon(std::move(icon)), position(position), rotation(rotation), size(size)
{
}

AddableItemModel::AddableItemModel(QObject* parent)
    : QAbstractListModel(parent)
{
}

int AddableItemModel::rowCount(const QModelIndex&) const {
    return static_cast<int>(_items.size());
}

QVariant AddableItemModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() >= static_cast<int>(_items.size()))
        return {};

    const AddableItem& item = _items[index.row()];

    if (role == Qt::DisplayRole)
        return item.name;
    else if (role == Qt::DecorationRole)
        return item.icon;

    return {};
}

void AddableItemModel::set_items(const std::vector<AddableItem>& items) {
    beginResetModel();
    _items = items;
    endResetModel();
}

AddableItem& AddableItemModel::item_at(int index) {
    if (index >= static_cast<int>(_items.size())) {
        qDebug() << "(AddableItemModel::item_at) index is not valid for list";
        return _items[static_cast<int>(_items.size())-1];
    }

    return _items[index];
}


AddItemsBox::AddItemsBox(QWidget *parent)
    :QWidget(parent)
{
    auto main_layout = new QVBoxLayout(this);
    setLayout(main_layout);

    {   // "Add - Delete" buttons
        auto button_layout = new QHBoxLayout;
        main_layout->addLayout(button_layout);

        _add_button = new QPushButton("Add", this);
        button_layout->addWidget(_add_button);

        _delete_button = new QPushButton("Delete", this);
        button_layout->addWidget(_delete_button);
    }

    _model = new AddableItemModel(this);
    _items_list = new QListView(this);
    _items_list->setSelectionMode(QAbstractItemView::SingleSelection);
    _items_list->setSelectionBehavior(QAbstractItemView::SelectRows);   // or SelectItems
    main_layout->addWidget(_items_list);


    // Default items
    std::vector<AddableItem> items;

    items.emplace_back("Cube", Type::CUBE, QIcon(":/icons/cube_icon.png"));
    items.emplace_back("Sphere", Type::SPHERE, QIcon(":/icons/sphere_icon.png"));
    items.emplace_back("Cylinder", Type::CYLINDER, QIcon(":/icons/cylinder_icon.png"));

    _model->set_items(items);
    _items_list->setModel(_model);

    connect(_add_button, &QPushButton::clicked, this, [this]() {
        QModelIndex index = _items_list->currentIndex();
        if (index.isValid())
            emit add_item_request(_model->item_at(index.row()));
    });

    connect(_delete_button, &QPushButton::clicked, this, [this]() {
        QModelIndex index = _items_list->currentIndex();
        if (index.isValid())
            emit delete_item_request(_model->item_at(index.row()));
    });

    connect(_items_list->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, [this](const QItemSelection& selected, const QItemSelection& deselected)
    {
        bool has_selection = !selected.indexes().isEmpty();
        _add_button->setEnabled(has_selection);
        _delete_button->setEnabled(has_selection);

        if (has_selection) {
            QModelIndex selected_index = selected.indexes().first();
            qDebug() << "[AddItemsBox] emitting item_selected";
            auto& item = _model->item_at(selected_index.row());
            emit item_selected(item);
        } else {
            qDebug() << "[AddItemsBox] emitting item_deselected";
            emit item_deselected();
        }
    });

    setStyleSheet(
        "QWidget {"
            "color: rgb(225, 226, 227);"
        "}"
        );

}
