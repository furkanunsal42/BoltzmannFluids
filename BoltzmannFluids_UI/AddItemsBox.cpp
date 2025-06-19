#include "AddItemsBox.h"

#include <QVBoxLayout>
#include <QPushButton>

#include "application.h"

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
    auto outer_layout = new QVBoxLayout(this);
    outer_layout->setContentsMargins(0, 0, 0, 0);
    outer_layout->setSpacing(0);

    auto main_frame = new QFrame(this);
    outer_layout->addWidget(main_frame);
    main_frame->setFrameShape(QFrame::StyledPanel);
    QString qss_style_sheet =(
        "QFrame {"
            "background-color: rgb(65, 66, 67); "
            "border: 1px solid rgb(65, 66, 67);"
            "border-radius: 5px;"
        "}"
        "QListView {"
            "color: rgb(225, 226, 227);"
            "background-color: rgb(65, 66, 67);"
            "border: 0px solid rgb(65, 66, 67);"
            "outline: 0;"   // Remove dotted focus border
        "}"
        "QListView::item {"
            "padding: 3px;"
            "border: 1px solid rgb(65, 66, 67);"
            "background-color: rgb(70, 71, 72);"
        "}"
        "QListView::item:selected {"
            "background-color: rgb(90, 91, 92);"
            "color: white;"
        "}"
        "QListView::item:hover {"
            "background-color: rgb(80, 81, 82);"
        "}"
        );

    auto inner_layout = new QVBoxLayout(main_frame);
    main_frame->setLayout(inner_layout);
    inner_layout->setContentsMargins(0, 0, 0, 0);
    //inner_layout->setSpacing(0);

    {   // "Add - Delete" buttons section
        auto buttons_frame = new QFrame(main_frame);
        inner_layout->addWidget(buttons_frame);

        // Style the frame and all QPushButtons inside it
        buttons_frame->setStyleSheet(
            "QFrame {"
                "background-color: rgb(70, 71, 72);"
            "}"
            "QPushButton {"
                "color: rgb(225, 226, 227);"
                "background-color: rgb(85, 86, 87);"
                "border: 1px solid rgb(70, 71, 72);"
                "border-radius: 5px;"
                "padding: 6px;"
                "font-weight: bold;"
            "}"
            "QPushButton:hover {"
                "background-color: rgb(105, 106, 107);"
            "}"
            "QPushButton:pressed {"
                "background-color: rgb(70, 71, 72);"
            "}"
            );

        // Layout
        auto button_layout = new QHBoxLayout(buttons_frame);
        button_layout->setContentsMargins(0, 0, 0, 0);
        buttons_frame->setLayout(button_layout);

        // Add stretch, buttons, stretch
        button_layout->addStretch();

        _add_button = new QPushButton("Add", buttons_frame);
        _add_button->setMinimumSize(QSize(62, 24));
        button_layout->addWidget(_add_button);

        _delete_button = new QPushButton("Delete", buttons_frame);
        _delete_button->setMinimumSize(QSize(62, 24));
        button_layout->addWidget(_delete_button);

        button_layout->addStretch();
    }


    _model = new AddableItemModel(this);
    _items_list = new QListView(this);
    inner_layout->addWidget(_items_list);
    _items_list->setSelectionMode(QAbstractItemView::SingleSelection);
    _items_list->setSelectionBehavior(QAbstractItemView::SelectRows);   // or SelectItems


    // Default items
    std::vector<AddableItem> items;

    items.emplace_back("Cube", Type::CUBE, QIcon(":/icons/cube_icon.png"));
    items.emplace_back("Sphere", Type::SPHERE, QIcon(":/icons/sphere_icon.png"));
    items.emplace_back("Cylinder", Type::CYLINDER, QIcon(":/icons/cylinder_icon.png"));

    _model->set_items(items);
    _items_list->setModel(_model);

    connect(_add_button, &QPushButton::clicked, this, [this]() {
        QModelIndex index = _items_list->currentIndex();
        if (index.isValid()){
            emit add_item_request(_model->item_at(index.row()));
            Application& BoltzmannFluids = Application::get();
            auto simulation = BoltzmannFluids.simulation;
            if(simulation != nullptr && index.row() < 3 && index.row() >= 0){
                std::cout << (SimulationController::BasicObject)index.row() << std::endl;
                simulation->add_object(
                    "Object",
                    (SimulationController::BasicObject)(index.row() + 1)
                    );
            }
        }


    });

    connect(_delete_button, &QPushButton::clicked, this, [this]() {
        QModelIndex index = _items_list->currentIndex();
        if (index.isValid())
            emit delete_item_request(_model->item_at(index.row()));

        Application& BoltzmannFluids = Application::get();
        auto simulation = BoltzmannFluids.simulation;
        auto viewport = BoltzmannFluids.main_window.viewport;
        if(simulation != nullptr && viewport->selected_object != SimulationController::not_an_object){
            simulation->objects.erase(viewport->selected_object);
        }

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

    setStyleSheet(qss_style_sheet);
}
