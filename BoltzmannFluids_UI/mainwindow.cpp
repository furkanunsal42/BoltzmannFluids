#include "MainWindow.h"
#include "./ui_MainWindow.h"
#include "AddItemsBox.h"
#include "InitialConditionsBox.h"
#include "CollabsibleBox.h"
#include "UI_Config.h"
#include "Timeline.h"
#include "ItemPropertiesBox.h"
#include "MenuBar.h"
#include "Viewport3D.h"

#include <QTextEdit>
#include <QOpenGLWidget>
#include <QStatusBar>
#include <QHBoxLayout>
#include <QFrame>
#include <QSplitter>
#include <QScrollArea>
#include <QGraphicsDropShadowEffect>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    // Setup
    ui->setupUi(this);

    QString qss_text;
    this->resize(main_window_width, main_window_height);

    // *** Menu Bar ***
    auto menu_bar = menuBar();
    init_menu_bar(menu_bar);

    // *** Central Layout ***
    auto central_widget = centralWidget();
    auto main_layout = new QHBoxLayout(central_widget);
    main_layout->setContentsMargins(0, 0, 0, 0);
    main_layout->setSpacing(0);

    qss_text +=
        "QWidget { "
            "background-color: rgb(22, 22, 22);"
            "color: rgb(180, 181, 182); "
            "border: 0px solid rgb(22, 22, 22);"
        "}"
        "QFrame { "
            "background-color: rgb(50, 51, 52); "
            "border: 1px solid rgb(50, 51, 52);"
            "border-radius: 8px;"
        "}"
        "QScrollArea {"
            "background-color: rgb(50, 51, 52); "
            "border-left:   1px solid rgb(22, 22, 22);" // WAS 75, 76, 77
            "border-bottom: 1px solid rgb(22, 22, 22);" // WAS 75, 76, 77
            "border-right:  1px solid rgb(22, 22, 22);" // WAS 75, 76, 77
            "border-top: none;"
            "border-radius: 8px;"
        "}"
        "QDoubleSpinBox {"
            "border: 2px solid rgb(50, 51, 52); "
        "}"
        "QCheckBox::indicator {"
            "width: 12px;"
            "height: 12px;"
        "}"
        "QCheckBox::indicator:checked {"
            "image: url(:/qt_icons/checkbox_checked3.png);"
        "}"
        "QCheckBox::indicator:unchecked {"
            "image: url(:/qt_icons/checkbox_unchecked2.png);"
        "}"
        "QSplitter {"
            "background-color: transparent;"
        "}"
        "QTextEdit {"
            "background-color: rgb(60, 61, 62);"
            "color: white;"
            "border: 1px solid rgb(75, 76, 77);"
        "}"
        ;

    auto main_splitter = new QSplitter(Qt::Horizontal, central_widget);
    main_splitter->setHandleWidth(1);
    main_layout->addWidget(main_splitter);


    // --- Left Panel ---

    auto left_scroll_area = new QScrollArea(central_widget);
    main_splitter->addWidget(left_scroll_area);
    left_scroll_area->setWidgetResizable(true);
    left_scroll_area->setMinimumWidth(100);
    left_scroll_area->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

    auto left_scroll_content = new QWidget(left_scroll_area);
    left_scroll_area->setWidget(left_scroll_content);
    left_scroll_content->setStyleSheet(
        "background-color: rgb(50, 51, 52);"
        "border-radius: 8px;"
        "padding: 1px;"
        );

    auto content_layout = new QVBoxLayout(left_scroll_content);
    content_layout->setContentsMargins(3, 3, 8, 3);
    content_layout->setSpacing(0);
    left_scroll_content->setLayout(content_layout);

    auto inner_splitter = new QSplitter(Qt::Vertical, left_scroll_content);
    content_layout->addWidget(inner_splitter);

    // Add Items Box
    auto add_items_box = new AddItemsBox(inner_splitter);
    inner_splitter->addWidget(add_items_box);

    // Item Properties Box
    auto item_properties_collapsible_box = new CollapsibleBox("Item Properties", inner_splitter);
    auto item_properties_box = new ItemPropertiesBox(inner_splitter);
    item_properties_collapsible_box->add_widget(item_properties_box);
    inner_splitter->addWidget(item_properties_collapsible_box);
    //content_layout->addStretch();

    inner_splitter->setStretchFactor(0, 0); // add items box    ->fixed size
    inner_splitter->setStretchFactor(1, 1); // item properties  ->growable

    inner_splitter->setCollapsible(0, false);
    inner_splitter->setCollapsible(1, false);


    item_properties_collapsible_box->setStyleSheet(
        "background-color: rgb(65, 66, 67);"
        "border: 1px solid rgb(65, 66, 67);"
        "border-radius: 5px;"
        "padding: 1px;"
        );

    /// Connect item_properties_box to add_items_box
    QObject::connect(add_items_box, &AddItemsBox::item_deselected, item_properties_box, &ItemPropertiesBox::reset_selected_item);
    QObject::connect(add_items_box, &AddItemsBox::item_selected, item_properties_box, &ItemPropertiesBox::set_selected_item);


    left_scroll_area->setStyleSheet(
        "QWidget { "
            "background-color: rgb(22, 22, 22); "
            "color: rgb(180, 181, 182); "
        "}"
        "QScrollArea {"
            "background-color: rgb(50, 51, 52); "
            "border-left:   1px solid rgb(22, 22, 22);" // WAS 75, 76, 77
            "border-bottom: 1px solid rgb(22, 22, 22);" // WAS 75, 76, 77
            "border-right:  1px solid rgb(22, 22, 22);" // WAS 75, 76, 77
            "border-top: none;"
            "border-radius: 8px;"
        "}"
        "QCheckBox::indicator {"
            "width: 12px;"
            "height: 12px;"
        "}"
        "QCheckBox::indicator:checked {"
            "image: url(:/qt_icons/checkbox_checked3.png);"
        "}"
        "QCheckBox::indicator:unchecked {"
            "image: url(:/qt_icons/checkbox_unchecked2.png);"
        "}"
        );


    // --- Middle Panel ---
    //auto middle_frame = new QFrame(central_widget);

    // Vertical splitter directly inside middle_frame
    auto middle_splitter = new QSplitter(Qt::Vertical, main_splitter);
    main_splitter->addWidget(middle_splitter);
    middle_splitter->setHandleWidth(1);
    middle_splitter->setStyleSheet(
        "QFrame { "
            "background-color: rgb(50, 51, 52); "
            "border: 0px solid rgb(50, 51, 52);"
            "border-radius: 8px;"
        "}"
        "QTextEdit { "
            "background-color: rgb(50, 51, 52); "
            //"border: 1px solid rgb(22, 22, 22);"
            "border-bottom: 1px solid rgb(22, 22, 22);"
            "border-radius: 8px;"
        "}"
        "QSplitter {"
            "background-color: rgb(50, 51, 52);"
            "border: 0px solid rgb(50, 51, 52);"
        "}"
        );

    // Layout to hold the splitter
    auto middle_vertical_layout = new QVBoxLayout(main_splitter);
    middle_vertical_layout->setContentsMargins(0, 0, 0, 0);
    middle_vertical_layout->setSpacing(0);
    middle_vertical_layout->addWidget(middle_splitter);


    // Add widgets directly to splitter
    auto render_box = new QOpenGLWidget();
    render_box->setMinimumSize(100, 100);
    //middle_splitter->addWidget(render_box);

    /// Rendering Box
    viewport = new Viewport3D(middle_splitter);
    viewport->setMinimumSize(100, 100);


    middle_splitter->addWidget(viewport); // Renderbox ->growable


    timeline = new Timeline();
    timeline->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    timeline->setMinimumHeight(75);
    middle_splitter->addWidget(timeline);

    auto application_output = new QTextEdit();
    application_output->setReadOnly(true);
    application_output->setText("Welcome to the BoltzmannFluids");
    application_output->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
    application_output->setMinimumHeight(100);
    middle_splitter->addWidget(application_output);

    // Set sizes and behavior
    middle_splitter->setSizes({render_box_height, timeline_height, application_output_height});
    middle_splitter->setStretchFactor(0, 1);
    middle_splitter->setStretchFactor(1, 0);
    middle_splitter->setStretchFactor(2, 0);

    middle_splitter->setCollapsible(0, false);
    middle_splitter->setCollapsible(1, false);
    middle_splitter->setCollapsible(2, false);


    // --- Right Panel ---

    auto right_scroll_area = new QScrollArea(central_widget);
    main_splitter->addWidget(right_scroll_area);
    right_scroll_area->setWidgetResizable(true); // Makes inner widget resize properly
    right_scroll_area->setMinimumWidth(100);
    right_scroll_area->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    //right_scroll_area->setMinimumWidth(120);

    auto right_scroll_content = new QWidget(right_scroll_area);
    right_scroll_area->setWidget(right_scroll_content);
    right_scroll_content->setStyleSheet(
        "background-color: rgb(50, 51, 52);"
        "border-radius: 8px;"
        "padding: 1px;"
        );

    auto scroll_layout = new QVBoxLayout(right_scroll_content);
    scroll_layout->setContentsMargins(8, 3, 3, 3);
    scroll_layout->setSpacing(3);
    right_scroll_content->setLayout(scroll_layout);


    /// Initial Conditions Box
    auto initial_conditions_box = new CollapsibleBox("Initial Conditions - 1", right_scroll_content);
    //initial_conditions_box->setObjectName("initial_conditions_box");
    initial_conditions_box->add_widget(new InitialConditionsBox(initial_conditions_box));
    scroll_layout->addWidget(initial_conditions_box);

    initial_conditions_box->setStyleSheet(
        "background-color: rgb(65, 66, 67);"
        "border: 1px solid rgb(65, 66, 67);"
        "border-radius: 5px;"
        "padding: 1px;"
        );



    scroll_layout->addStretch();


    /// Final touches
    main_splitter->setSizes({left_panel_width,
                             middle_panel_width,
                             right_panel_width});

    main_splitter->setStretchFactor(0, 0); // Left panel    ->fixed size
    main_splitter->setStretchFactor(1, 1); // Middle panel  ->growable
    main_splitter->setStretchFactor(2, 0); // Right panel   ->fixed size

    main_splitter->setCollapsible(0, false);
    main_splitter->setCollapsible(1, false);
    main_splitter->setCollapsible(2, false);

    // --- Status Bar ---
    {
        auto status_bar = statusBar();
        setStatusBar(status_bar);

        qss_text += "QStatusBar { "
                        "color: rgb(180, 181, 182); "
                        "background-color: rgb(60, 61, 62); "
                    "}"
            ;
    }

    // === Apply StyleSheet ===
    setStyleSheet(qss_text);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::update_timeline(int current_frame)
{
    this->timeline->set_frame_current(current_frame);
    this->timeline->set_frame_simulation_duration(current_frame);
}
