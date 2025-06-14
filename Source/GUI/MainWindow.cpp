#include "MainWindow.h"
#include "./ui_MainWindow.h"

#include <iostream>

#include <QMenu>
#include <QTextEdit>
#include <QOpenGLWidget>
#include <QStatusBar>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFrame>
#include <QSplitter>
#include <QScrollArea>
#include <QLabel>

#include "InitialConditionsBox.h"
#include "CollabsibleBox.h"
#include "UI_Config.h"
#include "Timeline.h"
#include "LBMWrapper.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QString qss_text;

    this->resize(main_window_width, main_window_height);

    // *** Menu Bar ***
    auto menu_bar = menuBar();
    {   // File Menu
        auto menu_File = new QMenu("File", this);
        menu_bar->addMenu(menu_File);

        auto actionNew_Project  = new QAction("New Project");
        auto actionOpen_Project = new QAction("Open Project");
        auto actionSave         = new QAction("Save");
        auto actionSave_as      = new QAction("Save as...");

        menu_File->addAction(actionNew_Project);
        menu_File->addAction(actionOpen_Project);
        menu_File->addSeparator();
        menu_File->addAction(actionSave);
        menu_File->addAction(actionSave_as);
    }
    {   // Edit Menu
        auto menu_Edit = new QMenu("Edit", this);
        menu_bar->addMenu(menu_Edit);

        auto actionUndo         = new QAction("Undo");
        auto actionRedo         = new QAction("Redo");
        auto actionCut          = new QAction("Cut");
        auto actionCopy         = new QAction("Copy");
        auto actionPaste        = new QAction("Paste");
        menu_Edit->addAction(actionUndo);
        menu_Edit->addAction(actionRedo);
        menu_Edit->addSeparator();
        menu_Edit->addAction(actionCut);
        menu_Edit->addAction(actionCopy);
        menu_Edit->addAction(actionPaste);
    }
    {   // View Menu
        auto menu_View = new QMenu("View", this);
        menu_bar->addMenu(menu_View);

        auto actionShow_Right_Side_Bar  = new QAction("Show Right Side Bar");
        menu_View->addAction(actionShow_Right_Side_Bar);
    }
    {   // Help Menu
        auto menu_Help = new QMenu("Help", this);
        menu_bar->addMenu(menu_Help);

    }
        qss_text += "QMenuBar { "
                        "background-color: rgb(45, 46, 47); "
                        "color: rgb(180, 181, 182); "
                        "border-bottom: 1px solid rgb(75, 76, 77); "
                    "}"
                    "QMenuBar::item { "
                        "padding: 5px 15px; "
                        "background: transparent; "
                    "}"
                    "QMenuBar::item:selected { "
                        "background-color: rgb(93, 94, 95); "
                    "}"
                    "QMenu {"
                        "background-color: rgb(55, 56, 57); "
                        "color: rgb(200, 200, 200); "
                        "border: 1px solid rgb(75, 76, 77);"
                    "}"
                    "QMenu::item:selected {"
                        "background-color: rgb(93, 94, 95);"
                    "}";


    // *** Central Layout ***
    auto central_widget = centralWidget();
    auto main_layout = new QHBoxLayout(central_widget);
    main_layout->setContentsMargins(0, 0, 0, 0);
    main_layout->setSpacing(0);

    qss_text += "QWidget { "
                    "background-color: rgb(50, 51, 52); "
                    "color: rgb(180, 181, 182); "
                "}"
                "QScrollArea {"
                    "border: 2px solid rgb(75, 76, 77); "
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
                "}";


    auto main_splitter = new QSplitter(Qt::Horizontal, central_widget);
    main_layout->addWidget(main_splitter);

    // --- Left Panel ---
    {
        auto left_scroll_area = new QScrollArea(central_widget);
        left_scroll_area->setWidgetResizable(true); // Makes inner widget resize properly
        left_scroll_area->setMinimumWidth(100);
        left_scroll_area->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

        auto left_panel_layout = new QVBoxLayout(left_scroll_area);
        left_scroll_area->setLayout(left_panel_layout);
        //left_panel_layout->setContentsMargins(0, 0, 0, 0);   // TODO
        left_panel_layout->setSpacing(2);                    // TODO
        {
            auto add_items_box = new CollapsibleBox("Add Items", left_scroll_area);
            left_panel_layout->addWidget(add_items_box);

            auto label1 = new QLabel("box1", add_items_box);
            add_items_box->addWidget(label1);
        }

        left_panel_layout->addStretch();
        main_splitter->addWidget(left_scroll_area);


        left_scroll_area->setStyleSheet(
            "QWidget { "
                "background-color: rgb(50, 51, 52); "
                "color: rgb(180, 181, 182); "
            "}"
            "QScrollArea {"
                "border: 2px solid rgb(75, 76, 77); "
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
            "}");
    }

    // --- Middle Panel ---
    auto middle_splitter_one = new QSplitter(Qt::Vertical, central_widget);
    main_splitter->addWidget(middle_splitter_one);

    {
        //auto middle_panel = new QWidget(central_widget);
        auto middle_vertical_layout = new QVBoxLayout(middle_splitter_one);
        middle_vertical_layout->setContentsMargins(0, 0, 0, 0);
        middle_vertical_layout->setSpacing(0);

        {
            // Rendering Box
            auto render_box = new QOpenGLWidget(middle_splitter_one);
            render_box->setMinimumSize(100, 100);

            //wrapper = new QT_LBMWrapper

            middle_vertical_layout->addWidget(render_box , 1); // Renderbox ->growable
        }
        {
            // --- Timeline ---
            this->timeline = new Timeline(1000, middle_splitter_one);
            timeline->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            qDebug() << "Constructing Timeline, this:" << this;         ////////////////////
            this->timeline->setMinimumHeight(75);
            middle_vertical_layout->addWidget(timeline, 0); // Timeline  ->fixed size
        }
        {
            // Application Output
            auto application_output = new QTextEdit(middle_splitter_one);
            application_output->setReadOnly(true);
            application_output->setText("asd\nThat is good!");
            application_output->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
            application_output->setMinimumHeight(100);

            middle_vertical_layout->addWidget(application_output, 0);  // App output ->fixed size


            qss_text += "QTextEdit {"
                        "background-color: rgb(51, 52, 53); "
                        "color: rgb(200, 201, 202); "
                        "border: 2px solid rgb(75, 76, 77); "
                        "}";

        }

        middle_splitter_one->setSizes({render_box_height,
                                 timeline_height,
                                 application_output_height});

        middle_splitter_one->setStretchFactor(0, 1); // Left panel    ->growable
        middle_splitter_one->setStretchFactor(1, 0); // Middle panel  ->fixed size
        middle_splitter_one->setStretchFactor(2, 0); // Right panel   ->fixed size

        middle_splitter_one->setCollapsible(0, false);
        middle_splitter_one->setCollapsible(1, false);
        middle_splitter_one->setCollapsible(2, false);
    }

    // --- Right Panel ---
    {
        auto right_scroll_area = new QScrollArea(central_widget);
        right_scroll_area->setWidgetResizable(true); // Makes inner widget resize properly
        right_scroll_area->setMinimumWidth(100);
        right_scroll_area->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        //right_scroll_area->setMinimumWidth(120);

        auto scroll_content = new QWidget(right_scroll_area);
        auto scroll_layout = new QVBoxLayout(scroll_content);
        scroll_layout->setContentsMargins(10, 3, 3, 3);
        scroll_layout->setSpacing(3);

        auto initial_conditions_box = new CollapsibleBox("Initial Conditions - 1", scroll_content);
        //initial_conditions_box->setObjectName("initial_conditions_box");
        initial_conditions_box->addWidget(new InitialConditionsBox());
        scroll_layout->addWidget(initial_conditions_box);

        scroll_layout->addStretch();
        scroll_content->setLayout(scroll_layout);
        right_scroll_area->setWidget(scroll_content);
        main_splitter->addWidget(right_scroll_area);

        initial_conditions_box->setStyleSheet(
            "background-color: rgb(65, 66, 67);"
            "border: 1px solid rgb(65, 66, 67);"
            "border-radius: 5px;"
            "padding: 1px;"
            );

    //    qss_text +=
    //        "#initial_conditions_box { "
    //            "background-color: rgb(165, 66, 67);"
    //            "border 1px solid rgb(165, 66, 67);"
    //            "border-radius: 8px;"
    //            "padding: 4px;"  // internal space so children don't leak
    //        "}"
    //        ;
    }


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
    qDebug() << "Calling set_frame on Timeline:" << this;
    this->timeline->set_frame(current_frame);
}

void MainWindow::on_actionNew_Project_triggered()
{
    std::cout << "New Project triggered!" << std::endl;
}
