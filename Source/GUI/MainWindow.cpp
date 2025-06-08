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

#include "InitialConditionsBox.h"
#include "CollabsibleBox.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QString qss_text;

    // *** Menu Bar ***
    auto menu_bar = menuBar();
    {   // File
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
    {   // Edit
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
    {   // View
        auto menu_View = new QMenu("View", this);
        menu_bar->addMenu(menu_View);

        auto actionShow_Right_Side_Bar  = new QAction("Show Right Side Bar");
        menu_View->addAction(actionShow_Right_Side_Bar);
    }
    {   // Help
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

    // --- Left Layout ---
    auto left_vertical_layout = new QVBoxLayout();
    main_layout->addLayout(left_vertical_layout);
    {
        // Rendering Box with Frame
        auto render_box_frame = new QFrame(central_widget);
        render_box_frame->setFrameStyle(QFrame::StyledPanel);
        render_box_frame->setLineWidth(2);
        render_box_frame->setContentsMargins(0, 0, 0, 0);

        auto render_box_layout = new QVBoxLayout(render_box_frame);
        render_box_layout->setContentsMargins(0, 0, 0, 0);
        render_box_layout->setSpacing(0);

        auto render_box = new QOpenGLWidget(render_box_frame);
        render_box->setMinimumSize(100, 100);
        render_box_layout->addWidget(render_box);

        left_vertical_layout->addWidget(render_box_frame);

        /* //HERE
        qss_text += "QFrame {"
                    "border: 2px solid rgb(75, 76, 77); "
                    "}";
        */
    }

    {
        // Application Output
        auto application_output = new QTextEdit(central_widget);
        application_output->setReadOnly(true);
        application_output->setText("asd\nThat is good!");
        application_output->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
        application_output->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

        left_vertical_layout->addWidget(application_output);

        qss_text += "QTextEdit {"
                    "background-color: rgb(51, 52, 53); "
                    "color: rgb(180, 181, 182); "
                    "border: 2px solid rgb(75, 76, 77); "
                    "}";
    }


    // --- Right Layout ---
    {
        auto right_side_panel = new CollapsibleBox("Initial Conditions");
        right_side_panel->addWidget(new InitialConditionsBox());
        right_side_panel->addWidget(new InitialConditionsBox());

        main_layout->addWidget(right_side_panel);
    }


    // --- Status Bar ---
    {
        auto status_bar = statusBar();
        setStatusBar(status_bar);

        qss_text += "QStatusBar { "
                    "background-color: rgb(60, 61, 62); "
                    "color: rgb(180, 181, 182); "
                    "}";
    }

    // === Apply StyleSheet ===
    setStyleSheet(qss_text);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionNew_Project_triggered()
{
    std::cout << "New Project triggered!" << std::endl;
}
