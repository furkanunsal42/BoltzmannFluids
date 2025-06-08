#include "MainWindow.h"
#include "./ui_MainWindow.h"

#include <iostream>

#include <QTextEdit>
#include <QOpenGLWidget>
#include "InitialConditionsBox.h"
#include "CollabsibleBox.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QString qss_text = "";


    // *** Menu Bar ***
    auto menu_bar = menuBar();
//    {
//        auto menu_bar_frame = new QFrame(menu_bar);
//        menu_bar_frame->setFrameStyle(QFrame::HLine | QFrame::Raised);
//        menu_bar->setFixedHeight(26);
//    }
    {   // File
        auto menu_File = new QMenu("File", this);
        menu_bar->addMenu(menu_File);
        {
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
    }
    {   // Edit
        auto menu_Edit = new QMenu("Edit", this);
        menu_bar->addMenu(menu_Edit);
        {
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
    }
    {   // View
        auto menu_View = new QMenu("View", this);
        menu_bar->addMenu(menu_View);
        {
            auto actionShow_Right_Side_Bar  = new QAction("Show Right Side Bar");
            menu_View->addAction(actionShow_Right_Side_Bar);
        }
    }
    {   // Help
        auto menu_Help = new QMenu("Help", this);
        menu_bar->addMenu(menu_Help);
        {
        }
    }
    qss_text += "QMenuBar { "
                    "background-color: rgb(45, 46, 47); "
                    "color: rgb(180, 181, 182); "
                    "border: 1px solid rgb(75, 76, 77); "
                "}"
                "QMenuBar::item { "
                    "padding: 5px 15px; "
                    "background: transparent;"
                "}"
                "QMenuBar::item:selected { "
                    "background-color: rgb(93, 94, 95); "
                "}";

    auto main_layout = new QHBoxLayout(this);
    main_layout->setContentsMargins(0, 0, 0, 0);
    main_layout->setSpacing(0);

    auto left_vertical_layout = new QVBoxLayout();
    main_layout->addLayout(left_vertical_layout);
    {
        auto render_box = new QOpenGLWidget();
        {
            left_vertical_layout->addWidget(render_box);
            render_box->setMinimumWidth(100);
            render_box->setMinimumHeight(100);
            //left_side_panel->setStyleSheet("background-color: rgb(45, 45, 45); color: white;");
        }
        auto application_output = new QTextEdit();
        {
            application_output->setReadOnly(true);
            application_output->setText("asd\nThat is good!");
            application_output->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
            application_output->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            std::cout << application_output->sizeHint().width() << " : " << application_output->sizeHint().height() << std::endl;
            qss_text += "QTextEdit {background-color: rgb(50, 51, 52); color: rgb(180, 181, 182); border: 2px solid gray; border-color: rgb(75, 76, 77);}";
            left_vertical_layout->addWidget(application_output);
        }
    }


    auto right_side_panel = new CollapsibleBox("Initial Conditions");
    {
        // Initial Conditions
        auto initial_conditions = new InitialConditionsBox();
        right_side_panel->addWidget(initial_conditions);
        main_layout->addWidget(right_side_panel);
        // Initial Conditions 2
        auto initial_conditions_2 = new InitialConditionsBox();
        right_side_panel->addWidget(initial_conditions_2);
    }




    // StatusBar
    auto status_bar = new QStatusBar();
    setStatusBar(status_bar);
    qss_text += "QStatusBar { background-color: rgb(60, 61, 62); color: rgb(180, 181, 182); }";

    // Finish
    auto central_widget = new QWidget();
    central_widget->setLayout(main_layout);
    setCentralWidget(central_widget);
    setStyleSheet(qss_text);

}

MainWindow::~MainWindow()
{
    delete ui;
}


#include <QFile>
auto file_new = new QFile();
void MainWindow::on_actionNew_Project_triggered()
{
    std::cout << "triggered!" << std::endl;
}
