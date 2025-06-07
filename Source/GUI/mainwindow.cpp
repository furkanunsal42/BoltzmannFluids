#include "MainWindow.h"
#include "./ui_MainWindow.h"

#include <QTextEdit>
#include "RightSidePanel.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QString qss_text = "";

    auto main_layout = new QHBoxLayout(this);

    auto left_vertical_layout = new QVBoxLayout();
    main_layout->addLayout(left_vertical_layout);
    {
        auto left_side_panel = new QWidget();
        {
            left_vertical_layout->addWidget(left_side_panel);
            left_side_panel->setMinimumWidth(300);
            left_side_panel->setMinimumHeight(400);
            //left_side_panel->setStyleSheet("background-color: rgb(45, 45, 45); color: white;");
        }
        auto application_output = new QTextEdit();
        {
            application_output->setReadOnly(true);
            application_output->setText("asd\nThat is good!");
            qss_text += "QTextEdit {background-color: rgb(50, 51, 52); color: rgb(180, 181, 182); }";
            left_vertical_layout->addWidget(application_output);
        }
    }

    auto right_side_panel = new RightSidePanel();
    main_layout->addWidget(right_side_panel);

    // MenuBar
    qss_text += "QMenuBar { background-color: rgb(45, 46, 47); color: rgb(180, 181, 182); }"
                "QMenuBar::item { padding: 5px 15px; background: transparent; }"
                "QMenuBar::item:selected { background-color: rgb(93, 94, 95); }";

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
