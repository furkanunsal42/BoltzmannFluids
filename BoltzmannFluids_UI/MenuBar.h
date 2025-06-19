#ifndef MENUBAR_H
#define MENUBAR_H

#include <QMenuBar>

void init_menu_bar(QMenuBar* menu_bar) {

    {   // File Menu
        auto menu_File = new QMenu("File", menu_bar);
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
        auto menu_Edit = new QMenu("Edit", menu_bar);
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
        auto menu_View = new QMenu("View", menu_bar);
        menu_bar->addMenu(menu_View);

        auto actionShow_Right_Side_Bar  = new QAction("Show Right Side Bar");
        menu_View->addAction(actionShow_Right_Side_Bar);
    }
    {   // Help Menu
        auto menu_Help = new QMenu("Help", menu_bar);
        menu_bar->addMenu(menu_Help);

    }
    menu_bar->setStyleSheet(
        "QMenuBar { "
            "background-color: rgb(40, 40, 40); "
            "color: rgb(200, 201, 202); "
            "border-bottom: 2px solid rgb(22, 22, 22); "
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
        "}");
}

#endif // MENUBAR_H
