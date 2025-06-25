#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "3D/Viewport3D.h"
#include "GUIApplication/SimulationController.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class Timeline;
class InitialConditionsBox;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:

    explicit MainWindow(QWidget *parent = nullptr);

    ~MainWindow();

    void update_timeline(int current_frame);
    Timeline* timeline = nullptr;
    Viewport3D* viewport = nullptr;
    InitialConditionsBox* initial_conditions = nullptr;

private:

    Ui::MainWindow *ui;

};
#endif // MAINWINDOW_H
