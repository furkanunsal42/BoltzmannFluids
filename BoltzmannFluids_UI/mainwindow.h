#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class Timeline;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:

    explicit MainWindow(QWidget *parent = nullptr);

    ~MainWindow();

    void update_timeline(int current_frame);

private slots:

    void on_actionNew_Project_triggered();


private:

    Timeline* timeline;

    Ui::MainWindow *ui;

};
#endif // MAINWINDOW_H
