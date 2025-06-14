#include "MainWindow.h"

#include <QApplication>

#include <QTimer>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    MainWindow w;
    w.show();


    QTimer::singleShot(0, &w, [&w]() {
        w.update_timeline(50);
    });

    return a.exec();
}
