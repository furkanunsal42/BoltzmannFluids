#include "MainWindow.h"

#include <QApplication>
#include <QTimer>

//#include <GraphicsCortex.h>
#include <LBM/LBM.h>

int main(int argc, char *argv[])
{
 
    QApplication a(argc, argv);

    MainWindow w;
    w.show();

    //auto timer = new QTimer();
    //int frame = 3000;

    //QTimer::singleShot(0, &w, [&w, &timer, &frame]() {
    //    timer->setInterval(1000 / 60);
    //    timer->start();

    //    QObject::connect(timer, &QTimer::timeout, [&w, &frame] {
    //        frame+= 1;
    //        w.update_timeline(frame);
    //    });
    //});

    return a.exec();
}
