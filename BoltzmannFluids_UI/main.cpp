#include "MainWindow.h"

#include <QApplication>
#include <QTimer>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    MainWindow w;
    w.show();

    auto timer = new QTimer();
    int frame = 0;

    QTimer::singleShot(0, &w, [&w, &timer, &frame]() {
        timer->setInterval(1000 / 60*2);
        timer->start();

        QObject::connect(timer, &QTimer::timeout, [&w, &frame] {
            frame+= 1;
            w.update_timeline(frame);
        });
    });

    return a.exec();
}
