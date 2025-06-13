#include "Timeline.h"

#include <QPainter>
#include <QMouseEvent>

Timeline::Timeline(int max_frame, QWidget *parent)
    : _max_frame(max_frame), QWidget(parent)
{
    _timer.setInterval(_frame_interval_ms);

    connect(&_timer, &QTimer::timeout, this, &Timeline::_advance);

    setMinimumHeight(40);
}

void Timeline::start()
{
    if (!_timer.isActive())
        _timer.start();
}

void Timeline::pause()
{
    _timer.stop();
}

void Timeline::stop()
{
    _timer.stop();
    _frame = start_frame;
    update();
    emit frame_changed(_frame);
    emit finished();
}

void Timeline::set_frame(int frame)
{
    _frame = frame;
}

int Timeline::get_current_frame() const
{
    return _frame;
}

int Timeline::get_max_frame() const
{
    return _max_frame;
}

void Timeline::set_max_frame(int new_max_frame)
{
    _max_frame = new_max_frame < 10000 ? new_max_frame : 10000;
    if (_frame > _max_frame)
        set_frame(_max_frame);
    update();
}

void Timeline::paintEvent(QPaintEvent *event)
{

}

void Timeline::mousePressEvent(QMouseEvent *event)
{

}

void Timeline::_advance()
{
    if (_frame < _max_frame) {
        _frame++;
        update();
        emit frame_changed(_frame);
    }
    else {
        stop();
    }

}
