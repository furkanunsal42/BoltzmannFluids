#include "TimelineRuler.h"
#include "Timeline.h"

#include <QPainter>
#include <QResizeEvent>

TimelineRuler::TimelineRuler(QWidget* parent, Timeline* parent_timeline)
    :QWidget(parent), parent_timeline(parent_timeline)
{
    setMinimumHeight(40);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setContentsMargins(10,0,10,0);

}

void TimelineRuler::paintEvent(QPaintEvent*) {
    QPainter pen(this);
    pen.fillRect(rect(), QColor(60, 61, 62));

    QPen ruler_pen(Qt::lightGray);
    pen.setPen(ruler_pen);

    int h = height()-1;
    int w = width()-10;

    pen.drawLine(1, h - 1, w+9, h - 1);

    int max_step_count = 10;
    float step_width = w / (max_step_count);
    float step_magnitude = (parent_timeline->get_frame_max() - parent_timeline->get_frame_begin()) / max_step_count;
    float step_x = 0;
    for (int step_index = 0; step_index <= max_step_count ; step_index++) {

        step_x = step_width * step_index;
        pen.drawLine(step_x, (h-1), step_x, (h-7));

        this->label = (QString::number(step_index * step_magnitude));
        pen.drawText(step_x, h - 10, label);

    }
    if (parent_timeline->get_frame_current() >= parent_timeline->get_frame_begin() &&
            parent_timeline->get_frame_current() <= parent_timeline->get_frame_max()) {

        // Simulation box
        QPen blue_pen(Qt::blue);
        QBrush brush(QColor(50, 50, 150, 200), Qt::SolidPattern);
        pen.setPen(blue_pen);
        pen.setBrush(brush);
        float line_x = (w * parent_timeline->get_frame_current()) / (parent_timeline->get_frame_max() - parent_timeline->get_frame_begin());
        pen.drawRect(0, h - 2, line_x, -(h-2));

        // Current Frame
        QPen red_pen(Qt::red);
        pen.setPen(red_pen);
        pen.drawLine(line_x, 0, line_x, h);

    }
}

void TimelineRuler::mousePressEvent(QMouseEvent *event)
{
    if(event->button() == Qt::LeftButton)
    {
        qDebug() << "that tickled";
    }
}
