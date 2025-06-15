#include "TimelineRuler.h"

#include <QPainter>
#include <QResizeEvent>

TimelineRuler::TimelineRuler(QWidget *parent)
    :QWidget(parent)
{
    setMinimumHeight(40);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setContentsMargins(10,0,10,0);

    _recompute_ruler();
}

void TimelineRuler::update_timeline(int current_frame)
{
    _frame_current = current_frame;
    update();
}

void TimelineRuler::set_frame_range(int begin, int end)
{
    _frame_begin    = begin;
    _frame_end      = end;
}


void TimelineRuler::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), QColor(60, 61, 62));

    QPen pen(Qt::lightGray);
    p.setPen(pen);

    int h = height()-1;
    int w = width()-10;

    p.drawLine(0, h - 1, w, h - 1);

    int max_step_count = 10;
    float step_width = w / (max_step_count);
    float step_magnitude = (_frame_end - _frame_begin) / max_step_count;
    float step_x = 0;
    for (int step_index = 0; step_index <= max_step_count ; step_index++) {

        step_x = step_width * step_index;
        p.drawLine(step_x, (h-1), step_x, (h-7));

        this->label = (QString::number(step_index * step_magnitude));
        p.drawText(step_x, h - 10, label);

    }
    if (_frame_current >= _frame_begin && _frame_current <= _frame_end) {
            QPen red(Qt::red);
            p.setPen(red);
            float line_x = (w * _frame_current) / (_frame_end - _frame_begin);
            p.drawLine(line_x, 0, line_x, h);
        }
}


void TimelineRuler::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    _recompute_ruler();
}

void TimelineRuler::_recompute_ruler()
{
    _ticks.clear();
    _labels.clear();

    int span = _frame_current - _frame_begin;
    if (span <= 0) return;

    int step = span / 3;

    for (int i = 0; i < span; i += step) {
        _ticks.append(i);
        _labels.append(QString::number(i));
    }

}
