#include "TimelineRuler.h"

#include <QPainter>
#include <QResizeEvent>

TimelineRuler::TimelineRuler(QWidget *parent)
    :QWidget(parent)
{
    setMinimumHeight(40);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

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

    int h = height();
    int w = width();
    p.drawLine(0, h - 1, w, h - 1);

    for (int i = 0; i < _ticks.size(); ++i) {
        int f   = _ticks[i];
        QString label = _labels[i];
        // normalize position
        float t = float(f - _frame_begin) / float(_frame_end - _frame_begin);
        int   x = int(t * w);
        // tick
        p.drawLine(x, h - 1, x, h - 7);
        // label
        p.drawText(x + 2, h - 9, label);
    }
    // optionally draw current-frame marker:
    if (_frame_current >= _frame_begin && _frame_current <= _frame_end) {
        float t = float(_frame_current - _frame_begin) / float(_frame_end - _frame_begin);
        int   x = int(t * w);
        QPen red(Qt::red);
        p.setPen(red);
        p.drawLine(x, 0, x, h);
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
