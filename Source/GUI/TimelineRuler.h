#ifndef TIMELINERULER_H
#define TIMELINERULER_H

#include <QWidget>

#include "UI_Config.h"

class TimelineRuler : public QWidget {
    Q_OBJECT

public:

    explicit TimelineRuler(QWidget* parent = nullptr);

    void update_timeline(int current_frame);    // Call when you want ruler to be redrawn

    void set_frame_range(int begin, int end);

protected:
    void paintEvent(QPaintEvent* event) override;   // Draw function

    void resizeEvent(QResizeEvent* event) override; // To track width changes.

private:

    void _recompute_ruler();   // Recalculate positions & labels when begin, max or width changes

    int _frame_current  = 0;    // current frame count, cumulative from the simulation start.
    int _frame_begin    = 0;    // usually 0
    int _frame_end      = 10000;// limit value for simulation to stop

    QVector<QString> _labels;
    QVector<int> _ticks;

    QString label;
};

#endif // TIMELINERULER_H
