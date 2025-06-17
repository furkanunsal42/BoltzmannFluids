#ifndef TIMELINERULER_H
#define TIMELINERULER_H

#include <QWidget>

#include "UI_Config.h"

class Timeline;

class TimelineRuler : public QWidget {
    Q_OBJECT

public:

    explicit TimelineRuler(QWidget* parent = nullptr, Timeline* parent_timeline = nullptr);

    //void update_timeline(int current_frame);    // Call when you want ruler to be redrawn

    //void set_frame_range(int begin, int end);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:

    Timeline* parent_timeline;

    QString label;
};

#endif // TIMELINERULER_H
