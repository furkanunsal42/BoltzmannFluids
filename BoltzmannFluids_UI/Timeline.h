#ifndef TIMELINE_H
#define TIMELINE_H

#include <QWidget>


class TimelineRuler;
class QLineEdit;

class Timeline : public QWidget
{
    Q_OBJECT

public:

    explicit Timeline(int max_frame = 10000, QWidget* parent = nullptr);

    void start();
    void pause();
    void stop();

    void set_frame(int frame);
    int get_current_frame() const;

    int get_frame_max() const;
    void set_frame_max(int new_max_frame);

signals:
    void frame_changed(int frame);
    void start_signal();
    void pause_signal();
    void stop_signal();
    //void finish_signal();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    TimelineRuler* _ruler;
    QLineEdit* frame_display_text = nullptr;

    int _frame      = 0;
    int start_frame = 0;
    int _frame_max  = 10000;
    bool _running = false;

};

#endif // TIMELINE_H
