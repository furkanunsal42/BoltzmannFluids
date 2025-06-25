#ifndef TIMELINE_H
#define TIMELINE_H

#include <QWidget>

class TimelineRuler;
class QLineEdit;

class Timeline : public QWidget
{
    Q_OBJECT

public:

    explicit Timeline(QWidget* parent = nullptr, int max_frame = 10000);

    void start();
    void pause();
    void stop();

    void set_frame_range(int begin, int end);

    bool is_running() const;
    void set_running(bool value);

    void set_frame_current(int frame);
    int get_frame_current() const;

    int get_frame_begin() const;
    void set_frame_begin(int new_frame_begin);

    int get_frame_max() const;
    void set_frame_max(int new_max_frame);

    int get_frame_simulation_duration() const;
    void set_frame_simulation_duration(int new_frame_simulation_duration);

signals:
    void frame_changed(int frame);
    void start_signal();
    void pause_signal();
    void stop_signal();
    //void finish_signal();

private:
    TimelineRuler* _ruler;  // Todo: make private
    QLineEdit* frame_display_text = nullptr;

    bool _running = false;

    int _frame_current  = 0;
    int _frame_begin    = 0;
    int _frame_end      = 10000;
    int _frame_simulation_duration = 0;

};

#endif // TIMELINE_H
