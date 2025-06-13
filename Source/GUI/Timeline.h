#ifndef TIMELINE_H
#define TIMELINE_H

#include <QWidget>

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

    int get_max_frame() const;
    void set_max_frame(int new_max_frame);

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
    int _frame;
    int _max_frame;
    int start_frame = 0;
    bool _running = false;

};

#endif // TIMELINE_H
