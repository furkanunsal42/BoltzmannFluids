#ifndef TIMELINE_H
#define TIMELINE_H

#include <QWidget>
#include <QTimer>

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
    void finished();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    int _frame;
    int _max_frame;
    int start_frame = 0;

    QTimer _timer;
    int _frame_interval_ms = 33; // ~30 fps

    void _advance();
};

#endif // TIMELINE_H
