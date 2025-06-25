#include "Timeline.h"
#include "TimelineRuler.h"
#include "GUIApplication/Application.h"

#include <QPushButton>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QLabel>

Timeline::Timeline(QWidget *parent, int max_frame)
    :QWidget(parent), _frame_end(max_frame)
{

    auto layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    auto frame = new QFrame(this);
    frame->setFrameShape(QFrame::StyledPanel);
    frame->setFrameShadow(QFrame::Raised);
    frame->setObjectName("buttons_frame");
    frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    layout->addWidget(frame);

    // Layout for the frame
    auto frame_layout = new QHBoxLayout(frame);
    frame_layout->setContentsMargins(6, 3, 6, 3);
    frame_layout->setSpacing(0);

    // --- Row 0 ---
    // Centered Buttons
    auto center_buttons_widget = new QWidget(frame);
    auto center_buttons_layout = new QHBoxLayout(center_buttons_widget);
    center_buttons_layout->setContentsMargins(0, 0, 0, 0);
    center_buttons_layout->setSpacing(1);

    auto start_pause_button = new QPushButton(center_buttons_widget);
    auto start_icon = QIcon(":/qt_icons/white_start.png");
    auto pause_icon = QIcon(":/qt_icons/white_pause.png");
    start_pause_button->setIcon(start_icon);
    start_pause_button->setIconSize(QSize(20, 20));
    center_buttons_layout->addWidget(start_pause_button);

    auto stop_button = new QPushButton(center_buttons_widget);
    auto stop_icon = QIcon(":/qt_icons/white_stop.png");
    stop_button->setIcon(stop_icon);
    stop_button->setIconSize(QSize(20, 20));
    center_buttons_layout->addWidget(stop_button);

    QObject::connect(start_pause_button, &QPushButton::clicked, this,
                     [this, start_icon, pause_icon, start_pause_button]() {
                        Application& BoltzmannFluids = Application::get();
                        auto simulation = BoltzmannFluids.simulation;

                         _running = !_running;

                         if (_running) {
                             start_pause_button->setIcon(pause_icon);
                             start();
                             if (simulation != nullptr){
                                 simulation->is_paused = false;
                             }
                             emit start_signal();
                         } else {
                             start_pause_button->setIcon(start_icon);
                             pause();
                             if (simulation != nullptr){
                                 simulation->is_paused = true;
                             }
                             emit pause_signal();
                         }


                     });

    QObject::connect(stop_button, &QPushButton::clicked, this,
                     [this, stop_button, start_pause_button, start_icon]() {
                         stop();
                         emit stop_signal();
                         _running = false;
                         start_pause_button->setIcon(start_icon);
                     });


    // Frame count displayer
    auto frame_display_box = new QWidget(frame);
    frame_display_box->setObjectName("frame_display_box");
    frame_display_box->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    frame_display_box->setContentsMargins(0, 0, 0, 0);
    frame_display_box->setMaximumWidth(100);

    auto frame_display_layout = new QHBoxLayout(frame_display_box);
    frame_display_layout->setAlignment(Qt::AlignRight);
    frame_display_layout->setContentsMargins(2, 2, 3, 2);
    frame_display_layout->setSpacing(3);

    auto frame_display_label = new QLabel("Frame");
    frame_display_label->setMaximumWidth(45);
    frame_display_label->setAlignment(Qt::AlignRight);
    frame_display_label->setContentsMargins(0, 1, 0, 0);
    frame_display_layout->addWidget(frame_display_label);

    frame_display_text = new QLineEdit("0", frame_display_box);                 // TODO: change to QSpinBox maybe
    frame_display_text->setMaximumSize(42, 22);
    frame_display_layout->addWidget(frame_display_text);

    frame_layout->addSpacerItem(new QSpacerItem(90, 0)); // Invisible space item
    frame_layout->addStretch();
    frame_layout->addWidget(center_buttons_widget);
    frame_layout->addStretch();
    frame_layout->addWidget(frame_display_box);


    // --- Row:2 ---
    // Timeline ruler
    {
        this->_ruler = new TimelineRuler(this, this);
        layout->addWidget(this->_ruler);

    }


    setStyleSheet(
        "#buttons_frame {"
            "background-color: rgb(65, 66, 67);"
            "border: 1px solid rgb(75, 76, 77);"
            "border-radius: 7px;"
        "}"
        "QWidget {"
            "background-color: transparent; "
        "}"
        "QPushButton {"
            "background-color: rgb(91, 92, 93);"
            "border: 1px solid rgb(75, 76, 77);"
            "border-radius: 4px;"
        "}"
        "QPushButton::hover {"
            "background-color: rgb(131, 132, 133);"
            "border-radius: 4px;"
        "}"
        "#frame_display_box {"
            "background-color: rgb(91, 92, 93);"
            "border-radius: 2px;"
        "}"
        "QLabel {"
            "border: none;"
            "color: rgb(230, 230, 230);"
        "}"
        "QLineEdit {"
            "border: none;"
            "color: rgb(230, 230, 230);"
            "background-color: rgb(111, 112, 113);"
        "}"
        "QLineEdit:hover {"
            "border: none;"
            "color: rgb(230, 230, 230);"
            "background-color: rgb(131, 132, 133);"
        "}"
        );

}

void Timeline::start()
{
    if (!_running)
        _running = true;
}

void Timeline::pause()
{
    if (_running)
        _running = true;
}

void Timeline::stop()
{
    if (_running)
        _running = true;
}

int Timeline::get_frame_current() const
{
    return _frame_current;
}

void Timeline::set_frame_current(int frame)
{
    _frame_current = frame;

    if (frame_display_text)
        frame_display_text->setText(QString::number(frame));

    this->_ruler->update();
    //if (this->_ruler)
    //    this->_ruler->update_timeline(_frame_current);

    emit frame_changed(_frame_current);
}

int Timeline::get_frame_begin() const
{
    return _frame_begin;
}

void Timeline::set_frame_begin(int new_frame_begin)
{
    _frame_begin = new_frame_begin;
}

int Timeline::get_frame_max() const
{
    return _frame_end;
}

void Timeline::set_frame_max(int new_max_frame)
{
    _frame_end = new_max_frame;
}

int Timeline::get_frame_simulation_duration() const
{
    return _frame_simulation_duration;
}

void Timeline::set_frame_simulation_duration(int new_frame_simulation_duration)
{
    _frame_simulation_duration = new_frame_simulation_duration;
}

bool Timeline::is_running() const
{
    return _running;
}

void Timeline::set_running(bool value)
{
    _running = value;
}

void Timeline::set_frame_range(int begin, int end)
{
    _frame_begin = begin;
    _frame_end = end;
}




