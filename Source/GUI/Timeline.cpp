#include "Timeline.h"

#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>

#include <QLabel>

Timeline::Timeline(int max_frame, QWidget *parent)
    :_max_frame(max_frame), QWidget(parent)
{

    auto layout = new QGridLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    auto frame = new QFrame(this);
    frame->setFrameShape(QFrame::StyledPanel);
    frame->setFrameShadow(QFrame::Raised);
    frame->setObjectName("buttons_frame");

    // Layout for the frame
    auto frame_layout = new QHBoxLayout(frame);
    frame_layout->setContentsMargins(1, 0, 1, 0);
    frame_layout->setSpacing(0);
    frame_layout->addStretch();
    frame_layout->setAlignment(Qt::AlignHCenter);

    // Buttons
    auto start_pause_button = new QPushButton(frame);
    auto start_icon = QIcon(":/qt_icons/white_start.png");
    auto pause_icon = QIcon(":/qt_icons/white_pause.png");
    start_pause_button->setIcon(start_icon);
    start_pause_button->setIconSize(QSize(20, 20));
    frame_layout->addWidget(start_pause_button);

    QObject::connect(start_pause_button, &QPushButton::clicked, this, [this, start_icon, start_pause_button, pause_icon]() {
        _running = !_running;

        if (!_running) {
            start_pause_button->setIcon(start_icon);
            emit start_signal();
        }
        else {
            start_pause_button->setIcon(pause_icon);
            emit pause_signal();
        }
    });

    auto stop_button = new QPushButton(frame);
    auto stop_icon = QIcon(":/qt_icons/white_stop.png");
    stop_button->setIcon(stop_icon);
    stop_button->setIconSize(QSize(20, 20));
    frame_layout->addWidget(stop_button);
    frame_layout->addStretch();

    //layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum), 0, 0);
    layout->addWidget(frame, 0, 0);
    //layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum), 0, 2);


    setStyleSheet(
        "#buttons_frame {"
            "background-color: rgb(65, 66, 67);"
            "border: 1px solid rgb(75, 76, 77);"
            "border-radius: 10px;"
        "}"
        "QPushButton {"
            "background-color: rgb(101, 102, 103);"
            "border-radius: 2px;"
        "}"
        "QPushButton::hover {"
            "background-color: rgb(121, 122, 123);"
            "border-radius: 2px;"
        "}"
        );


    // Timeline area
    {
        auto label = new QLabel("delete");
        layout->addWidget(label);

    }

}

void Timeline::start()
{

}

void Timeline::pause()
{

}

void Timeline::stop()
{

}

void Timeline::paintEvent(QPaintEvent *event)
{

}

void Timeline::mousePressEvent(QMouseEvent *event)
{

}


