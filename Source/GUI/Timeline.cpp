#include "Timeline.h"

#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTextEdit>
#include <QLabel>

Timeline::Timeline(int max_frame, QWidget *parent)
    :_max_frame(max_frame), QWidget(parent)
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
//    frame_layout->setAlignment(Qt::AlignHCenter);

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

    auto stop_button = new QPushButton(center_buttons_widget);
    auto stop_icon = QIcon(":/qt_icons/white_stop.png");
    stop_button->setIcon(stop_icon);
    stop_button->setIconSize(QSize(20, 20));
    center_buttons_layout->addWidget(stop_button);


    // Frame count displayer
    auto frame_display_box = new QWidget(frame);
    frame_display_box->setObjectName("frame_display_box");
    frame_display_box->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    frame_display_box->setContentsMargins(0, 0, 0, 0);
    frame_display_box->setMaximumWidth(100);

    auto frame_display_layout = new QHBoxLayout(frame_display_box);
    frame_display_layout->setAlignment(Qt::AlignRight);
    frame_display_layout->setContentsMargins(3, 1, 0, 1);
    frame_display_layout->setSpacing(3);

    auto frame_display_label = new QLabel("Frame");
    frame_display_label->setMaximumWidth(40);
    frame_display_label->setAlignment(Qt::AlignRight);
    frame_display_label->setContentsMargins(0, 3, 0, 0);
    frame_display_layout->addWidget(frame_display_label);

    auto frame_display_text = new QTextEdit("0", frame_display_box);            // TODO: change to QSpinBox maybe
    frame_display_text->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    frame_display_text->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    frame_display_text->setMaximumSize(42, 22);
    frame_display_layout->addWidget(frame_display_text);


    frame_layout->addSpacerItem(new QSpacerItem(90, 0)); // Invisible space item
    frame_layout->addStretch();
    frame_layout->addWidget(center_buttons_widget);
    frame_layout->addStretch();
    frame_layout->addWidget(frame_display_box);


    // (--- Row:2 ---
    // Timeline ruler
    {
        auto ruler_area = new QWidget(this);
        layout->addWidget(ruler_area);

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
            "background-color: rgb(111, 112, 113);"
            "border-radius: 4px;"
        "}"
        "#frame_display_box {"
            "background-color: rgb(91, 92, 93);"
            "border-radius: 2px;"
        "}"
        "QTextEdit {"
            "background-color: rgb(91, 92, 93);"
            "border: 0px solid;"
            "border-radius: 4px;"
        "}"
        "QTextEdit::hover {"
        "background-color: rgb(111, 112, 113);"
        "border-radius: 4px;"
        "}"
        );

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


