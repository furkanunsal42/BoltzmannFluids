#include "CollabsibleBox.h"

CollapsibleBox::CollapsibleBox(const QString &title, QWidget *parent)
    : QWidget(parent)
{
    // Main layout of the CollapsibleBox
    setAttribute(Qt::WA_StyledBackground, true); // allow styling
    auto main_layout = new QVBoxLayout(this);
    main_layout->setContentsMargins(6, 6, 6, 6);
    main_layout->setSpacing(0);
    main_layout->setAlignment(Qt::AlignTop);
    setLayout(main_layout);

    // Header Button
    header_button = new QToolButton(this);
    header_button->setStyleSheet("QToolButton { border: none; color: rgb(245, 246, 247); font-weight: bold; font-size:10pt; }"); //header_button->setStyleSheet("QToolButton { border: none; color: rgb(245, 246, 247); font-weight: bold; font-size:10pt; border-radius: 1px;");
    header_button->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    header_button->setArrowType(Qt::DownArrow);
    header_button->setIconSize(QSize(11, 11));
    header_button->setText("  " + title);
    header_button->setCheckable(true);
    header_button->setChecked(true);
    header_button->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    main_layout->addWidget(header_button);

    // Container
    content_area = new QFrame(this);
    content_area->setFrameShape(QFrame::NoFrame);
    content_area_layout = new QVBoxLayout(content_area);
    content_area_layout->setContentsMargins(4, 4, 4, 4);
    content_area->setLayout(content_area_layout);
    main_layout->addWidget(content_area);

    connect(header_button, &QToolButton::clicked, this, [=](bool checked){
        content_area->setVisible(checked);
        header_button->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);

        // For a smooth transition you can also implement an animation here.
    });
}

void CollapsibleBox::add_widget(QWidget *widget) {
    content_area_layout->addWidget(widget);
}
