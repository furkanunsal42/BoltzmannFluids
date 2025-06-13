#include "CollabsibleBox.h"

CollapsibleBox::CollapsibleBox(const QString &title, QWidget *parent)
    : QWidget(parent)
{
    this->setStyleSheet("background-color: rgb(60, 61, 62)");

    header_button = new QToolButton;
    header_button->setStyleSheet("QToolButton { border: none; color: rgb(245, 246, 247); font-weight: bold; font-size:10pt; }");
    header_button->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    header_button->setArrowType(Qt::DownArrow);
    header_button->setIconSize(QSize(11, 11));
    header_button->setText("  " + title);
    header_button->setCheckable(true);
    header_button->setChecked(true);
    header_button->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    // Container
    content_area = new QFrame;
    content_area->setFrameShape(QFrame::NoFrame);
    //content_area->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    content_area_layout = new QVBoxLayout;
    content_area_layout->setContentsMargins(0, 0, 0, 0);
    content_area->setLayout(content_area_layout);

    // Main layout of the CollapsibleBox
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(0);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->addWidget(header_button);
    mainLayout->addWidget(content_area);
    mainLayout->setAlignment(Qt::AlignTop);

    connect(header_button, &QToolButton::clicked, this, [=](bool checked){
        content_area->setVisible(checked);
        header_button->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);

        // For a smooth transition you can also implement an animation here.
    });
}

void CollapsibleBox::addWidget(QWidget *widget) {
    content_area_layout->addWidget(widget);
}
