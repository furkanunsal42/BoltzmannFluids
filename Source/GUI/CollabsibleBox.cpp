#include "CollabsibleBox.h"

CollapsibleBox::CollapsibleBox(const QString &title, QWidget *parent)
    : QWidget(parent)
{
    headerButton = new QToolButton;
    headerButton->setStyleSheet("QToolButton { border: none; color: rgb(225, 226, 227); font-weight: bold; font-size:10pt; }");
    headerButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    headerButton->setArrowType(Qt::DownArrow);
    headerButton->setIconSize(QSize(11, 11));
    headerButton->setText("  " + title);
    headerButton->setCheckable(true);
    headerButton->setChecked(true);

    // Container
    contentArea = new QFrame;
    contentArea->setFrameShape(QFrame::NoFrame);
    contentArea->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed); // HERE
    contentAreaLayout = new QVBoxLayout;
    contentAreaLayout->setContentsMargins(0, 0, 0, 0);
    contentArea->setLayout(contentAreaLayout);

    // Main layout of the CollapsibleBox
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(0);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->addWidget(headerButton);
    mainLayout->addWidget(contentArea);
    mainLayout->setAlignment(Qt::AlignTop);

    connect(headerButton, &QToolButton::clicked, this, [=](bool checked){
        contentArea->setVisible(checked);
        headerButton->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);

        // For a smooth transition you can also implement an animation here.
    });
}

void CollapsibleBox::addWidget(QWidget *widget) {
    contentAreaLayout->addWidget(widget);
}
