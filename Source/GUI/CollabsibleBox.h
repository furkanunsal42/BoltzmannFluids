#ifndef COLLABSIBLEBOX_H
#define COLLABSIBLEBOX_H

#include <QWidget>
#include <QToolButton>
#include <QFrame>
#include <QVBoxLayout>
#include <QScrollArea>

class CollapsibleBox : public QWidget {
    Q_OBJECT
public:

    CollapsibleBox(const QString &title = "", QWidget *parent = nullptr);

    void addWidget(QWidget *widget);

private:
    QToolButton* header_button;
    QFrame* content_area;
    QScrollArea* scroll_area;
    QVBoxLayout* content_area_layout;
};

#endif // COLLABSIBLEBOX_H
