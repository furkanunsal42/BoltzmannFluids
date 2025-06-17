#ifndef COLLABSIBLEBOX_H
#define COLLABSIBLEBOX_H

#include <QWidget>
#include <QToolButton>
#include <QFrame>
#include <QVBoxLayout>
#include <QScrollArea>

class CollapsibleBox : public QWidget
{
    Q_OBJECT

public:

    explicit CollapsibleBox(const QString &title = "", QWidget *parent = nullptr);

    void add_widget(QWidget *widget);

private:
    QToolButton* header_button;
    QFrame* content_area;
    QScrollArea* scroll_area;
    QVBoxLayout* content_area_layout;
};

#endif // COLLABSIBLEBOX_H
