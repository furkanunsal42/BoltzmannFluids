#ifndef COLLABSIBLEBOX_H
#define COLLABSIBLEBOX_H

#include <QWidget>
#include <QToolButton>
#include <QFrame>
#include <QVBoxLayout>

class CollapsibleBox : public QWidget {
    Q_OBJECT
public:

    CollapsibleBox(const QString &title = "", QWidget *parent = nullptr);

    void addWidget(QWidget *widget);

private:
    QToolButton *headerButton;
    QFrame *contentArea;
    QVBoxLayout *contentAreaLayout;
};

#endif // COLLABSIBLEBOX_H
