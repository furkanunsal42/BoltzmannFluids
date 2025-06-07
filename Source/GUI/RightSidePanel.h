#ifndef R_GHTSIDEPANEL_H
#define R_GHTSIDEPANEL_H

#include <QWidget>

#include <QVBoxLayout>
#include <QGroupBox>

class RightSidePanel : public QWidget
{
    Q_OBJECT
public:

    explicit RightSidePanel(QWidget* parent = nullptr);

private:

    QVBoxLayout* mainLayout;

    QGroupBox* createInitialConditionsGroup();

};

#endif // R_GHTSIDEPANEL_H
