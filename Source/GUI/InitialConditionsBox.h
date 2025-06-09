#ifndef R_GHTSIDEPANEL_H
#define R_GHTSIDEPANEL_H

#include <QWidget>

#include <QVBoxLayout>
#include <QGroupBox>

class InitialConditionsBox : public QWidget
{
    Q_OBJECT
public:

    explicit InitialConditionsBox(QWidget* parent = nullptr);

private:

    QVBoxLayout* main_layout;

    QGroupBox* createInitialConditionsGroup();

};

#endif // R_GHTSIDEPANEL_H
