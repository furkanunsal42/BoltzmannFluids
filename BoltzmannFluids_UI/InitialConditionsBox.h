#ifndef R_GHTSIDEPANEL_H
#define R_GHTSIDEPANEL_H

#include <QFrame>

class QGroupBox;

class InitialConditionsBox : public QFrame
{
    Q_OBJECT
public:

    explicit InitialConditionsBox(QWidget* parent = nullptr);

private:

    QGroupBox* createInitialConditionsGroup();

};

#endif // R_GHTSIDEPANEL_H
