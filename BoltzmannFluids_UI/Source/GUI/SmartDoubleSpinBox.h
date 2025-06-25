#ifndef SMARTDOUBLESPINBOX_H
#define SMARTDOUBLESPINBOX_H

#include <QDoubleSpinBox>

class SmartDoubleSpinBox : public QDoubleSpinBox
{
    Q_OBJECT
public:
    explicit SmartDoubleSpinBox(QWidget* parent = nullptr);

protected:

    QString textFromValue(double value) const override;

    QValidator::State validate(QString& input, int& pos) const override;

};


#endif // SMARTDOUBLESPINBOX_H
