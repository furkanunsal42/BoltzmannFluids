#include "SmartDoubleSpinBox.h"

#include "UI_Config.h"

SmartDoubleSpinBox::SmartDoubleSpinBox(QWidget* parent)
    :QDoubleSpinBox(parent)
{
    this->setLocale(QLocale::C);
    setDecimals(DECIMAL_COUNT);
    setRange(SMARTDOUBLE_MIN, SMARTDOUBLE_MAX);
    setValue(0.00);
    setSingleStep(0.01);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    setMinimumWidth(50);
}

QString SmartDoubleSpinBox::textFromValue(double value) const
{
    QString text = QLocale::c().toString(value, 'f', DECIMAL_COUNT);

    static QRegularExpression remove_trailing_zeroes("0+$");
    static QRegularExpression remove_trailing_dot("\\.$");
    text.replace(remove_trailing_zeroes, "");
    text.replace(remove_trailing_dot, "");

    int dot_index = text.indexOf('.');
    if (dot_index == -1) {
        text += ".00";
    } else {
        int digits = text.length() - dot_index - 1;
        if (digits < 2) {
            text += QString(2 - digits, '0');
        }
    }

    return text;
}

QValidator::State SmartDoubleSpinBox::validate(QString& input, int& pos) const {

    if (input.contains(',')) {
        return QValidator::Invalid;
    }

    return QDoubleSpinBox::validate(input, pos);
}
