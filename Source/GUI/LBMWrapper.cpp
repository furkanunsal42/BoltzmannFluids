#include "LBMWrapper.h"


QT_LBMWrapper::QT_LBMWrapper( QObject *parent)
    :QObject(parent)
{
}

void QT_LBMWrapper::update_qt_wrapper(int new_frame)
{
    _frame = new_frame;
}

int QT_LBMWrapper::get_frame_count() const
{
    return _frame;
}



LBM& QT_LBMWrapper::get_solver() const
{
    return *solver;
}

void QT_LBMWrapper::set_solver(LBM& new_solver)
{
    solver = &new_solver;
}

int QT_LBMWrapper::max_frame() const
{
    return _max_frame;
}

void QT_LBMWrapper::setMax_frame(int newMax_frame)
{
    _max_frame = newMax_frame;
}
