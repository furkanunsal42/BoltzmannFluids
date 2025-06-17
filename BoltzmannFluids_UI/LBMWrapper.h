#ifndef LBMWRAPPER_H
#define LBMWRAPPER_H

#ifdef INCLUDE_LBM
#include "LBM.h"
#endif

#include <QObject>

class LBM;

class QT_LBMWrapper : public QObject {
    Q_OBJECT

public:

    explicit QT_LBMWrapper(QObject* parent = nullptr);

    void update_qt_wrapper(int new_frame);

    int get_frame_count() const;

    LBM& get_solver() const;
    void set_solver(LBM& new_solver);

    int max_frame() const;
    void setMax_frame(int newMax_frame);

signals:

    void current_frame_changed_signal(int frame);

private:

    LBM* solver = nullptr;
    int _frame = 0;
    int _max_frame = 10000;

};

#endif // LBMWRAPPER_H
