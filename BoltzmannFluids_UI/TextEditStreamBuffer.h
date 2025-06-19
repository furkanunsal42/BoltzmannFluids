#ifndef TEXTEDITSTREAMBUFFER_H
#define TEXTEDITSTREAMBUFFER_H

#include <streambuf>
#include <iostream>
#include <QTextEdit>

class TextEditStreamBuf : public std::streambuf {
public:
    TextEditStreamBuf(QTextEdit* widget) : text_edit(widget) {}

protected:
    int_type overflow(int_type v) override {
        if (v == '\n') {
            text_edit->append(QString::fromStdString(buffer));
            buffer.clear();
        } else {
            buffer += static_cast<char>(v);
        }
        return v;
    }

    std::streamsize xsputn(const char* p, std::streamsize n) override {
        for (std::streamsize i = 0; i < n; ++i) {
            overflow(p[i]);
        }
        return n;
    }

private:
    QTextEdit* text_edit;
    std::string buffer;
};


#endif // TEXTEDITSTREAMBUFFER_H
