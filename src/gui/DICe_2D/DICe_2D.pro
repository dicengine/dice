#-------------------------------------------------
#
# Project created by QtCreator 2016-02-24T08:11:48
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = DICe_2D
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    qimageroiselector.cpp \
    qselectionarea.cpp

HEADERS  += mainwindow.h \
    qimageroiselector.h \
    qselectionarea.h \
    qdebugstream.h

FORMS    += mainwindow.ui \
    qimageroiselector.ui

RESOURCES += \
    resources.qrc
