#-------------------------------------------------
#
# Project created by QtCreator 2016-02-24T08:11:48
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = DICe_2D
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
        simpleqtvtk.cpp

HEADERS  += mainwindow.h \
    qdebugstream.h \
    simpleqtvtk.h

FORMS    += mainwindow.ui \
         simpleqtvtk.ui

RESOURCES += \
    resources.qrc
