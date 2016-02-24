#ifndef QIMAGEROISELECTOR_H
#define QIMAGEROISELECTOR_H

#include <QWidget>
#include <QFileInfo>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>

namespace Ui {
class QImageROISelector;
}

class QImageROISelector : public QWidget
{
    Q_OBJECT

public:
    explicit QImageROISelector(QWidget *parent = 0);
    ~QImageROISelector();

    void setImage(QFileInfo & file);

private:
    Ui::QImageROISelector *ui;
     QGraphicsScene* imageScene;
     QGraphicsPixmapItem* imagePixmapItem;

};

#endif // QIMAGEROISELECTOR_H
