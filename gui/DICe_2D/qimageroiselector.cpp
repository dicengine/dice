#include "qimageroiselector.h"
#include "ui_qimageroiselector.h"

QImageROISelector::QImageROISelector(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::QImageROISelector)
{
    ui->setupUi(this);
    imageScene = new QGraphicsScene();
}

QImageROISelector::~QImageROISelector()
{
    //delete imageScene;
    //delete imagePixmapItem;
    delete ui;
}


void QImageROISelector::setImage(QFileInfo & file)
{
    ui->imageFileLabel->setText(file.filePath());
    QImage image(file.filePath());

    int w = ui->graphicsView->width()-10;
    int h = ui->graphicsView->height()-10;

    if(imagePixmapItem){
        delete imagePixmapItem;
    }
    imagePixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(image.scaled(w,h,Qt::KeepAspectRatio)));
    imageScene->clear();
    imageScene->addItem(imagePixmapItem);
    ui->graphicsView->setScene(imageScene);
    ui->graphicsView->show();

}
