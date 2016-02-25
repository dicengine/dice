// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 Sandia Corporation.
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact: Dan Turner (dzturne@sandia.gov)
//
// ************************************************************************
// @HEADER

#include "qselectionarea.h"
#include <QPainter>
#include <QPaintEvent>
#include <iostream>
#include "DICe_InputVars.h"

QSelectionArea::QSelectionArea(QWidget *parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_StaticContents);
    myPenWidth = 3;
    scale_factor = 1.0;
    myPenColor = Qt::yellow;
    image_width = 0;
    image_height = 0;
    lastPoint = QPoint(0,0);
    parent_ = parent;
}

void QSelectionArea::resizeImage(QImage *image, const QSize &newSize)
{
    if (image->size() == newSize){
        return;
    }

    double scale_factor_x = (double)newSize.width() / (double)image->size().width();
    double scale_factor_y = (double)newSize.height() / (double)image->size().height();
    if(scale_factor_x < scale_factor_y)
        scale_factor = scale_factor_x;
    else
        scale_factor = scale_factor_y;
    std::cout << "scale factor: " << scale_factor << std::endl;

    QImage scaledImage = image->scaled(newSize.width(),newSize.height(),Qt::KeepAspectRatio);
    scaledImage = scaledImage.convertToFormat(QImage::Format_RGB32);
    //newImage.fill(qRgb(255, 255, 255));
    //QPainter painter(&scaledImage);
    //painter.drawImage(QPoint(0, 0), *image);
    *image = scaledImage;
    image_width = image->width();
    image_height = image->height();
}

void QSelectionArea::resetImage(){
    if(fileName_!=""){
        openImage(fileName_);
    }
}

bool QSelectionArea::openImage(const QString &fileName)
{
    fileName_ = fileName;
    QImage loadedImage;
    if (!loadedImage.load(fileName))
        return false;
    QSize newSize = size();
    resizeImage(&loadedImage, newSize);
    image = loadedImage;
    update();
    return true;
}

void QSelectionArea::resizeEvent(QResizeEvent *event)
{
    if(!image.isNull()){
        resizeImage(&image, QSize(width(), height()));
        update();
    }
    QWidget::resizeEvent(event);
}

void QSelectionArea::paintEvent(QPaintEvent *event)
{
    if(!image.isNull()){
        QPainter painter(this);
        QRect dirtyRect = event->rect();
        painter.drawImage(dirtyRect, image, dirtyRect);
    }
}

void QSelectionArea::decrementVertexSet(const bool excluded)
{
    // if a shape is in progress, just clear the currnet shape
    bool shape_in_progress = !is_first_point();

    // reset the image
    resetImage();

    // reset the points
    resetLocation();
    clear_current_roi_vertices();

    if(!shape_in_progress){
        // remove the last shape from the set
        if(excluded)
            DICe::gui::Input_Vars::instance()->decrement_excluded_vertex_vector();
        else
            DICe::gui::Input_Vars::instance()->decrement_vertex_vector();
    }

    // redraw the other shapes
    QColor color = Qt::yellow;
    for(QList<QList<QPoint> >::iterator it=DICe::gui::Input_Vars::instance()->get_roi_vertex_vectors()->begin();
        it!=DICe::gui::Input_Vars::instance()->get_roi_vertex_vectors()->end();++it){
        drawShape(*it,color);
    }
    color = Qt::red;
    for(QList<QList<QPoint> >::iterator it=DICe::gui::Input_Vars::instance()->get_roi_excluded_vertex_vectors()->begin();
        it!=DICe::gui::Input_Vars::instance()->get_roi_excluded_vertex_vectors()->end();++it){
        drawShape(*it,color);
    }
    //DICe::gui::Input_Vars::instance()->display_roi_vertices();
}

void QSelectionArea::drawShapeLine(QPoint & pt,bool excluded)
{
    // offset the point to account for the location of the image viewer
    offsetPoint(pt);

    // the first point doesn't need a line
    if(is_first_point()){
        std::cout << "first point!" << std::endl;
        originPoint = pt;
        lastPoint = pt;
        // add the vertex to the current shape
        current_roi_vertices.append(pt);
        return;
    }

    // if the current point is near the origin of the shape
    // close the polygon
    int tol = 10; // pixels
    bool closure = abs(pt.x() - originPoint.x()) + abs(pt.y() - originPoint.y()) < tol;

    if(closure){
        std::cout << " closure!" << std::endl;
        pt = originPoint;
    }

    // draw the line
    QColor color = Qt::yellow;
    if(excluded) color = Qt::red;
    drawLineTo(pt,color);

    if(closure){
        std::cout << "reset location! " << std::endl;
        resetLocation();
        if(excluded){
            // no need to append the last point to the current vertices because the begining is already in there
            // append the vertex set to the Input_Vars vector and clear the current shape
            DICe::gui::Input_Vars::instance()->append_excluded_vertex_vector(current_roi_vertices);
        }
        else{
            // no need to append the last point to the current vertices because the begining is already in there
            // append the vertex set to the Input_Vars vector and clear the current shape
            DICe::gui::Input_Vars::instance()->append_vertex_vector(current_roi_vertices);
            //std::cout << originPoint.x() << " " << originPoint.y() << " " << lastPoint.x() << " " << lastPoint.y() << std::endl;
            DICe::gui::Input_Vars::instance()->display_roi_vertices();
        }
        current_roi_vertices.clear();
    }
    else{
        // append the point to the current shape
        current_roi_vertices.append(pt);
        lastPoint = pt;
    }
}


void QSelectionArea::drawShape(QList<QPoint> & vertices, QColor & color){
    // iterate the vertices drawling lines between the points

    // append the begin point on the end of the list
    QList<QPoint> vertices_ap = vertices;
    vertices_ap.append(*vertices.begin());

    QPoint prev_point;
    for(QList<QPoint>::iterator it=vertices_ap.begin();it!=vertices_ap.end();++it){
        if(it==vertices_ap.begin()){
            prev_point = *it;
            continue;
        }
        QPainter painter(&image);
        painter.setPen(QPen(color, myPenWidth, Qt::SolidLine, Qt::RoundCap,
                            Qt::RoundJoin));
        painter.drawLine(prev_point, *it);
        prev_point = *it;
    }
    update();
}


void QSelectionArea::drawLineTo(const QPoint &endPoint, QColor & color)
{
    // take the offset away from endpoint
    //QPoint offset(endPoint.x() - parent_->x() - x(),endPoint.y() - parent_->y() - y());
    QPainter painter(&image);
    painter.setPen(QPen(color, myPenWidth, Qt::SolidLine, Qt::RoundCap,
                        Qt::RoundJoin));
    painter.drawLine(lastPoint, endPoint);
    update();
}

void QSelectionArea::offsetPoint(QPoint & pt)
{
    QPoint offset(pt.x() - parent_->x() - x(),pt.y() - parent_->y() - y());
    pt = offset;
}

void QSelectionArea::resetLocation()
{
    lastPoint = QPoint(0,0);
    originPoint = QPoint(0,0);
}
