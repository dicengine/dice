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
    scaleFactor = 1.0;
    myPenColor = Qt::yellow;
    lastPoint = QPoint(0,0);
}

void QSelectionArea::resizeImage(QImage *image, const QSize &newSize)
{
    if (image->size() == newSize){
        return;
    }
    double scaleFactorX = (double)newSize.width() / (double)image->size().width();
    double scaleFactorY = (double)newSize.height() / (double)image->size().height();
    if(scaleFactorX < scaleFactorY)
        scaleFactor = scaleFactorX;
    else
        scaleFactor = scaleFactorY;
    //std::cout << "scale factor: " << scaleFactor << std::endl;

    QImage scaledImage = image->scaled(newSize.width(),newSize.height(),Qt::KeepAspectRatio);
    scaledImage = scaledImage.convertToFormat(QImage::Format_RGB32);
    *image = scaledImage;
}

void QSelectionArea::resetImage(){
    if(imageFileName!=""){
        openImage(imageFileName);
    }
}

bool QSelectionArea::openImage(const QString & fileName)
{
    imageFileName = fileName;
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

void QSelectionArea::drawShapes()
{
    // clear and redraw the background image
    resetImage();

    // redraw the boundary shapes
    QColor color = Qt::yellow;
    for(QList<QList<QPoint> >::iterator it=DICe::gui::Input_Vars::instance()->get_roi_vertex_vectors()->begin();
        it!=DICe::gui::Input_Vars::instance()->get_roi_vertex_vectors()->end();++it){
        drawShape(*it,color);
    }
    // redraw the excluded shapes
    color = Qt::red;
    for(QList<QList<QPoint> >::iterator it=DICe::gui::Input_Vars::instance()->get_roi_excluded_vertex_vectors()->begin();
        it!=DICe::gui::Input_Vars::instance()->get_roi_excluded_vertex_vectors()->end();++it){
        drawShape(*it,color);
    }

    // draw lines for any segments for shapes in progress
    //if(currentShapeVertices.size()>1){
    //    QPoint prevPoint = *currentShapeVertices.begin();
    //    for(QList<QPoint>::iterator it=currentShapeVertices.begin();it!=currentShapeVertices.end();++it){
    //        if(it==currentShapeVertices.begin()) continue;
    //        drawLine(prevPoint,*it,myPenColor);
    //        prevPoint = *it;
    //    }
    //}
}

void QSelectionArea::decrementShapeSet(const bool excluded, const bool refreshOnly)
{
    // if a shape is in progress, just clear the currnet shape
    bool shape_in_progress = shapeInProgress();

    // reset the points
    resetOriginAndLastPt();
    clearCurrentShapeVertices();

    if(!shape_in_progress&&!refreshOnly){
        // remove the last shape from the set
        if(excluded)
            DICe::gui::Input_Vars::instance()->decrement_excluded_vertex_vector();
        else
            DICe::gui::Input_Vars::instance()->decrement_vertex_vector();
    }

    // redraw the other shapes
    //drawShapes();
    drawFinalShapes();
}

void QSelectionArea::updateVertices(QPoint & pt,const bool excluded, const bool forceClosure)
{
    // the first point doesn't need a line
    if(!shapeInProgress()){
        std::cout << "first point!" << std::endl;
        originPoint = pt;
        lastPoint = pt;
        // add the vertex to the current shape
        currentShapeVertices.append(pt);
        return;
    }

    // if the current point is near the origin of the shape
    // close the polygon
    int tol = 10; // pixels
    bool closure = (abs(pt.x() - originPoint.x()) + abs(pt.y() - originPoint.y()) < tol) || forceClosure;
    if(closure){
        pt = originPoint;
    }

    if(closure){
        // append the vertex set to the Input_Vars vector before adding the origin so that
        // the origin is not duplicated in the set
        if(excluded){
            DICe::gui::Input_Vars::instance()->append_excluded_vertex_vector(currentShapeVertices);
        }
        else{
            DICe::gui::Input_Vars::instance()->append_vertex_vector(currentShapeVertices);
        }
        // add the last point now for drawing purposes
        // TODO comment this out later
        lastPoint = pt;
        //drawShapes();
        resetOriginAndLastPt();
        currentShapeVertices.clear();
    }
    else{
        // append the point to the current shape
        currentShapeVertices.append(pt);
        lastPoint = pt;
        //drawShapes();
    }
}


void QSelectionArea::drawShape(QList<QPoint> & vertices, QColor & color){
    // append the begin point on the end of the list
    QList<QPoint> vertices_ap = vertices;
    vertices_ap.append(*vertices.begin());

    // iterate the vertices drawling lines between the points
    QPoint prevPoint;
    for(QList<QPoint>::iterator it=vertices_ap.begin();it!=vertices_ap.end();++it){
        if(it==vertices_ap.begin()){
            prevPoint = *it;
            continue;
        }
        drawLine(prevPoint,*it,color);
        prevPoint = *it;
    }
    update();
}


void QSelectionArea::drawLine(const QPoint &fromPoint, const QPoint &endPoint, QColor & color)
{
    QPainter painter(&image);
    painter.setPen(QPen(color, myPenWidth, Qt::SolidLine, Qt::RoundCap,
                        Qt::RoundJoin));
    painter.drawLine(fromPoint, endPoint);
    update();
}

void QSelectionArea::resetOriginAndLastPt()
{
    lastPoint = QPoint(0,0);
    originPoint = QPoint(0,0);
}


void QSelectionArea::mousePressEvent(QMouseEvent *event)
{
    // if the user presses the right button and there
    // are already at least three points in the shape
    // close the shape
    bool forceClosure = currentShapeVertices.size() > 2 && event->button() == Qt::RightButton;

    // do nothing for other-wise right clicks
    if(!forceClosure && event->button() == Qt::RightButton) return;

    // check the boundary plus button is pressed
    if(addBoundaryEnabled){
        // draw the points:
        QColor color = Qt::yellow;
        setPenColor(color);
        QPoint pt(event->x(),event->y());
        updateVertices(pt,false,forceClosure);
    }
    // check the excluded plus button is pressed
    else if(addExcludedEnabled){
        // draw the points:
        QColor color = Qt::red;
        setPenColor(color);
        QPoint pt(event->x(),event->y());
        updateVertices(pt,true,forceClosure);
    }
}

void QSelectionArea::drawFinalShapes()
{
    // clear and redraw the background image
    resetImage();

    QPainter painter(&image);
    // Brush
    QBrush brush;
    brush.setColor(Qt::green);
    brush.setStyle(Qt::SolidPattern);
    // Fill polygon
    QPainterPath masterPath;
    masterPath.setFillRule(Qt::WindingFill);
    painter.setOpacity(0.2);
    // pen for boundary outlines
    QPen boundaryPen(Qt::green, 3, Qt::DashLine, Qt::RoundCap, Qt::RoundJoin);
    // pen for excluded outlines
    QPen excludedPen(Qt::red, 3, Qt::DashLine, Qt::RoundCap, Qt::RoundJoin);

    // redraw the boundary shapes
    for(QList<QList<QPoint> >::iterator it=DICe::gui::Input_Vars::instance()->get_roi_vertex_vectors()->begin();
        it!=DICe::gui::Input_Vars::instance()->get_roi_vertex_vectors()->end();++it){
        QList<QPoint> vertices = *it;
        QPolygon poly;
        for(QList<QPoint>::iterator i=vertices.begin();i!=vertices.end();++i){
            poly << *i;
        }
        QPainterPath path;
        path.setFillRule(Qt::WindingFill);
        path.addPolygon(poly);
        masterPath += path;
        // draw an outline of the boundary shapes
        painter.setPen(boundaryPen);
        painter.drawPolygon(poly);
    }

    // redraw the excluded shapes
    for(QList<QList<QPoint> >::iterator it=DICe::gui::Input_Vars::instance()->get_roi_excluded_vertex_vectors()->begin();
        it!=DICe::gui::Input_Vars::instance()->get_roi_excluded_vertex_vectors()->end();++it){
        QList<QPoint> vertices = *it;
        QPolygon poly;
        for(QList<QPoint>::iterator i=vertices.begin();i!=vertices.end();++i){
            poly << *i;
        }
        QPainterPath path;
        path.setFillRule(Qt::WindingFill);
        path.addPolygon(poly);
        masterPath -= path;
        // draw the outline of the excluded shapes
        painter.setPen(excludedPen);
        painter.drawPolygon(poly);
    }

    // Draw a filled polygon representing the active ROI area
    painter.fillPath(masterPath, brush);
}

void QSelectionArea::drawPreviewPolygon(const QPoint & pt, const QColor & color){
    // a shape must be in progress
    if(!shapeInProgress()) return;

    QPainter painter(&image);
    QPolygon poly;
    for(QList<QPoint>::iterator it=currentShapeVertices.begin();it!=currentShapeVertices.end();++it){
        poly << *it;
    }
    poly << pt;

    // pen
    QPen pen(color, 3, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
    painter.setPen(pen);
    // opacity
    painter.setOpacity(0.2);

    // Brush
    QBrush brush;
    brush.setColor(color);
    brush.setStyle(Qt::SolidPattern);

    // Fill polygon
    QPainterPath path;
    path.addPolygon(poly);

    // Draw polygon
    painter.drawPolygon(poly);
    painter.fillPath(path, brush);
}

void QSelectionArea::mouseMoveEvent(QMouseEvent *event)
{
    // one of the draw buttons must be pressed
    if(addBoundaryEnabled||addExcludedEnabled){
        //drawShapes();
        drawFinalShapes();
        if(addBoundaryEnabled)
            drawPreviewPolygon(event->pos(),Qt::green);
        else
            drawPreviewPolygon(event->pos(),Qt::red);
    }
}


