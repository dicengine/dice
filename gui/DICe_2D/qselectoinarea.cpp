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

QSelectionArea::QSelectionArea(QWidget *parent)
    : QFrame(parent)
{
    setAttribute(Qt::WA_StaticContents);
    myPenWidth = 3;
    scaleFactor = 1.0;
    myPenColor = Qt::yellow;
    lastPoint = QPoint(0,0);
    addBoundaryEnabled = false;
    addExcludedEnabled = false;
    panX = 2;
    panY = 2;
    prevPanX = 2;
    prevPanY = 2;
    panInProgress = false;
    currentImageX = 0;
    currentImageY = 0;
    originalImageHeight = 0;
    originalImageWidth = 0;
    currentImageHeight = 0;
    currentImageWidth = 0;
    zoomInProgress = false;
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
    currentImageWidth = newSize.width();
    currentImageHeight = newSize.height();

    QImage scaledImage = image->scaled(currentImageWidth,currentImageHeight,Qt::KeepAspectRatio,Qt::SmoothTransformation);
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
    originalImageWidth = loadedImage.width();
    originalImageHeight = loadedImage.height();
    int standardImageWidth = size().width() - 4;
    int standardImageHeight = size().height() - 4;
    QSize newSize(standardImageWidth,standardImageHeight);
    // if the scale factor is not zero, scale according to the current scale factor
    if(zoomInProgress){
        QSize scaledSize(currentImageWidth,currentImageHeight);
        newSize = scaledSize;
    }
    resizeImage(&loadedImage, newSize);
    backgroundImage = loadedImage;
    update();
    return true;
}

void QSelectionArea::clearShapesSet()
{
    if(!activeImage()) return;
    resetOriginAndLastPt();
    clearCurrentShapeVertices();
    excludedShapes.clear();
    boundaryShapes.clear();
    resetImage();
}

void QSelectionArea::decrementShapeSet(const bool excluded, const bool refreshOnly)
{
    // return if there is no image
    if(!activeImage()) return;

    // if a shape is in progress, just clear the currnet shape
    bool shape_in_progress = shapeInProgress();

    // reset the points
    resetOriginAndLastPt();
    clearCurrentShapeVertices();

    if(!shape_in_progress&&!refreshOnly){
        // remove the last shape from the set
        if(excluded){
            if(excludedShapes.size()>0)
                excludedShapes.removeLast();
        }
        else{
            if(boundaryShapes.size()>0)
                boundaryShapes.removeLast();
        }
    }

    // redraw the other shapes
    //drawShapes();
    drawExistingShapes();
}

void QSelectionArea::updateVertices(QPoint & pt,const bool excluded, const bool forceClosure)
{
    pt = mapToImageCoords(pt);
    // the first point doesn't need a line
    if(!shapeInProgress()){
        originPoint = pt;
        lastPoint = pt;
        // add the vertex to the current shape
        currentShapeVertices.append(pt);
        return;
    }

    // if the current point is near the origin of the shape
    // close the polygon
    int tol = 3; // pixels
    bool closure = (abs(pt.x() - originPoint.x()) + abs(pt.y() - originPoint.y()) < tol) || forceClosure;

    if(closure){
        // append the vertex set to the shapes list
        if(excluded){
            excludedShapes.append(currentShapeVertices);
        }
        else{
            boundaryShapes.append(currentShapeVertices);
        }
        resetOriginAndLastPt();
        currentShapeVertices.clear();
    }
    else{
        // append the point to the current shape
        currentShapeVertices.append(pt);
        lastPoint = pt;
    }
}

void QSelectionArea::resetOriginAndLastPt()
{
    lastPoint = QPoint(0,0);
    originPoint = QPoint(0,0);
}

void QSelectionArea::drawExistingShapes()
{
    // clear and redraw the background image
    resetImage();

    if(boundaryShapes.size()==0&&excludedShapes.size()==0) return;

    QPainter painter(&backgroundImage);
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
    for(QList<QList<QPoint> >::iterator it=boundaryShapes.begin();
        it!=boundaryShapes.end();++it){
        QList<QPoint> vertices = *it;
        QPolygon poly;
        for(QList<QPoint>::iterator i=vertices.begin();i!=vertices.end();++i){
            QPoint mappedPt = mapToViewerCoords(*i);
            poly << mappedPt;
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
    for(QList<QList<QPoint> >::iterator it=excludedShapes.begin();
        it!=excludedShapes.end();++it){
        QList<QPoint> vertices = *it;
        QPolygon poly;
        for(QList<QPoint>::iterator i=vertices.begin();i!=vertices.end();++i){
            QPoint mappedPt = mapToViewerCoords(*i);
            poly << mappedPt;
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

QPoint QSelectionArea::mapToImageCoords(const QPoint &pt)
{
    // prevent div by zero
    if(scaleFactor==0.0) return pt;
    int mappedX = int((pt.x() - panX)/scaleFactor);
    int mappedY = int((pt.y() - panY)/scaleFactor);
    return QPoint(mappedX,mappedY);
}

QPoint QSelectionArea::mapToViewerCoords(const QPoint &pt)
{
    // no panning here because the panning is taken care of by the image painter
    int mappedX = int(pt.x()*scaleFactor);
    int mappedY = int(pt.y()*scaleFactor);
    return QPoint(mappedX,mappedY);
}

void QSelectionArea::drawPreviewPolygon(const QPoint & pt, const QColor & color){
    // a shape must be in progress
    if(!shapeInProgress()) return;

    QPoint pannedPt(pt.x()-panX,pt.y()-panY);

    QPainter painter(&backgroundImage);
    QPolygon poly;
    for(QList<QPoint>::iterator it=currentShapeVertices.begin();it!=currentShapeVertices.end();++it){
        poly << mapToViewerCoords(*it);
    }
    poly << pannedPt;

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

void QSelectionArea::panImage(const QPoint & pt)
{
    // determine the offset from the mouse pos when clicked
    panX = prevPanX + pt.x() - mousePressX;
    panY = prevPanY + pt.y() - mousePressY;
    drawExistingShapes();
}

void QSelectionArea::resetView()
{
    // return if there is no image
    if(!activeImage()) return;
    zoomInProgress = false;
    resetImagePan();
    scaleFactor = 1.0;
    drawExistingShapes();
}

//
// EVENTS
//

void QSelectionArea::resizeEvent(QResizeEvent *event)
{
    if(!backgroundImage.isNull()){
        resizeImage(&backgroundImage, QSize(width(), height()));
        update();
    }
    QWidget::resizeEvent(event);
}

void QSelectionArea::paintEvent(QPaintEvent *event)
{
    QFrame::paintEvent(event);
    if(!backgroundImage.isNull()){
        QPoint leftTop(event->rect().left(),event->rect().top());
        QSize size = backgroundImage.size();
        QRect imageRect(leftTop,size);

        QPainter painter(this);
        painter.translate(panX,panY);
        //QRect dirtyRect = event->rect();
        //painter.drawImage(dirtyRect, backgroundImage, dirtyRect);
        painter.drawImage(imageRect, backgroundImage, imageRect);
    }
}

void QSelectionArea::zoom(const bool out){
    zoomInProgress = true;

    double localScaleFactor = 1.41;
    if(out) localScaleFactor = 0.71;
    // takes care of wheel scaling and initial scaling to fit in viewer
    double totalScaleFactor = scaleFactor * localScaleFactor;

    bool heightIsLargerDim = backgroundImage.width() <= backgroundImage.height();
    int minW = originalImageWidth;
    int minH = originalImageHeight;
    int maxWH = 5000; // pixels

    QImage magImage;
    if (!magImage.load(imageFileName))
        return;

    int w = magImage.width();
    int h = magImage.height();
    int newW = (int)(w*totalScaleFactor);
    int newH = (int)(h*totalScaleFactor);
    if(heightIsLargerDim && newH < minH){
        return;
    }
    else if(newW < minW){
        return;
    }
    //if(newW < minWH || newH < minWH) return;
    if(newW > maxWH || newH > maxWH){
        return;
    }
    scaleFactor = totalScaleFactor;
    resizeImage(&magImage,QSize(newW,newH));
    backgroundImage = magImage;
    drawExistingShapes();
    update();
}


void QSelectionArea::wheelEvent(QWheelEvent *event)
{
    // return if there is no image
    if(!activeImage()) return;

    if(panInProgress) return;
    const bool out = event->delta()<0;
    zoom(out);
}

void QSelectionArea::mouseReleaseEvent(QMouseEvent *event)
{
    setCursor(Qt::ArrowCursor);
    if(event->button()==Qt::MidButton){
        panInProgress = false;
        prevPanX = panX;
        prevPanY = panY;
    }
}

void QSelectionArea::mousePressEvent(QMouseEvent *event)
{
    // return if there is no image
    if(!activeImage()) return;

    // store the location the mouse was pressed
    mousePressX = event->x();
    mousePressY = event->y();

    // detect pan request
    if(event->button()==Qt::MidButton){
        panInProgress = true;
        return;
    }

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

void QSelectionArea::mouseMoveEvent(QMouseEvent *event)
{
    // return if there is no image
    if(!activeImage())return;

    QPoint currentImageCoords = mapToImageCoords(event->pos());
    currentImageX = currentImageCoords.x();
    currentImageY = currentImageCoords.y();
    if(currentImageX >= 0 && currentImageX < originalImageWidth && currentImageY >=0 && currentImageY < originalImageHeight){
        emit mousePos();
    }

    // detect if the middle button is pressed (pan request)
    if(panInProgress){
        // change the pointer to a hand
        setCursor(Qt::ClosedHandCursor);
        panImage(event->pos());
        return;
    }

    // one of the draw buttons must be pressed
    if(addBoundaryEnabled||addExcludedEnabled){
        //drawShapes();
        drawExistingShapes();
        if(addBoundaryEnabled)
            drawPreviewPolygon(event->pos(),Qt::green);
        else
            drawPreviewPolygon(event->pos(),Qt::red);
    }
}
