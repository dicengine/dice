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

#ifndef QSELECTIONAREA_H
#define QSELECTIONAREA_H

#include <QWidget>
#include <QImage>

class QSelectionArea : public QWidget
{
    Q_OBJECT

public:
    QSelectionArea(QWidget *parent = 0);

    /// sets the image that the ROI's
    bool openImage(const QString &fileName);

    /// remove all the drawn shapes and re-render the background image
    void resetImage();

    /// sets the origin and last point back to zero
    void resetOriginAndLastPt();

    /// returns true if a shape is currently being drawn
    bool shapeInProgress()const{return lastPoint.x()!=0||lastPoint.y()!=0;}

    /// sets the color of the shapes pen
    void setPenColor(const QColor & color){
        myPenColor = color;
    }

    /// update the stored set of vertices
    void updateVertices(QPoint & pt, const bool excluded=false);

    /// draw a shape from a set of vertices
    void drawShape(QList<QPoint> & vertices,QColor & color);

    /// draw all the shapes stored and lines for ones in progress
    void drawShapes();

    /// clear the stored vertices for the current shape in progress
    void clearCurrentShapeVertices(){
        currentShapeVertices.clear();
    }

    /// reset the shape drawing if a shape is in progress (reresh)
    /// remove the previous shape from the saved set of shapes
    void decrementShapeSet(const bool excluded=false, const bool refreshOnly = false);

    /// sets the active de-active flag for drawing boundary shapes
    void setAddBoundaryEnabled(const bool & flag){
        addBoundaryEnabled = flag;
    }

    /// sets the active de-active flag for drawing excluded shapes
    void setAddExcludedEnabled(const bool & flag){
        addExcludedEnabled = flag;
    }

    /// returns true if drawing boundary shapes in enabled
    bool getAddBoundaryEnabled(){
        return addBoundaryEnabled;
    }

    /// returns true if drawing excluded shapes in enabled
    bool getAddExcludedEnabled(){
        return addExcludedEnabled;
    }

    /// mouse press event
    void mousePressEvent(QMouseEvent *event);

    /// mouse move event
    void mouseMoveEvent(QMouseEvent *event);

protected:

    /// paint event
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;

    /// resize event
    // TODO prevent resize (static drawing window size, what if user changes size while drawing shape?)
    void resizeEvent(QResizeEvent *event) Q_DECL_OVERRIDE;

private:

    /// draw a line from the fromPoint to the endPoint
    void drawLine(const QPoint &fromPoint, const QPoint &endPoint, QColor & color);

    /// resize the image to fit the window with proportional scaling
    void resizeImage(QImage *image, const QSize &newSize);

    /// The width of the pen
    int myPenWidth;

    /// The color of the shapes pen
    QColor myPenColor;

    /// the background image for the display
    QImage image;

    /// stores the current scale of the image as displayed vs. size in pixels
    double scaleFactor;

    /// Stores the last point that was set by a mousePress event
    QPoint lastPoint;

    /// Stores the origin of the shape (first mousePress)
    QPoint originPoint;

    /// The current list of vertices for a shape being drawn
    QList<QPoint> currentShapeVertices;

    /// Store the name of the image file in case it needs to be loaded later
    QString imageFileName;

    /// Drawing boundary shapes is enabled
    bool addBoundaryEnabled;

    /// Drawing excluded shapes is enabled
    bool addExcludedEnabled;
};


#endif // QSELECTIONAREA_H
