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

#include <QFrame>
#include <QImage>

/// \class QSelectionArea
/// \brief This class enables the user to draw a collection of shapes on an image
/// to be used in creating ROIs, subsets, or regions. There are three basic types of
/// shapes. The boundary is the outer limit of the region. An exclusion is a portion of
/// the region to subtract from the region. An obstruction is used to block portions of
/// an image that may occlude a subset or ROI. Obstructions are usually only used with
/// subsets.
///
/// There are several helper methods that enable the user to zoom and pan the image
/// for the purpose of drawing the shapes.

class QSelectionArea : public QFrame
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
    void updateVertices(QPoint & pt, const bool excluded=false, const bool forceClosure=false);

    /// draw the final polygons with portions missing for excluded regions
    void drawExistingShapes();

    /// draw a polygon that connects the vertices in the current shape being drawn
    void drawPreviewPolygon(const QPoint & pt, const QColor & color);

    /// zoom in or out for the viewer
    void zoom(const bool out);

    /// map coordinates to the image coordinate system
    QPoint mapToImageCoords(const QPoint & pt);

    /// map coordinates to the current view coordinate system
    QPoint mapToViewerCoords(const QPoint & pt);

    /// clear the stored vertices for the current shape in progress
    void clearCurrentShapeVertices(){
        if(currentShapeVertices.size()>0)
            currentShapeVertices.clear();
    }

    /// reset the shape drawing if a shape is in progress (reresh)
    /// remove the previous shape from the saved set of shapes
    void decrementShapeSet(const bool excluded=false, const bool refreshOnly = false);

    /// clear the saved shapes
    void clearShapesSet();

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

    /// returns true if there is an active image
    bool activeImage(){
        return !backgroundImage.isNull();
    }

    /// returns the current image coordinates of the cursor
    int getCurrentImageX(){return currentImageX;}
    int getCurrentImageY(){return currentImageY;}

    /// pan the image
    void panImage(const QPoint & pt);

    /// reset the pan for the image
    void resetImagePan(){
        panX = 2;
        panY = 2;
        prevPanX = 2;
        prevPanY = 2;
        scaleFactor = 1.0;
    }

    /// returns the original width of the current image
    int getOriginalImageWidth(){return originalImageWidth;}

    /// returns the original height of the current image
    int getOriginalImageHeight(){return originalImageHeight;}

    /// reset the image view, but not the shapes
    void resetView();

    /// mouse release event
    void mouseReleaseEvent(QMouseEvent *event);

    /// mouse press event
    void mousePressEvent(QMouseEvent *event);

    /// mouse move event
    void mouseMoveEvent(QMouseEvent *event);

    /// mouse wheel event
    void wheelEvent(QWheelEvent *event);

protected:

    /// paint event
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;

    /// resize event
    // TODO prevent resize (static drawing window size, what if user changes size while drawing shape?)
    void resizeEvent(QResizeEvent *event) Q_DECL_OVERRIDE;

signals:
    void mousePos();

private:

    /// draw a line from the fromPoint to the endPoint
    //void drawLine(const QPoint &fromPoint, const QPoint &endPoint, QColor & color);

    /// resize the image to fit the window with proportional scaling
    void resizeImage(QImage *image, const QSize &newSize);

    /// original image size
    int originalImageWidth;
    int originalImageHeight;

    /// current image size
    int currentImageWidth;
    int currentImageHeight;

    /// the width of the pen
    int myPenWidth;

    /// the color of the shapes pen
    QColor myPenColor;

    /// the background image for the display
    QImage backgroundImage;

    /// stores the current scale of the image as displayed vs. size in pixels
    double scaleFactor;

    /// stores the last point that was set by a mousePress event
    QPoint lastPoint;

    /// stores the origin of the shape (first mousePress)
    QPoint originPoint;

    /// the current list of vertices for a shape being drawn
    QList<QPoint> currentShapeVertices;

    /// store the name of the image file in case it needs to be loaded later
    QString imageFileName;

    /// drawing boundary shapes is enabled
    bool addBoundaryEnabled;

    /// drawing excluded shapes is enabled
    bool addExcludedEnabled;

    /// panning offsets
    int panX;
    int panY;
    int prevPanX;
    int prevPanY;

    /// track the location that the mouse was pressed
    int mousePressX;
    int mousePressY;

    /// current image coordinates
    int currentImageX;
    int currentImageY;

    /// panning in progress flag
    bool panInProgress;

    /// Boundary
    QList<QList <QPoint> > boundaryShapes;

    /// Excluded
    QList<QList <QPoint> > excludedShapes;

    /// flag that the image has been rescaled by a wheel event
    bool zoomInProgress;

};


#endif // QSELECTIONAREA_H
