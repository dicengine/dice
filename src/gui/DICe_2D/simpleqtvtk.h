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
#ifndef SIMPLEQTVTK_H
#define SIMPLEQTVTK_H

#include <QWidget>
#include <QFileInfo>
#include <QMessageBox>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkRenderer.h>
#include <vtkDoubleArray.h>
#include <vtkDelaunay2D.h>
#include <vtkPolyDataMapper.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkScalarBarActor.h>
#include <vtkImageActor.h>
#include <vtkImageData.h>
#include <vtkColorTransferFunction.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkCornerAnnotation.h>
#include <vtkCommand.h>
#include <vtkPropPicker.h>
#include <vtkInteractorStyle.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderWindow.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkAssemblyPath.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkPolygon.h>
#include <vtkProperty.h>
#include <vtkTriangleFilter.h>
#include <vtkLine.h>
#include <vtkImageMapToColors.h>
#include <vtkLookupTable.h>


namespace Ui {
class SimpleQtVTK;
}

/// \class PolygonMouseInteractorStyle
/// \brief Enables the user to draw regions ond display regions on the image
class PolygonMouseInteractorStyle : public vtkInteractorStyleImage
{
  public:
    /// constructor
    static PolygonMouseInteractorStyle* New();
    /// constructor
    PolygonMouseInteractorStyle();
    /// enable boundary region creation
    void setBoundaryEnabled(const bool flag);
    /// enable excluded region creation
    void setExcludedEnabled(const bool flag);
    /// set the extents of the background image
    void setImageExtents(const double & startX, const double & endX,
                         const double & startY, const double & endY, const double & spacing);
    /// returns 1 if the three points form a counterclockwise angle, -1 if clockwise, and 0 if colinear
    int clockwise(const double* a, const double *b, const double*c);
    /// returns true if the lines ap-aq and bp-bq intersect
    bool isIntersecting(const double* ap, const double* aq,
                        const double *bp, const double *bq);
    /// checks for a valid polygon
    // 1 is valid polygon, -1 is degenerate points, -2 is co-linear points, -3 is self-intersecting or colinear, -4 too few points
    int isValidPolygon();
    /// draws lines that define the existing shapes
    void drawExistingLines(const bool useExcluded);
    /// draws lines for a region in progress
    void drawLines(vtkSmartPointer<vtkPoints> ptSet);
    /// draws a polygon for a region in progress
    void drawPolygon(vtkSmartPointer<vtkPoints> ptSet);
    /// set the extents of the mask image
    void initializeMaskImage();
    /// remove all the polygons from storage
    void clearPolygons();
    /// reset any regions in progress and redraw existing ones
    void resetShapesInProgress();
    /// add all the region actors to the renderer
    void addActors();
    /// remove all the region actors from the renderer
    void removeActors();
    /// remove the last polygon from the stored set
    bool decrementPolygon(const bool excluded=false);
    /// set all the pixels inside active regions to a certain color
    void markPixelsInPolygon(const double & value, const bool useExcluded=false);
    /// draw all of the stored polygons
    void drawExistingPolygons();
    /// create a list of the polygon vertices
    void exportVertices(QList<QList<QPoint> > * boundary, QList<QList<QPoint> > * excluded);
    /// Mouse interactions
    virtual void OnRightButtonDown();
    virtual void OnMouseMove();
    virtual void OnLeftButtonDown();
private:
    /// vertices of polygons in progress or being drawn
    vtkSmartPointer<vtkPoints> currentPoints;
    /// a copy of the polygon vertices in progress with one vertex added for the mouse position
    vtkSmartPointer<vtkPoints> currentPointsP1;
    /// mapper to draw polygons in progress
    vtkSmartPointer<vtkPolyDataMapper> mapper;
    /// actor to show polygons in progress
    vtkSmartPointer<vtkActor> actor;
    /// mapper to draw lines of polygons in progress
    vtkSmartPointer<vtkPolyDataMapper> lineMapper;
    /// actor to display lines for polygons in progress
    vtkSmartPointer<vtkActor> lineActor;
    /// mapper to draw lines for excluded polygons
    vtkSmartPointer<vtkPolyDataMapper> excludedLineMapper;
    /// actor to display lines for excluded polygons
    vtkSmartPointer<vtkActor> excludedLineActor;
    /// mapper to draw lines for boundary polygons
    vtkSmartPointer<vtkPolyDataMapper> boundaryLineMapper;
    /// actor to display lines for boundary polygons
    vtkSmartPointer<vtkActor> boundaryLineActor;
    /// mapper to draw existing polygons
    vtkSmartPointer<vtkPolyDataMapper> existingMapper;
    /// actor to display existing polygons
    vtkSmartPointer<vtkImageActor> existingActor;
    /// used for converting mouse coordinates to image coords
    vtkSmartPointer<vtkCoordinate> coordinate;
    /// a mask that shows all the existing polygons
    vtkSmartPointer<vtkImageData> maskImage;
    /// lookup table for the polygon mask
    vtkSmartPointer<vtkLookupTable> maskLUT;
    /// converts the double values in the image to an alpha value
    vtkSmartPointer<vtkImageMapToColors> mapTransparency;
    /// storage for boundary vertices
    std::vector<vtkSmartPointer<vtkPoints> > boundaryPointsVector;
    /// storage for excluded vertices
    std::vector<vtkSmartPointer<vtkPoints> > excludedPointsVector;
    /// start image coord in X
    double imageStartX;
    /// start image coord in Y
    double imageStartY;
    /// end image coord in X
    double imageEndX;
    /// end image coord in Y
    double imageEndY;
    /// scale of the image vs. display coords
    double imageSpacing;
    /// determines if drawing boundary regions is enabled
    bool boundaryEnabled;
    /// determines if drawing excluded regions is enabled
    bool excludedEnabled;
};

// The mouse motion callback, to pick the image and recover pixel values
class vtkImageInteractionCallback : public vtkCommand
{
public:
    static vtkImageInteractionCallback *New(){
        return new vtkImageInteractionCallback;
    }
    vtkImageInteractionCallback(){
        //this->Viewer     = NULL;
        this->Image = NULL;
        this->Actor = NULL;
        this->Renderer = NULL;
        this->Interactor = NULL;
        this->Picker     = NULL;
        this->Annotation = NULL;
    }
    ~vtkImageInteractionCallback(){
        //this->Viewer     = NULL;
        this->Image = NULL;
        this->Actor = NULL;
        this->Renderer = NULL;
        this->Interactor = NULL;
        this->Picker     = NULL;
        this->Annotation = NULL;
    }
    void SetOriginSpacing(const double & originX, const double & originY, const double & spacingX, const double & spacingY){
        this->ox = originY;
        this->oy = originX;
        this->sx = spacingX;
        this->sy = spacingY;
    }
    void SetPointers(vtkPropPicker *picker,
                     vtkCornerAnnotation *annotation,
                     vtkRenderWindowInteractor *interactor,
                     vtkRenderer *renderer,
                     vtkImageActor *actor,
                     vtkImageData *image){

        this->Picker = picker;
        this->Annotation = annotation;
        this->Interactor = interactor;
        this->Renderer = renderer;
        this->Actor = actor;
        this->Image = image;
    }
    virtual void Execute(vtkObject *, unsigned long vtkNotUsed(event), void *){
        vtkInteractorStyle *style = vtkInteractorStyle::SafeDownCast(
                    Interactor->GetInteractorStyle());

        // Pick at the mouse location provided by the interactor
        this->Picker->Pick(Interactor->GetEventPosition()[0],
                Interactor->GetEventPosition()[1],
                0.0, Renderer);

        // There could be other props assigned to this picker, so
        // make sure we picked the image actor
        vtkAssemblyPath* path = this->Picker->GetPath();
        bool validPick = false;

        if (path)
        {
            vtkCollectionSimpleIterator sit;
            path->InitTraversal(sit);
            vtkAssemblyNode *node;
            for (int i = 0; i < path->GetNumberOfItems() && !validPick; ++i)
            {
                node = path->GetNextNode(sit);
                if (Actor == vtkImageActor::SafeDownCast(node->GetViewProp()))
                {
                    validPick = true;
                }
            }
        }

        if (!validPick)
        {
            this->Annotation->SetText(0, "(-,-)");
            Interactor->Render();
            // Pass the event further on
            style->OnMouseMove();
            return;
        }

        // Get the world coordinates of the pick
        double pos[3];
        this->Picker->GetPickPosition(pos);

        int image_coordinate[3];
        image_coordinate[0] = sx==0.0? 0.0 : vtkMath::Round(pos[0]/sx+ox);
        image_coordinate[1] = sy==0.0? 0.0 : vtkMath::Round(pos[1]/sy+oy);
        image_coordinate[2] = 0.0;

        std::string message = "(";
        message += vtkVariant(image_coordinate[0]).ToString();
        message += ",";
        message += vtkVariant(image_coordinate[1]).ToString();
        message += ")";
        this->Annotation->SetText( 0, message.c_str() );
        Interactor->Render();
        style->OnMouseMove();
    }

private:
    vtkImageActor* Actor;
    vtkRenderer* Renderer;
    vtkRenderWindowInteractor *Interactor;
    vtkPropPicker*        Picker;      // Pointer to the picker
    vtkCornerAnnotation*  Annotation;  // Pointer to the annotation
    vtkImageData* Image;
    double ox,oy;
    double sx,sy;
};


class SimpleQtVTK : public QWidget
{
    Q_OBJECT

public:
    /// constructor
    explicit SimpleQtVTK(QWidget *parent = 0);

    /// destructor
    ~SimpleQtVTK();

    /// read the results file
    void readResultsFile(const std::string & fileName);

    /// read a background image
    int readImageFile(const std::string & fileName);

    /// initialize the data structures
    void initializeClassMembers();

    /// create the polydata from the points
    void createPolyData();

    /// create a triangulation of the data points
    void triangulate();

    /// render a field on the triangulation
    /// ** this should only be called by the fieldsCombo change slot
    void renderField(const int index);

    /// returns a pointer to the vector of field names
    std::vector<std::string> * getFieldNames(){
        return & fieldNames;
    }

    /// set the file names for the viewer
    /// first one has no images defined
    void setFileNames(QStringList & resFiles){
        QStringList imgFiles;
        setFileNames(resFiles,imgFiles);
    }
    void setFileNames(QStringList & resFiles, QStringList & imgFiles);

    /// update the file that is displayed in the renderer
    void updateCurrentFile(const int fileIndex, const bool resetAlpha=false);

    /// reset the color scale
    void resetColorScale(const int & index);

    /// reset the camera
    void resetCamera();

    /// determine the alpha to use for triangulation
    void estimateTriAlpha();

    /// determine the indices of certain fields like coordinates
    void determinePrimaryFieldIndices();

    /// export the vertices from the geometry interface
    void exportVertices(QList<QList<QPoint> > * boundary, QList<QList<QPoint> > * excluded){
        style->exportVertices(boundary,excluded);
    }

    /// setup the viewer for the right mode of interaction
    void changeInteractionMode(const int mode);

private slots:
    void on_fieldsCombo_currentIndexChanged(int index);

    void on_opacitySpin_editingFinished();

    void on_showPointsBox_clicked();

    void on_resetViewButton_clicked();

    void on_showScaleBox_clicked();

    void on_fileCombo_currentIndexChanged(int index);

    void on_resetScaleButton_clicked();

    void on_showMeshBox_clicked();

    void on_printColorBox_clicked();

    void on_triAlphaSpin_editingFinished();

    void on_showAxesBox_clicked();

    void on_displaceMeshBox_clicked();

    void on_screenShotButton_clicked();

    void on_boundaryPlus_clicked();

    void on_excludedPlus_clicked();

    void on_hideCoordsBox_clicked();

    void on_boundaryMinus_clicked();

    void on_excludedMinus_clicked();

    void on_hideImageBox_clicked();

    void on_resetGeoButton_clicked();

    void on_tabWidget_tabBarClicked(int index);

private:
    Ui::SimpleQtVTK *ui;

    /// flag to mark if the class has been initialized
    bool isInitialized;
    /// the name of the results file
    std::string resultsFileName;
    /// all of the valid field names
    std::vector<std::string> fieldNames;
    /// a vector of results file names
    QStringList resultsFiles;
    /// a vector of image file names
    QStringList imageFiles;
    /// the number of data points in the output
    int numPoints;
    /// the field data
    std::vector<vtkSmartPointer<vtkDoubleArray> > fieldData;
    // mins and maxs
    std::vector<double> fieldMins;
    std::vector<double> fieldMaxs;
    /// polydata
    vtkSmartPointer<vtkPolyData> polyData;
    /// triangulation
    vtkSmartPointer<vtkDelaunay2D> delaunay;
    /// rendered
    vtkSmartPointer<vtkRenderer> renderer;
    /// color transfer function
    vtkSmartPointer<vtkColorTransferFunction> colorTransferFunction;
    /// mesh mapper
    vtkSmartPointer<vtkPolyDataMapper> meshMapper;
    /// mesh actor
    vtkSmartPointer<vtkActor> meshActor;
    /// plotter scalar bar
    vtkSmartPointer<vtkScalarBarActor> scalarBar;
    /// glyph filter
    vtkSmartPointer<vtkVertexGlyphFilter> glyphFilter;
    /// node mapper
    vtkSmartPointer<vtkPolyDataMapper> pointMapper;
    /// point actor
    vtkSmartPointer<vtkActor> pointActor;
    /// image actor
    vtkSmartPointer<vtkImageActor> imageActor;
    /// axes
    vtkSmartPointer<vtkAxesActor> axes;
    /// orientation
    vtkSmartPointer<vtkOrientationMarkerWidget> orientationWidget;
    // interactor style
    //vtkSmartPointer<vtkInteractorStyleImage> style;
    vtkSmartPointer<PolygonMouseInteractorStyle> style;

    /// field index of x coords
    int XIndex;
    /// field index of y coords
    int YIndex;
    /// field index of disp x
    int dispXIndex;
    /// field index of disp y
    int dispYIndex;

    /// original camera position info
    double imageOrigin[3];
    double imageSpacing[3];
    int imageExtent[6];

    /// window annotations
    vtkSmartPointer<vtkCornerAnnotation> cornerAnnotation;
    /// picker
    vtkSmartPointer<vtkPropPicker> propPicker;
    /// call back
    vtkSmartPointer<vtkImageInteractionCallback> callback;


};





// TODO: need to check for points that are coincident and deal with them
//       two points may lie in the same place, the triangulation will get rid of one
//       but there will still be input data for that point from the file

/// jet color map
/// public class Jet {
const static double jet[200][3] = {
    {0,0,5.200000e-01},
    {0,0,5.400000e-01},
    {0,0,5.600000e-01},
    {0,0,5.800000e-01},
    {0,0,6.000000e-01},
    {0,0,6.200000e-01},
    {0,0,6.400000e-01},
    {0,0,6.600000e-01},
    {0,0,6.800000e-01},
    {0,0,7.000000e-01},
    {0,0,7.200000e-01},
    {0,0,7.400000e-01},
    {0,0,7.600000e-01},
    {0,0,7.800000e-01},
    {0,0,8.000000e-01},
    {0,0,8.200000e-01},
    {0,0,8.400000e-01},
    {0,0,8.600000e-01},
    {0,0,8.800000e-01},
    {0,0,9.000000e-01},
    {0,0,9.200000e-01},
    {0,0,9.400000e-01},
    {0,0,9.600000e-01},
    {0,0,9.800000e-01},
    {0,0,1},
    {0,2.000000e-02,1},
    {0,4.000000e-02,1},
    {0,6.000000e-02,1},
    {0,8.000000e-02,1},
    {0,1.000000e-01,1},
    {0,1.200000e-01,1},
    {0,1.400000e-01,1},
    {0,1.600000e-01,1},
    {0,1.800000e-01,1},
    {0,2.000000e-01,1},
    {0,2.200000e-01,1},
    {0,2.400000e-01,1},
    {0,2.600000e-01,1},
    {0,2.800000e-01,1},
    {0,3.000000e-01,1},
    {0,3.200000e-01,1},
    {0,3.400000e-01,1},
    {0,3.600000e-01,1},
    {0,3.800000e-01,1},
    {0,4.000000e-01,1},
    {0,4.200000e-01,1},
    {0,4.400000e-01,1},
    {0,4.600000e-01,1},
    {0,4.800000e-01,1},
    {0,5.000000e-01,1},
    {0,5.200000e-01,1},
    {0,5.400000e-01,1},
    {0,5.600000e-01,1},
    {0,5.800000e-01,1},
    {0,6.000000e-01,1},
    {0,6.200000e-01,1},
    {0,6.400000e-01,1},
    {0,6.600000e-01,1},
    {0,6.800000e-01,1},
    {0,7.000000e-01,1},
    {0,7.200000e-01,1},
    {0,7.400000e-01,1},
    {0,7.600000e-01,1},
    {0,7.800000e-01,1},
    {0,8.000000e-01,1},
    {0,8.200000e-01,1},
    {0,8.400000e-01,1},
    {0,8.600000e-01,1},
    {0,8.800000e-01,1},
    {0,9.000000e-01,1},
    {0,9.200000e-01,1},
    {0,9.400000e-01,1},
    {0,9.600000e-01,1},
    {0,9.800000e-01,1},
    {0,1,1},
    {2.000000e-02,1,9.800000e-01},
    {4.000000e-02,1,9.600000e-01},
    {6.000000e-02,1,9.400000e-01},
    {8.000000e-02,1,9.200000e-01},
    {1.000000e-01,1,9.000000e-01},
    {1.200000e-01,1,8.800000e-01},
    {1.400000e-01,1,8.600000e-01},
    {1.600000e-01,1,8.400000e-01},
    {1.800000e-01,1,8.200000e-01},
    {2.000000e-01,1,8.000000e-01},
    {2.200000e-01,1,7.800000e-01},
    {2.400000e-01,1,7.600000e-01},
    {2.600000e-01,1,7.400000e-01},
    {2.800000e-01,1,7.200000e-01},
    {3.000000e-01,1,7.000000e-01},
    {3.200000e-01,1,6.800000e-01},
    {3.400000e-01,1,6.600000e-01},
    {3.600000e-01,1,6.400000e-01},
    {3.800000e-01,1,6.200000e-01},
    {4.000000e-01,1,6.000000e-01},
    {4.200000e-01,1,5.800000e-01},
    {4.400000e-01,1,5.600000e-01},
    {4.600000e-01,1,5.400000e-01},
    {4.800000e-01,1,5.200000e-01},
    {5.000000e-01,1,5.000000e-01},
    {5.200000e-01,1,4.800000e-01},
    {5.400000e-01,1,4.600000e-01},
    {5.600000e-01,1,4.400000e-01},
    {5.800000e-01,1,4.200000e-01},
    {6.000000e-01,1,4.000000e-01},
    {6.200000e-01,1,3.800000e-01},
    {6.400000e-01,1,3.600000e-01},
    {6.600000e-01,1,3.400000e-01},
    {6.800000e-01,1,3.200000e-01},
    {7.000000e-01,1,3.000000e-01},
    {7.200000e-01,1,2.800000e-01},
    {7.400000e-01,1,2.600000e-01},
    {7.600000e-01,1,2.400000e-01},
    {7.800000e-01,1,2.200000e-01},
    {8.000000e-01,1,2.000000e-01},
    {8.200000e-01,1,1.800000e-01},
    {8.400000e-01,1,1.600000e-01},
    {8.600000e-01,1,1.400000e-01},
    {8.800000e-01,1,1.200000e-01},
    {9.000000e-01,1,1.000000e-01},
    {9.200000e-01,1,8.000000e-02},
    {9.400000e-01,1,6.000000e-02},
    {9.600000e-01,1,4.000000e-02},
    {9.800000e-01,1,2.000000e-02},
    {1,1,0},
    {1,9.800000e-01,0},
    {1,9.600000e-01,0},
    {1,9.400000e-01,0},
    {1,9.200000e-01,0},
    {1,9.000000e-01,0},
    {1,8.800000e-01,0},
    {1,8.600000e-01,0},
    {1,8.400000e-01,0},
    {1,8.200000e-01,0},
    {1,8.000000e-01,0},
    {1,7.800000e-01,0},
    {1,7.600000e-01,0},
    {1,7.400000e-01,0},
    {1,7.200000e-01,0},
    {1,7.000000e-01,0},
    {1,6.800000e-01,0},
    {1,6.600000e-01,0},
    {1,6.400000e-01,0},
    {1,6.200000e-01,0},
    {1,6.000000e-01,0},
    {1,5.800000e-01,0},
    {1,5.600000e-01,0},
    {1,5.400000e-01,0},
    {1,5.200000e-01,0},
    {1,5.000000e-01,0},
    {1,4.800000e-01,0},
    {1,4.600000e-01,0},
    {1,4.400000e-01,0},
    {1,4.200000e-01,0},
    {1,4.000000e-01,0},
    {1,3.800000e-01,0},
    {1,3.600000e-01,0},
    {1,3.400000e-01,0},
    {1,3.200000e-01,0},
    {1,3.000000e-01,0},
    {1,2.800000e-01,0},
    {1,2.600000e-01,0},
    {1,2.400000e-01,0},
    {1,2.200000e-01,0},
    {1,2.000000e-01,0},
    {1,1.800000e-01,0},
    {1,1.600000e-01,0},
    {1,1.400000e-01,0},
    {1,1.200000e-01,0},
    {1,1.000000e-01,0},
    {1,8.000000e-02,0},
    {1,6.000000e-02,0},
    {1,4.000000e-02,0},
    {1,2.000000e-02,0},
    {1,0,0},
    {9.800000e-01,0,0},
    {9.600000e-01,0,0},
    {9.400000e-01,0,0},
    {9.200000e-01,0,0},
    {9.000000e-01,0,0},
    {8.800000e-01,0,0},
    {8.600000e-01,0,0},
    {8.400000e-01,0,0},
    {8.200000e-01,0,0},
    {8.000000e-01,0,0},
    {7.800000e-01,0,0},
    {7.600000e-01,0,0},
    {7.400000e-01,0,0},
    {7.200000e-01,0,0},
    {7.000000e-01,0,0},
    {6.800000e-01,0,0},
    {6.600000e-01,0,0},
    {6.400000e-01,0,0},
    {6.200000e-01,0,0},
    {6.000000e-01,0,0},
    {5.800000e-01,0,0},
    {5.600000e-01,0,0},
    {5.400000e-01,0,0},
    {5.200000e-01,0,0},
    {5.000000e-01,0,0}
};


#endif // SIMPLEQTVTK_H
