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


namespace Ui {
class SimpleQtVTK;
}

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
    void readImageFile(const std::string & fileName);

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

    /// determine the alpha to use for triangulation
    void estimateTriAlpha();

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
