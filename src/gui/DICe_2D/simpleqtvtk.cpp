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

#include "simpleqtvtk.h"
#include "ui_simpleqtvtk.h"
#include <sstream>
#include <algorithm>
#include <string.h>
#include <stdio.h>
#include <QFileDialog>
#include <QFileInfo>

#include <vtkPointData.h>
#include <vtkRenderWindow.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkTextProperty.h>
#include <vtkTIFFReader.h>
#include <vtkImageData.h>
#include <vtkColorSeries.h>
#include <vtkInteractorStyleImage.h>
#include <vtkCaptionActor2D.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

SimpleQtVTK::SimpleQtVTK(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SimpleQtVTK)
{
    ui->setupUi(this);

    // set the defaults for the user controls
    ui->opacitySpin->setMinimum(0.0);
    ui->opacitySpin->setMaximum(1.0);
    ui->opacitySpin->setValue(0.7);
    ui->opacitySpin->setSingleStep(0.1);
    ui->showScaleBox->setChecked(true);
    ui->showAxesBox->setChecked(true);
    ui->displaceMeshBox->setChecked(true);

    isInitialized = false;

    // set the auto scale to true
    ui->autoScaleBox->setChecked(true);

    // initialize the class members
    initializeClassMembers();    
}

SimpleQtVTK::~SimpleQtVTK()
{
    delete ui;
}

void SimpleQtVTK::initializeClassMembers(){
    /// geo and field data
    polyData = vtkSmartPointer<vtkPolyData>::New();
    // triangulation
    delaunay = vtkSmartPointer<vtkDelaunay2D>::New();
    //delaunay->SetAlpha(30.0);
    // color transfer function for plotting
    colorTransferFunction = vtkSmartPointer<vtkColorTransferFunction>::New();
    // mesh mapper
    meshMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    // mesh actor
    meshActor = vtkSmartPointer<vtkActor>::New();
    meshActor->GetProperty()->SetEdgeColor(1.0,1.0,1.0); //(R,G,B)
    meshActor->GetProperty()->EdgeVisibilityOff();
    // scalar bar
    scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar->SetNumberOfLabels(5);
    scalarBar->SetTextPad(10);
    //scalarBar->SetOrientationToVertical();
    scalarBar->SetHeight(0.25);
    scalarBar->SetWidth(0.15);
    scalarBar->SetBarRatio(0.1);
    // glyph filter
    glyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    // node mapper
    pointMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    // point actor
    pointActor = vtkSmartPointer<vtkActor>::New();
    pointActor->GetProperty()->SetColor(1,0,0);
    pointActor->GetProperty()->SetPointSize(3);
    // renderer
    renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->GradientBackgroundOn();
    renderer->SetBackground(0.4,0.4,0.4); // Background color
    renderer->SetBackground2(0.1,0.1,0.1);

    // image actor
    imageActor = vtkSmartPointer<vtkImageActor>::New();

    // set the widget render window to the renderer
    ui->vtkWidget->GetRenderWindow()->AddRenderer(renderer);

    // automatically update alpha for the triangulation
    ui->autoTriAlphaBox->setChecked(true);

    // set the style of the interaction
    vtkSmartPointer<vtkInteractorStyleImage> style =
          vtkSmartPointer<vtkInteractorStyleImage>::New();
    ui->vtkWidget->GetInteractor()->SetInteractorStyle(style);

    axes = vtkSmartPointer<vtkAxesActor>::New();
    orientationWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    orientationWidget->SetOutlineColor( 0.9300, 0.5700, 0.1300 );
    orientationWidget->SetOrientationMarker( axes );
    orientationWidget->SetInteractor( ui->vtkWidget->GetInteractor() );
    //orientationWidget->SetViewport( 0.0, 0.0, 0.4, 0.4 );
    orientationWidget->SetEnabled( 1 );
    orientationWidget->InteractiveOn();

}

void SimpleQtVTK::updateCurrentFile(const int fileIndex, const bool resetAlpha){
    isInitialized = false;

    QFileInfo result = resultsFiles[fileIndex];
    readResultsFile(result.fileName().toStdString());

    determinePrimaryFieldIndices();

    // estimate alpha:
    if(resetAlpha)
        estimateTriAlpha();

    // create the poly data
    createPolyData();

    // create a triangulation of the data
    triangulate();

    isInitialized = true;

    if(ui->fieldsCombo->currentIndex()!=0)
        // force a render with the current index
        renderField(ui->fieldsCombo->currentIndex());
    // start with the first non-coordinate field (index 2, 0 is x 1 is y)
    else
        ui->fieldsCombo->setCurrentIndex(2);

    // put an image on the viewer:
    if(imageFiles.size()>0){
        QFileInfo image = imageFiles[fileIndex];
        readImageFile(image.fileName().toStdString());
    }
    renderer->ResetCamera();
    renderer->SetActiveCamera(NULL);
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();

}

void SimpleQtVTK::setFileNames(QStringList & resFiles, QStringList & imgFiles){
    if(imgFiles.size()!=0 && imgFiles.size()!=resFiles.size()){
        std::cout << "ERROR: results file list and image file list are not the same size but must be." << std::endl;
    }
    resultsFiles.clear();
    imageFiles.clear();
    resultsFiles = resFiles;
    imageFiles = imgFiles;

    // add the names to the combo box
    ui->fileCombo->clear();
    for(QStringList::iterator it=resultsFiles.begin(); it!=resultsFiles.end();++it){
        QFileInfo currentFile = *it;
        QString shortFileName = currentFile.fileName();
        ui->fileCombo->addItem(shortFileName);
    }
}


void SimpleQtVTK::determinePrimaryFieldIndices(){
    XIndex = -1;
    YIndex = -1;
    dispXIndex = -1;
    dispYIndex = -1;

    // iterate the fields to find coords x
    for(int i=0;i<fieldNames.size();++i){
        if(strncmp(fieldData[i]->GetName(),"COORDINATE_X",12)==0)
            XIndex = i;
    }
    for(int i=0;i<fieldNames.size();++i){
        if(strncmp(fieldData[i]->GetName(),"COORDINATE_Y",12)==0)
            YIndex = i;
    }
    for(int i=0;i<fieldNames.size();++i){
        if(strncmp(fieldData[i]->GetName(),"DISPLACEMENT_X",14)==0)
            dispXIndex = i;
    }
    for(int i=0;i<fieldNames.size();++i){
        if(strncmp(fieldData[i]->GetName(),"DISPLACEMENT_Y",14)==0)
            dispYIndex = i;
    }
    if(XIndex==-1||YIndex==-1||dispXIndex==-1||dispYIndex==-1)
    {
        std::cout << "ERROR: could not find COORDINATES_X, COORDINATES_Y, DISPLACEMENT_X, or DISPLACEMENT_Y field" << std::endl;
        std::cout << "Coords X field index: " << XIndex << std::endl;
        std::cout << "Coords Y field index: " << YIndex << std::endl;
        std::cout << "Disp X field index: " << dispXIndex << std::endl;
        std::cout << "Disp Y field index: " << dispYIndex << std::endl;
    }
}


void SimpleQtVTK::readImageFile(const std::string & fileName){
    //std::cout << "reading image " << fileName << std::endl;
    vtkSmartPointer<vtkImageData> imageData;
    vtkSmartPointer<vtkTIFFReader> tifReader =
            vtkSmartPointer<vtkTIFFReader>::New();

    if( !tifReader->CanReadFile( fileName.c_str() ) )
    {
        std::cerr << "Error reading file " << fileName << std::endl;
    }
    tifReader->SetFileName ( fileName.c_str() );
    tifReader->Update();
    imageData = tifReader->GetOutput();
    // Create an image actor to display the image
    imageActor->SetInputData(imageData);
    renderer->AddActor2D(imageActor);
}

void SimpleQtVTK::estimateTriAlpha(){
    // assert that the first couple fields are X Y COORDS ...
    if(fieldData.size() < 2)
        std::cout << "ERROR: fieldData.size() < 2, not enough fields to create poly data" << std::endl;

    // TODO there's probably a better way to figure out alpha than computing the average distance...
    double dist_x = 0.0;
    double dist_y = 0.0;
    double avg_dist = 0.0;
    //double min_dist = std::numeric_limits<double>::max();
    //double max_dist = 0.0;
    double x = 0.0;
    double y = 0.0;
    double xp = 0.0;
    double yp = 0.0;
    for(int i=0;i<numPoints;++i){
        x = fieldData[XIndex]->GetValue(i);
        y = fieldData[YIndex]->GetValue(i);
        if(i>0){
            xp = fieldData[XIndex]->GetValue(i-1);
            yp = fieldData[YIndex]->GetValue(i-1);
            dist_x = std::sqrt((x-xp)*(x-xp));
            dist_y = std::sqrt((y-yp)*(y-yp));
            //if(dist_x > max_dist) max_dist = dist_x;
            //if(dist_x > 0.0 && dist_x < min_dist) min_dist = dist_x;
            //if(dist_y > max_dist) max_dist = dist_y;
            //if(dist_y > 0.0 && dist_y < min_dist) min_dist = dist_y;
            avg_dist += dist_x > dist_y ? dist_x : dist_y;
        }
    }
    avg_dist /= (numPoints-1);
    //std::cout << "min dist " << min_dist << " max dist " << max_dist << " 2 max dist " << 2.0*max_dist << std::endl;
    //std::cout << "avg " << avg_dist << " 1.5 avg " << 1.5*avg_dist << std::endl;

    // set the alpha value for the triangulation as 1.5 times the average distance
    ui->triAlphaSpin->setMinimum(0.5*avg_dist);
    ui->triAlphaSpin->setMaximum(avg_dist*3.0);
    ui->triAlphaSpin->setSingleStep((avg_dist*3.0 - avg_dist*0.5) / 5);
    ui->triAlphaSpin->setValue(avg_dist);
}



void SimpleQtVTK::createPolyData(){
    vtkSmartPointer<vtkPoints> points =
            vtkSmartPointer<vtkPoints>::New();

    // assert that the first couple fields are X Y COORDS ...
    if(fieldData.size() < 2)
        std::cout << "ERROR: fieldData.size() < 2, not enough fields to create poly data" << std::endl;

    // if the mesh is to be displaced, check that the next 2 fields are displacement
    if(ui->displaceMeshBox->isChecked()){
        if(fieldData.size() < 4)
            std::cout << "ERROR: fieldData.size() < 4, not enough fields to create poly data" << std::endl;
        for(int i=0;i<numPoints;++i){
            points->InsertNextPoint(fieldData[XIndex]->GetValue(i)+fieldData[dispXIndex]->GetValue(i),
                    fieldData[YIndex]->GetValue(i)+fieldData[dispYIndex]->GetValue(i),
                    0.01); // small offset to prevent mangling with background image
        }
    }
    else{
        for(int i=0;i<numPoints;++i){
            points->InsertNextPoint(fieldData[XIndex]->GetValue(i),
                    fieldData[YIndex]->GetValue(i),0.01); // small offset to prevent mangling with background image
        }
    }
    polyData->Initialize();
    polyData->SetPoints(points);
    for(int i=0;i<fieldData.size();++i){
        polyData->GetPointData()->AddArray(fieldData[i]);
    }
    //std::cout << " polydata has " << polyData->GetNumberOfPoints() << " points" << std::endl;
}

void SimpleQtVTK::triangulate(){
    // Triangulate the grid points
    delaunay->SetAlpha(ui->triAlphaSpin->value());
    delaunay->SetInputData(polyData);
    delaunay->Update();
}

void SimpleQtVTK::resetColorScale(const int & index){
    // color transfer function
    colorTransferFunction->RemoveAllPoints();
    const double min = fieldMins[index];
    const double max = fieldMaxs[index];
    // TODO set diverging?
    double d = (max-min)/(199);
    for(int i = 0; i < 200; i++)
        colorTransferFunction->AddRGBPoint(min + i*d, jet[i][0], jet[i][1], jet[i][2]);
}

void SimpleQtVTK::renderField(const int index){
    polyData->GetPointData()->SetActiveScalars(fieldNames[index].c_str());

    if(ui->autoScaleBox->isChecked())
        resetColorScale(index);

    // Visualize
    meshMapper->SetInputConnection(delaunay->GetOutputPort());
    meshMapper->SetScalarModeToUsePointData();
    meshMapper->SetColorModeToMapScalars();
    meshMapper->SetLookupTable(colorTransferFunction);
    meshMapper->SetScalarRange(polyData->GetScalarRange());
    meshActor->SetMapper(meshMapper);
    meshActor->GetProperty()->SetOpacity(ui->opacitySpin->value());

    scalarBar->SetLookupTable(meshMapper->GetLookupTable());
    scalarBar->SetTitle(fieldNames[index].c_str());

    glyphFilter->SetInputData(polyData);
    glyphFilter->Update();

    pointMapper->SetInputConnection(glyphFilter->GetOutputPort());
    pointActor->SetMapper(pointMapper);

    renderer->AddActor2D(meshActor);
    if(ui->showPointsBox->isChecked())
      renderer->AddActor2D(pointActor);
    if(ui->showScaleBox->isChecked())
        renderer->AddActor2D(scalarBar);

    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::readResultsFile(const std::string & fileName){
    //std::cout << "reading file: " << fileName << std::endl;
    resultsFileName = fileName;

    fieldNames.clear();
    // save off the current field index
    int currentFieldIndex = ui->fieldsCombo->currentIndex();
    ui->fieldsCombo->clear();

    std::ifstream filestream(fileName.c_str());

    // TODO get rid of the *** header lines
    std::string commentStr = "***";
    std::string header = commentStr;
    int numComments = -1;
    while(header.find(commentStr)!=std::string::npos){
      std::getline(filestream,header);
      numComments++;
    }

    // read the header line to get the names of the fields
    std::stringstream ss(header);
    std::string fieldName;
    while(std::getline(ss,fieldName,','))
        fieldNames.push_back(fieldName);
    for (int i=0; i< fieldNames.size(); i++){
        // update the fields combo box
        ui->fieldsCombo->addItem(QString::fromStdString(fieldNames[i]));
        //std::cout << "Field: " << fieldNames[i] <<std::endl;
    }
    // TODO assert that each line has the same number of points
    // read the number of points in the file:
    numPoints = std::count(std::istreambuf_iterator<char>(filestream),
                 std::istreambuf_iterator<char>(), '\n');

    // resize the data arrays
    //std::cout << "There are " << numPoints << " data points" << std::endl;
    fieldData.resize(fieldNames.size());
    fieldMins.resize(fieldNames.size());
    fieldMaxs.resize(fieldNames.size());
    for(int i=0;i<fieldMins.size();++i){
        fieldMaxs[i] = 0.0;
        fieldMins[i] = std::numeric_limits<double>::max();
    }

    for(int i=0;i<fieldData.size();++i){
        fieldData[i] = vtkSmartPointer<vtkDoubleArray>::New();
        fieldData[i]->SetNumberOfTuples(numPoints);
        fieldData[i]->SetNumberOfComponents(1);
        fieldData[i]->SetName(fieldNames[i].c_str());
    }
    //std::cout << "num fields " << fieldData.size() <<  std::endl;

    // read the field data into memory
    filestream.clear();
    filestream.seekg(0,ios::beg);

    // skip the comment lines
    for(int i=0;i<numComments+1;++i)
        std::getline(filestream,header);

    std::string line;
    int pointIndex = 0;
    while(std::getline(filestream,line)){
        std::stringstream ss_line(line);
        std::string valueStr;
        double value = 0.0;
        int fieldIndex = 0;
        while(std::getline(ss_line,valueStr,',')){
            value = std::stod(valueStr);
            if(value < fieldMins[fieldIndex])fieldMins[fieldIndex] = value;
            if(value > fieldMaxs[fieldIndex])fieldMaxs[fieldIndex] = value;
            fieldData[fieldIndex]->SetTuple(pointIndex,&value);
            //std::cout << "point " << pointIndex << " field " << fieldIndex << " value " << fieldData[fieldIndex][pointIndex] << std::endl;
            fieldIndex++;
        }
        pointIndex++;
    }
    filestream.close();

    // replace the current field if it is still valid
    if(ui->fieldsCombo->count()>currentFieldIndex && currentFieldIndex > 0)
        ui->fieldsCombo->setCurrentIndex(currentFieldIndex);

    //std::cout << " Field mins " << std::endl;
    //for(int i=0;i<fieldNames.size();++i){
    //    std::cout << i << " " << fieldMins[i] << std::endl;
    //}
    //std::cout << " Field maxs " << std::endl;
    //for(int i=0;i<fieldNames.size();++i){
    //    std::cout << i << " " << fieldMaxs[i] << std::endl;
    //}
}

void SimpleQtVTK::on_fieldsCombo_currentIndexChanged(int index)
{
    // if the data has not been loaded, return
    if(!isInitialized) return;
//    std::cout << " The active field is now " << fieldNames[index] << std::endl;
    // render the selected field
    renderField(index);

}

void SimpleQtVTK::on_opacitySpin_editingFinished()
{
    if(!isInitialized) return;
    meshActor->GetProperty()->SetOpacity(ui->opacitySpin->value());
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::on_showPointsBox_clicked()
{
    if(!isInitialized) return;
    if(ui->showPointsBox->isChecked()){
        renderer->AddActor2D(pointActor);
    }
    else{
        renderer->RemoveActor(pointActor);
    }
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::on_resetViewButton_clicked()
{
    if(!isInitialized) return;
    renderer->ResetCamera();
    //ui->vtkWidget->GetInteractor()->Initialize();
    renderer->SetActiveCamera(NULL);
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::on_showScaleBox_clicked()
{
    if(!isInitialized) return;
    if(ui->showScaleBox->isChecked()){
        renderer->AddActor2D(scalarBar);
    }
    else{
        renderer->RemoveActor(scalarBar);
    }
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::on_fileCombo_currentIndexChanged(int index)
{
    updateCurrentFile(index,ui->autoTriAlphaBox->isChecked());
}


void SimpleQtVTK::on_resetScaleButton_clicked()
{
    if(!isInitialized) return;
    resetColorScale(ui->fieldsCombo->currentIndex());
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::on_showMeshBox_clicked()
{
    if(!isInitialized) return;
    if(ui->showMeshBox->isChecked()){
        meshActor->GetProperty()->EdgeVisibilityOn();
    }
    else{
        meshActor->GetProperty()->EdgeVisibilityOff();
    }
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::on_printColorBox_clicked()
{
    if(ui->printColorBox->isChecked()){
        // change the colors to more printer friendly
        renderer->GradientBackgroundOff();
        renderer->SetBackground(1.0,1.0,1.0); // Background color
        scalarBar->GetProperty()->SetColor(0, 0, 0);
        scalarBar->GetTitleTextProperty()->SetColor(0,0,0);
        scalarBar->GetLabelTextProperty()->ShadowOff();
        // TODO figure out how to chang the axes color for printing
        //vtkTextProperty* axisLabelTextProperty = vtkTextProperty::New();
        //axisLabelTextProperty->SetFontFamilyToArial();
        //axisLabelTextProperty->SetColor( 0.0, 0.0, 0.0 );
        //axisLabelTextProperty->ShadowOff();
        //axes->GetXAxisCaptionActor2D()->SetCaptionTextProperty(axisLabelTextProperty);
        //axisLabelTextProperty->Delete();
    }
    else{
        // change the colors to more visualization friendly
        renderer->GradientBackgroundOn();
        renderer->SetBackground(0.4,0.4,0.4); // Background color
        renderer->SetBackground2(0.1,0.1,0.1);
        scalarBar->GetProperty()->SetColor(1, 1, 1);
        scalarBar->GetTitleTextProperty()->SetColor(1,1,1);
        scalarBar->GetLabelTextProperty()->ShadowOn();
        // vtkTextProperty* axisLabelTextProperty = vtkTextProperty::New();
        //axisLabelTextProperty->SetFontFamilyToArial();
        //axisLabelTextProperty->SetColor( 1,1,1 );
        //axisLabelTextProperty->ShadowOff();
        //axes->GetXAxisCaptionActor2D()->SetCaptionTextProperty(axisLabelTextProperty);
        //axisLabelTextProperty->Delete();
    }
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::on_triAlphaSpin_editingFinished()
{
    updateCurrentFile(ui->fileCombo->currentIndex(),false);
}

void SimpleQtVTK::on_showAxesBox_clicked()
{
    if(!isInitialized) return;
    if(ui->showAxesBox->isChecked()){
        orientationWidget->SetEnabled( 1 );
    }
    else{
        orientationWidget->SetEnabled( 0 );
    }
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::on_displaceMeshBox_clicked()
{
    updateCurrentFile(ui->fileCombo->currentIndex(),false);
}

void SimpleQtVTK::on_screenShotButton_clicked()
{
    // open file dialog box to select reference file
    QFileInfo screenFileInfo = QFileDialog::getSaveFileName(this,
      tr("Select file"), "/home",
      tr("Portable Network Graphics (*.png)"));

    if(screenFileInfo.fileName()=="") return;

    // Screenshot
      vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =
        vtkSmartPointer<vtkWindowToImageFilter>::New();
      windowToImageFilter->SetInput(ui->vtkWidget->GetRenderWindow());
      windowToImageFilter->SetMagnification(1); //set the resolution of the output image (3 times the current resolution of vtk render window)
      windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
      windowToImageFilter->ReadFrontBufferOff(); // read from the back buffer
      windowToImageFilter->Update();

      std::cout << "Writing .png image to " << screenFileInfo.filePath().toStdString() << std::endl;
      vtkSmartPointer<vtkPNGWriter> writer =
        vtkSmartPointer<vtkPNGWriter>::New();
      writer->SetFileName(screenFileInfo.filePath().toStdString().c_str());
      writer->SetInputConnection(windowToImageFilter->GetOutputPort());
      writer->Write();
}
