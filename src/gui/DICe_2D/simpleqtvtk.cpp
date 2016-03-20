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
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkTextProperty.h>
#include <vtkTIFFReader.h>
#include <vtkImageData.h>
#include <vtkImageMapper3D.h>
#include <vtkColorSeries.h>
#include <vtkCaptionActor2D.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkCamera.h>
#include <vtkImageProperty.h>
#include <vtkImageStencil.h>
#include <vtkPolyDataToImageStencil.h>

vtkStandardNewMacro(PolygonMouseInteractorStyle);

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

    // set the style of the interaction and flip the orientation for y to down
    style = vtkSmartPointer<PolygonMouseInteractorStyle>::New();
    ui->vtkWidget->GetInteractor()->SetInteractorStyle(style);
    style->SetCurrentRenderer(renderer);
    axes = vtkSmartPointer<vtkAxesActor>::New();
    orientationWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    orientationWidget->SetOutlineColor( 0.9300, 0.5700, 0.1300 );
    orientationWidget->SetOrientationMarker( axes );
    orientationWidget->SetInteractor( ui->vtkWidget->GetInteractor() );
    orientationWidget->SetEnabled( 1 );
    orientationWidget->InteractiveOn();

    for(int i=0;i<3;++i){
        imageOrigin[i] = 0.0;
        imageSpacing[i] = 0.0;
    }
    for(int i=0;i<6;++i){
        imageExtent[i] = 0.0;
    }

    // Annotate the image with window/level and mouse over pixel
    // information
    cornerAnnotation = vtkSmartPointer<vtkCornerAnnotation>::New();
    cornerAnnotation->SetLinearFontScaleFactor(2);
    cornerAnnotation->SetNonlinearFontScaleFactor(1);
    cornerAnnotation->SetMaximumFontSize(20);
    cornerAnnotation->SetText(0, "(-,-,-)");
    cornerAnnotation->GetTextProperty()->SetColor(1, 1, 1);
    cornerAnnotation->GetTextProperty()->SetFontFamilyToCourier();
    renderer->AddViewProp(cornerAnnotation);

    // Picker to pick pixels
    propPicker = vtkSmartPointer<vtkPropPicker>::New();
    propPicker->PickFromListOn();
    // Give the picker a prop to pick
    propPicker->AddPickList(imageActor);
    // disable interpolation, so we can see each pixel
    imageActor->InterpolateOff();

    // Callback listens to MouseMoveEvents invoked by the interactor's style
    callback = vtkSmartPointer<vtkImageInteractionCallback>::New();
    callback->SetPointers(propPicker,cornerAnnotation,ui->vtkWidget->GetInteractor(),
                          renderer,imageActor,imageActor->GetInput());
    // InteractorStyleImage allows for the following controls:
    // 1) middle mouse + move = camera pan
    // 2) left mouse + move = window/level
    // 3) right mouse + move = camera zoom
    // 4) middle mouse wheel scroll = zoom
    // 5) 'r' = reset window/level
    // 6) shift + 'r' = reset camera
    style->AddObserver(vtkCommand::MouseMoveEvent, callback);
}

void SimpleQtVTK::updateCurrentFile(const int fileIndex, const bool resetAlpha){
    isInitialized = false;

    QFileInfo result = resultsFiles[fileIndex];
    readResultsFile(result.filePath().toStdString());
    std::stringstream label;
    label << result.filePath().toStdString() << "\n";
    label << "Digital Image Correlation Engine 1.0";
    cornerAnnotation->SetText(1, label.str().c_str());

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
        readImageFile(image.filePath().toStdString());
    }
    //resetCamera();
    //renderer->ResetCamera();
    //renderer->SetActiveCamera(NULL);
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

void SimpleQtVTK::resetCamera(){
    if(imageExtent[3]==0.0 && imageExtent[2]==0.0) return; // no image in view or uninitialized
    vtkCamera* camera = renderer->GetActiveCamera();
    camera->ParallelProjectionOn();
    double xc = imageOrigin[0] + 0.5*(imageExtent[0] + imageExtent[1])*imageSpacing[0];
    double yc = imageOrigin[1] + 0.5*(imageExtent[2] + imageExtent[3])*imageSpacing[1];
    double yd = (imageExtent[3] - imageExtent[2] + 1)*imageSpacing[1];
    double d = camera->GetDistance();
    camera->SetParallelScale(0.5*yd);
    camera->SetFocalPoint(xc,yc,0.0);
    camera->SetPosition(xc,yc,-d);
    camera->SetViewUp(0.0,-1.0,0.0);
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::readImageFile(const std::string & fileName){
    vtkSmartPointer<vtkImageData> imageData;
    vtkSmartPointer<vtkTIFFReader> tifReader =
            vtkSmartPointer<vtkTIFFReader>::New();
    std::stringstream label;
    label << fileName << "\n";
    label << "Digital Image Correlation Engine 1.0";
    cornerAnnotation->SetText(1, label.str().c_str());
    if( !tifReader->CanReadFile( fileName.c_str() ) )
    {
        std::cerr << "Error reading file " << fileName << std::endl;
    }
    tifReader->SetFileName ( fileName.c_str() );
    tifReader->Update();
    imageData = tifReader->GetOutput();

    // Create an image actor to display the image
    imageActor->SetInputData(imageData);
    renderer->AddActor(imageActor);

    // set the focal point to the center of the image
    renderer->ResetCamera();
    //double posx,posy,posz;
    //renderer->GetActiveCamera()->GetPosition(posx,posy,posz);
    //std::cout << "Position: " << posx << " " << posy << " " << posz << std::endl;
    //double fcx,fcy,fcz;
    //renderer->GetActiveCamera()->GetFocalPoint(fcx,fcy,fcz);
    //std::cout << "Focal point: " << fcx << " " << fcy << " " << fcz << std::endl;
    //renderer->GetActiveCamera()->ParallelProjectionOn();
    imageData->GetOrigin( imageOrigin );
    imageData->GetSpacing( imageSpacing );
    imageData->GetExtent( imageExtent );

    // update the polygon interactor data
    style->setImageExtents(imageExtent[0],imageExtent[1],
            imageExtent[2],imageExtent[3],imageSpacing[1]);

    // update the dynamic coords display variables
    callback->SetOriginSpacing(imageOrigin[0],imageOrigin[1],imageSpacing[0],imageSpacing[1]);

    //for(int i=0;i<3;++i){
    //    std::cout << " origin " << imageOrigin[i] << std::endl;
    //}
    //for(int i=0;i<3;++i){
    //    std::cout << " spacing " << imageSpacing[i] << std::endl;
    //}
    //for(int i=0;i<6;++i){
    //    std::cout << " extent " << imageExtent[i] << std::endl;
    //}
    resetCamera();
}

void SimpleQtVTK::estimateTriAlpha(){
    // assert that the first couple fields are X Y COORDS ...
    if(fieldData.size() < 2)
        std::cout << "ERROR: fieldData.size() < 2, not enough fields to create poly data" << std::endl;

    // TODO there's probably a better way to figure out alpha than computing the average distance...
    double dist_x = 0.0;
    double dist_y = 0.0;
    double avg_dist = 0.0;
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
    resetCamera();
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
        cornerAnnotation->GetTextProperty()->SetColor(0,0,0);
        // TODO figure out how to chang the axes color for printing
    }
    else{
        // change the colors to more visualization friendly
        renderer->GradientBackgroundOn();
        renderer->SetBackground(0.4,0.4,0.4); // Background color
        renderer->SetBackground2(0.1,0.1,0.1);
        scalarBar->GetProperty()->SetColor(1, 1, 1);
        scalarBar->GetTitleTextProperty()->SetColor(1,1,1);
        scalarBar->GetLabelTextProperty()->ShadowOn();
        cornerAnnotation->GetTextProperty()->SetColor(1, 1, 1);
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
      tr("Select file"), "DICeScreenShot",
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

void SimpleQtVTK::on_boundaryPlus_clicked()
{
    if(ui->boundaryPlus->isChecked()){
        //ui->excludedPlus->setEnabled(false);
        ui->excludedPlus->setChecked(false);
        //ui->boundaryMinus->setEnabled(true);
        style->setExcludedEnabled(false);
        style->setBoundaryEnabled(true);
        style->resetShapesInProgress();
    }
    else{
        style->setBoundaryEnabled(false);
        //ui->excludedPlus->setEnabled(true);
        style->resetShapesInProgress();
    }
}

void SimpleQtVTK::on_excludedPlus_clicked()
{
    if(ui->excludedPlus->isChecked()){
        //ui->boundaryPlus->setEnabled(false);
        ui->boundaryPlus->setChecked(false);
        //ui->excludedMinus->setEnabled(true);
        style->setBoundaryEnabled(false);
        style->setExcludedEnabled(true);
        style->resetShapesInProgress();
    }
    else{
        style->setExcludedEnabled(false);
        //ui->boundaryPlus->setEnabled(true);
        style->resetShapesInProgress();
    }
}

void SimpleQtVTK::on_hideCoordsBox_clicked()
{
    if(ui->hideCoordsBox->isChecked()){
        renderer->RemoveViewProp(cornerAnnotation);
    }
    else{
        renderer->AddViewProp(cornerAnnotation);
    }
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::on_boundaryMinus_clicked()
{
    style->decrementPolygon(false);
}

void SimpleQtVTK::on_excludedMinus_clicked()
{
    style->decrementPolygon(true);
}

void SimpleQtVTK::on_hideImageBox_clicked()
{
    if(ui->hideImageBox->isChecked()){
        renderer->RemoveActor(imageActor);
    }
    else{
        renderer->AddActor(imageActor);
    }
    if(ui->vtkWidget->GetRenderWindow()->IsDrawable())
        ui->vtkWidget->GetRenderWindow()->Render();
}

void SimpleQtVTK::on_resetGeoButton_clicked()
{
    style->clearPolygons();
}

void SimpleQtVTK::on_tabWidget_tabBarClicked(int index)
{
    changeInteractionMode(index);
}

void SimpleQtVTK::changeInteractionMode(const int mode){
    std::cout << " I was clicked " << mode << std::endl;
    style->resetShapesInProgress();
    if(mode==0) // region selection mode
    {
        ui->definePage->setEnabled(true);
        ui->visualizePage->setEnabled(false);
        ui->boundaryPlus->setEnabled(true);
        ui->excludedPlus->setEnabled(true);
        // turn on the region actors
        style->addActors();
        // turn off the results actors
        //renderer->RemoveActor(meshActor);
        //renderer->RemoveActor(pointActor);
    }
    else if(mode==1){
        ui->definePage->setEnabled(false);
        ui->visualizePage->setEnabled(true);
        ui->boundaryPlus->setChecked(false);
        ui->excludedPlus->setChecked(false);
        ui->boundaryPlus->setEnabled(false);
        ui->excludedPlus->setEnabled(false);
        style->setBoundaryEnabled(false);
        style->setExcludedEnabled(false);
        // turn off the region actors
        style->removeActors();
        // add results actors
        //renderer->AddActor(meshActor);
        //renderer->AddActor(pointActor);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

PolygonMouseInteractorStyle::PolygonMouseInteractorStyle(){
    currentPoints = vtkSmartPointer<vtkPoints>::New();
    currentPointsP1 = vtkSmartPointer<vtkPoints>::New();
    mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    actor = vtkSmartPointer<vtkActor>::New();
    existingMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    existingActor = vtkSmartPointer<vtkImageActor>::New();
    lineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    lineActor = vtkSmartPointer<vtkActor>::New();
    lineActor->GetProperty()->SetColor(1.0,1.0,0.0);
    lineActor->GetProperty()->SetLineWidth(2);
    boundaryLineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    boundaryLineActor = vtkSmartPointer<vtkActor>::New();
    boundaryLineActor->GetProperty()->SetColor(0.0,1.0,0.0);
    boundaryLineActor->GetProperty()->SetLineWidth(2);
    // TODO make the lines dashed
    excludedLineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    excludedLineActor = vtkSmartPointer<vtkActor>::New();
    excludedLineActor->GetProperty()->SetColor(1.0,0.0,0.0);
    excludedLineActor->GetProperty()->SetOpacity(0.5);
    excludedLineActor->GetProperty()->SetLineWidth(2);
    coordinate = vtkSmartPointer<vtkCoordinate>::New();
    coordinate->SetCoordinateSystemToDisplay();
    actor->GetProperty()->SetOpacity(0.6);
    existingActor->GetProperty()->SetOpacity(1.0);
    maskImage = vtkSmartPointer<vtkImageData>::New();
    maskLUT = vtkSmartPointer<vtkLookupTable>::New();
    maskLUT->SetNumberOfTableValues(2);
    maskLUT->SetRange(0.0,1.0);
    maskLUT->SetTableValue( 0, 0.0, 0.0, 0.0, 0.0 ); //label 0 is transparent
    maskLUT->SetTableValue( 1, 0.0, 1.0, 0.0, 0.5 ); //label 1 is opaque and green
    maskLUT->Build();
    mapTransparency = vtkSmartPointer<vtkImageMapToColors>::New();
    mapTransparency->SetLookupTable(maskLUT);
    mapTransparency->PassAlphaToOutputOn();
    imageStartX = 0.0;
    imageStartY = 0.0;
    imageEndX = 0.0;
    imageEndY = 0.0;
    boundaryEnabled = false;
    excludedEnabled = false;
}
void PolygonMouseInteractorStyle::addActors(){
    if(excludedPointsVector.size()>0)
        this->GetCurrentRenderer()->AddActor(excludedLineActor);
    if(boundaryPointsVector.size()>0)
        this->GetCurrentRenderer()->AddActor(boundaryLineActor);
    if(boundaryPointsVector.size()>0 || excludedPointsVector.size() >0)
        this->GetCurrentRenderer()->AddActor(existingActor);
    this->GetCurrentRenderer()->GetRenderWindow()->Render();
}

void PolygonMouseInteractorStyle::removeActors(){
    this->GetCurrentRenderer()->RemoveActor(excludedLineActor);
    this->GetCurrentRenderer()->RemoveActor(boundaryLineActor);
    this->GetCurrentRenderer()->RemoveActor(existingActor);
    this->GetCurrentRenderer()->GetRenderWindow()->Render();
}

void PolygonMouseInteractorStyle::setBoundaryEnabled(const bool flag){
    boundaryEnabled = flag;
}

void PolygonMouseInteractorStyle::setExcludedEnabled(const bool flag){
    excludedEnabled = flag;
}

void PolygonMouseInteractorStyle::setImageExtents(const double & startX, const double & endX,
                     const double & startY, const double & endY, const double & spacing){
    imageSpacing = spacing;
    imageStartX = startX*imageSpacing;
    imageStartY = startY*imageSpacing;
    imageEndX = endX*imageSpacing;
    imageEndY = endY*imageSpacing;
    //std::cout << "Image extents have been set to " << imageStartX << " " << imageEndX << " " << imageStartY << " " << imageEndY << std::endl;
    initializeMaskImage();
}

int PolygonMouseInteractorStyle::clockwise(const double* a, const double *b, const double*c){
    double valueDbl = ((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]));
    int value = valueDbl>0.0? 1: valueDbl<0.0? -1 : 0;
    if(value!=0&&value!=1&&value!=-1) {
        std::cout << "ERROR: invalid clockwise vlaue" << std::endl;
        return 2;
    }
    return value;
}
bool PolygonMouseInteractorStyle::isIntersecting(const double* ap, const double* aq, const double *bp, const double *bq){
    if (clockwise(ap,aq,bp)*clockwise(ap,aq,bq)>0) return false;
    if (clockwise(bp,bq,ap)*clockwise(bp,bq,aq)>0) return false;
    return true;
}

// 1 is valid polygon, -1 is degenerate points, -2 is co-linear points, -3 is self-intersecting or colinear, -4 too few points
int PolygonMouseInteractorStyle::isValidPolygon(){
    // compute the sum angle of all the vertices:
    const int numPoints = currentPoints->GetNumberOfPoints();
    const double tol = 1.0E-3;
    if(numPoints < 3){
        return -4;
    }
    for(int i=0;i<numPoints;++i){
        double ipt[3]; // left end
        currentPoints->GetPoint(i,ipt);
        double iqpt[3]; // right end
        if (i==numPoints-1) currentPoints->GetPoint(0,iqpt);
        else currentPoints->GetPoint(i+1,iqpt);
        // check for degenerate points
        for(int j=0;j<numPoints;++j){
            if(j==i||j==i+1||j==i-1) continue;
            if(i==0&&j==numPoints-1) continue;
            if(i==numPoints-1&&j==0) continue;
            double jpt[3];
            currentPoints->GetPoint(j,jpt);
            if(std::abs(ipt[0]-jpt[0]) + std::abs(ipt[1]-jpt[1]) < tol) return -1;
            // check for intersections
            double jqpt[3]; // right end
            if (j==numPoints-1) currentPoints->GetPoint(0,jqpt);
            else currentPoints->GetPoint(j+1,jqpt);
            if(isIntersecting(ipt,iqpt,jpt,jqpt)) return -3;
        }
        // check for colinear points
        double vl[3];
        double vr[3];
        if(i==0) currentPoints->GetPoint(numPoints-1,vl);
        else currentPoints->GetPoint(i-1,vl);
        if(i==numPoints-1) currentPoints->GetPoint(0,vr);
        else currentPoints->GetPoint(i+1,vr);
        if(clockwise(vl,ipt,vr)==0) return -2;
    }
    return 1;
}

void PolygonMouseInteractorStyle::clearPolygons(){
    boundaryPointsVector.resize(0);
    excludedPointsVector.resize(0);
    drawExistingPolygons();
}

bool PolygonMouseInteractorStyle::decrementPolygon(const bool excluded){
    if(excluded){
        if(excludedPointsVector.size()<1)return true;
        excludedPointsVector.resize(excludedPointsVector.size()-1);
    }
    else{
        if(boundaryPointsVector.size()<1)return true;
        boundaryPointsVector.resize(boundaryPointsVector.size()-1);
    }
    drawExistingPolygons();
    if(excluded) return excludedPointsVector.empty(); // return true if empty
    else return excludedPointsVector.empty();
}

void PolygonMouseInteractorStyle::OnRightButtonDown()
{
    if(boundaryEnabled || excludedEnabled){
        if(currentPoints->GetNumberOfPoints()>2){
            // check for valid polygon
            int validFlag = isValidPolygon();
            // 1 is valid polygon, -1 is degenerate points, -2 is co-linear points, -3 is self-intersecting or colinear, -4 too few points
            if(validFlag==-1){
                QMessageBox msgBox;
                msgBox.setText("Error: Invalid polygon, degenerate points detected.");
                msgBox.exec();

            }
            if(validFlag==-2){
                QMessageBox msgBox;
                msgBox.setText("Error: Invalid polygon, colinear vertices detected.");
                msgBox.exec();
            }
            if(validFlag==-3){
                QMessageBox msgBox;
                msgBox.setText("Error: Invalid polygon, self-intersection detected.");
                msgBox.exec();
            }
            if(validFlag==-4){
                QMessageBox msgBox;
                msgBox.setText("Error: Invalid polygon, too few points.");
                msgBox.exec();
            }
            if(boundaryEnabled && validFlag==1){
                boundaryPointsVector.resize(boundaryPointsVector.size()+1);
                boundaryPointsVector[boundaryPointsVector.size()-1] = vtkSmartPointer<vtkPoints>::New();
                boundaryPointsVector[boundaryPointsVector.size()-1]->DeepCopy(currentPoints);
            }
            else if(validFlag==1){
                excludedPointsVector.resize(excludedPointsVector.size()+1);
                excludedPointsVector[excludedPointsVector.size()-1] = vtkSmartPointer<vtkPoints>::New();
                excludedPointsVector[excludedPointsVector.size()-1]->DeepCopy(currentPoints);
            }
        }
        /// clear the currentPoints
        this->GetCurrentRenderer()->RemoveActor(actor);
        this->GetCurrentRenderer()->RemoveActor(lineActor);
        this->GetCurrentRenderer()->GetRenderWindow()->Render();
        if(currentPoints->GetNumberOfPoints()>2) drawExistingPolygons();
        currentPoints->Reset();
        currentPointsP1->Reset();
    }
    else{
        // Forward events
        vtkInteractorStyleImage::OnRightButtonDown();
    }
}

void PolygonMouseInteractorStyle::resetShapesInProgress(){
    this->GetCurrentRenderer()->RemoveActor(actor);
    this->GetCurrentRenderer()->RemoveActor(lineActor);
    this->GetCurrentRenderer()->GetRenderWindow()->Render();
    drawExistingPolygons();
    currentPoints->Reset();
    currentPointsP1->Reset();
}

void PolygonMouseInteractorStyle::drawExistingLines(const bool useExcluded){
    if(useExcluded) this->GetCurrentRenderer()->RemoveActor(excludedLineActor);
    else this->GetCurrentRenderer()->RemoveActor(boundaryLineActor);
    std::vector<vtkSmartPointer<vtkPoints> > * PointsVector = useExcluded ? &excludedPointsVector : &boundaryPointsVector;
    const int loopSize = PointsVector->size();

    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPolyData> linesPolyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
    int index = 0;
    for(int id=0;id<loopSize;++id){
        const int numPts = (*PointsVector)[id]->GetNumberOfPoints();
        int beginIndex = 0;
        for(int j=0;j<numPts;++j){
            if(j==0) beginIndex = index;
            double pt[3];
            (*PointsVector)[id]->GetPoint(j,pt);
            pts->InsertNextPoint(pt);
            vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
            line->GetPointIds()->SetId(0, index);
            if(j==numPts-1) line->GetPointIds()->SetId(1, beginIndex);
            else line->GetPointIds()->SetId(1, index+1);
            lines->InsertNextCell(line);
            index++;
        }
    }
    linesPolyData->SetPoints(pts);
    linesPolyData->SetLines(lines);
    if(useExcluded){
        excludedLineMapper->SetInputData(linesPolyData);
        excludedLineActor->SetMapper(excludedLineMapper);
        this->GetCurrentRenderer()->AddActor(excludedLineActor);
    }
    else{
        boundaryLineMapper->SetInputData(linesPolyData);
        boundaryLineActor->SetMapper(boundaryLineMapper);
        this->GetCurrentRenderer()->AddActor(boundaryLineActor);
    }
}


void PolygonMouseInteractorStyle::drawLines(vtkSmartPointer<vtkPoints> ptSet){
    const int numPoints = ptSet->GetNumberOfPoints();
    if(numPoints < 2) {
        this->GetCurrentRenderer()->RemoveActor(lineActor);
        return;
    }
    vtkSmartPointer<vtkPolyData> linesPolyData = vtkSmartPointer<vtkPolyData>::New();
    linesPolyData->SetPoints(ptSet);
    // draw the lines
    vtkSmartPointer<vtkCellArray> lines =
            vtkSmartPointer<vtkCellArray>::New();
    for(int i=0;i<numPoints;++i){
        vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
        line->GetPointIds()->SetId(0, i); // the second 0 is the index of the Origin in linesPolyData's points
        if (i==numPoints-1)
            line->GetPointIds()->SetId(1, 0); // the second 1 is the index of P0 in linesPolyData's points
        else
            line->GetPointIds()->SetId(1, i+1); // the second 1 is the index of P0 in linesPolyData's points
        lines->InsertNextCell(line);
    }
    linesPolyData->SetLines(lines);
    lineMapper->SetInputData(linesPolyData);
    lineActor->SetMapper(lineMapper);
    this->GetCurrentRenderer()->AddActor(lineActor);
    this->GetCurrentRenderer()->GetRenderWindow()->Render();
}

void PolygonMouseInteractorStyle::initializeMaskImage(){
    int bounds[6];
    bounds[0] = 0;
    bounds[1] = (int)(imageEndX/imageSpacing);
    bounds[2] = 0;
    bounds[3] = (int)(imageEndY/imageSpacing);
    bounds[4] = 0;
    bounds[5] = 0;
    double spacing[3]; // desired volume spacing
    spacing[0] = imageSpacing;
    spacing[1] = imageSpacing;
    spacing[2] = imageSpacing;
    maskImage->SetSpacing(spacing);
    // compute dimensions
    maskImage->SetExtent(bounds);
    double origin[3];
    origin[0] = 0.0;
    origin[1] = 0.0;
    origin[2] = 0.0;
    maskImage->SetOrigin(origin);
}

void PolygonMouseInteractorStyle::markPixelsInPolygon(const double & value, const bool useExcluded){
    std::vector<vtkSmartPointer<vtkPoints> > * PointsVector = useExcluded ? &excludedPointsVector : &boundaryPointsVector;
    const int loopSize = PointsVector->size();
    for(int id=0;id<loopSize;++id){
        const int numPts = (*PointsVector)[id]->GetNumberOfPoints();
        double minX = std::numeric_limits<double>::max();
        double maxX = 0.0;
        double minY = std::numeric_limits<double>::max();
        double maxY = 0.0;
        double pt[3];
        for(int j=0;j<numPts;++j){
            (*PointsVector)[id]->GetPoint(j,pt);
            if(pt[0] < minX) minX = pt[0];
            if(pt[0] > maxX) maxX = pt[0];
            if(pt[1] < minY) minY = pt[1];
            if(pt[1] > maxY) maxY = pt[1];
        }
        int min_x = (int)(minX/imageSpacing) - 1;
        if(min_x < 0) min_x = 0;
        int max_x = (int)(maxX/imageSpacing) + 1;
        if(max_x > (int)(imageEndX/imageSpacing)) max_x = (int)(imageEndX/imageSpacing);
        int min_y = (int)(minY/imageSpacing) - 1;
        if(min_y < 0) min_y = 0;
        int max_y = (int)(maxY/imageSpacing) + 1;
        if(max_y > (int)(imageEndY/imageSpacing)) max_y = (int)(imageEndY/imageSpacing);
        int width = (int)(imageEndX/imageSpacing)+1;
        //int height = (int)(imageEndY/imageSpacing)+1;

        // check if each point in the box is in the shape or not:
        double dx1=0.0,dx2=0.0,dy1=0.0,dy2=0.0;
        double ptL[3];
        double ptR[3];
        double angle=0.0;
        // rip over the points in the extents of the polygon to determine which onese are inside

        for(int y=min_y;y<=max_y;++y){
            for(int x=min_x;x<=max_x;++x){
                // x and y are the global coordinates of the point to test
                angle=0.0;
                for (int i=0;i<numPts;++i) {
                    (*PointsVector)[id]->GetPoint(i,ptL);
                    if(i==numPts-1)(*PointsVector)[id]->GetPoint(0,ptR);
                    else (*PointsVector)[id]->GetPoint(i+1,ptR);
                    // get the two end points of the polygon side and construct
                    // a vector from the point to each one:
                    dx1 = ptL[0]/imageSpacing - x;
                    dy1 = ptL[1]/imageSpacing - y;
                    dx2 = ptR[0]/imageSpacing - x;
                    dy2 = ptR[1]/imageSpacing - y;
                    double dtheta=0.0,theta1=0.0,theta2=0.0;
                    theta1 = std::atan2(dy1,dx1);
                    theta2 = std::atan2(dy2,dx2);
                    dtheta = theta2 - theta1;
                    while (dtheta > vtkMath::Pi())
                        dtheta -= vtkMath::Pi()*2.0;
                    while (dtheta < -vtkMath::Pi())
                        dtheta += vtkMath::Pi()*2.0;
                    angle += dtheta;
                }
                // if the angle is greater than PI, the point is in the polygon
                if(std::abs(angle) >= vtkMath::Pi()){
                    // paint pixel black
                    maskImage->GetPointData()->GetScalars()->SetTuple1(y*width + x, value);
                }
            }
        }
    }
}

void PolygonMouseInteractorStyle::drawExistingPolygons(){
    if(imageSpacing==0.0) return;
    this->GetCurrentRenderer()->RemoveActor(existingActor);
    maskImage->AllocateScalars(VTK_DOUBLE,1);
    // fill the image with foreground
    vtkIdType count = maskImage->GetNumberOfPoints();
    for (vtkIdType i = 0; i < count; ++i){
        maskImage->GetPointData()->GetScalars()->SetTuple1(i, 0.0);
    }
    // mark the boundary included pixels
    markPixelsInPolygon(1.0);
    // mark the excluded pixels
    markPixelsInPolygon(0.0,true);
    // add an actor to show the image on the screen
    mapTransparency->SetInputData(maskImage);
    existingActor->GetMapper()->SetInputConnection(mapTransparency->GetOutputPort());
    this->GetCurrentRenderer()->AddActor(existingActor);
    drawExistingLines(false);
    drawExistingLines(true);
    this->GetCurrentRenderer()->GetRenderWindow()->Render();
}

void PolygonMouseInteractorStyle::drawPolygon(vtkSmartPointer<vtkPoints> ptSet){
    const int numPoints = ptSet->GetNumberOfPoints();
    if(numPoints < 3) return;
    if(boundaryEnabled){
        actor->GetProperty()->SetColor(0.0,1.0,0.0);
    }else{
        actor->GetProperty()->SetColor(1.0,0.0,0.0);
    }
    vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();
    vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkPolyData> polygonPolyData = vtkSmartPointer<vtkPolyData>::New();
    polygon->GetPointIds()->SetNumberOfIds(numPoints);
    for(int i=0;i<numPoints;++i)
        polygon->GetPointIds()->SetId(i, i);
    polygons->InsertNextCell(polygon);
    polygonPolyData->SetPoints(ptSet);
    polygonPolyData->SetPolys(polygons);
    vtkSmartPointer<vtkTriangleFilter> triangleFilter =
            vtkSmartPointer<vtkTriangleFilter>::New();
    triangleFilter->SetInputData(polygonPolyData);
    triangleFilter->Update();
    mapper->SetInputConnection(triangleFilter->GetOutputPort());
    actor->SetMapper(mapper);
    this->GetCurrentRenderer()->AddActor(actor);
    this->GetCurrentRenderer()->GetRenderWindow()->Render();
}

void PolygonMouseInteractorStyle::OnMouseMove()
{
    if(boundaryEnabled || excludedEnabled){
        const int numPts = currentPoints->GetNumberOfPoints();
        if(numPts < 1){
            // Forward events
            vtkInteractorStyleImage::OnMouseMove();
            return;
        }
        currentPointsP1->Resize(numPts);
        int x = this->Interactor->GetEventPosition()[0];
        int y = this->Interactor->GetEventPosition()[1];
        //vtkSmartPointer<vtkCoordinate> coordinate =
        //        vtkSmartPointer<vtkCoordinate>::New();
        coordinate->SetValue(x,y,0);
        double* world = coordinate->GetComputedWorldValue(this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer());
        //std::cout << "World coordinate: " << world[0] << ", " << world[1] << ", " << world[2] << std::endl;
        currentPointsP1->InsertNextPoint(world[0],world[1], -0.01); // offset from 0 in z so that it always displays over image
        drawLines(currentPointsP1);
        drawPolygon(currentPointsP1);
    }
    else{
        // Forward events
        vtkInteractorStyleImage::OnMouseMove();
    }
}

void PolygonMouseInteractorStyle::OnLeftButtonDown()
{
    if(boundaryEnabled || excludedEnabled){
        int x = this->Interactor->GetEventPosition()[0];
        int y = this->Interactor->GetEventPosition()[1];
        coordinate->SetValue(x,y,0);
        double* world = coordinate->GetComputedWorldValue(this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer());
        //std::cout << "World coordinate: " << world[0] << ", " << world[1] << ", " << world[2] << std::endl;
        if(world[0]>imageStartX && world[0]<imageEndX && world[1] > imageStartY && world[1] < imageEndY){
            // round the points
            currentPoints->InsertNextPoint(world[0],world[1], -0.01);// offset from 0 in z so that it always displays over image
            currentPointsP1->InsertNextPoint(world[0],world[1], -0.01);// offset from 0 in z so that it always displays over image
        }
        if(boundaryEnabled){
            actor->GetProperty()->SetColor(0.0,1.0,0.0);
        }else{
            actor->GetProperty()->SetColor(1.0,0.0,0.0);
        }
    }
    else{
        // Forward events
        vtkInteractorStyleImage::OnLeftButtonDown();
    }
}

void PolygonMouseInteractorStyle::exportVertices(QList<QList<QPoint> > * boundary, QList<QList<QPoint> > * excluded){
    boundary->clear();
    excluded->clear();
    if(imageSpacing==0.0) return;
    for(int id=0;id<boundaryPointsVector.size();++id){
        const int numPts = boundaryPointsVector[id]->GetNumberOfPoints();
        QList<QPoint> polygon;
        double pt[3];
        for(int j=0;j<numPts;++j){
            boundaryPointsVector[id]->GetPoint(j,pt);
            QPoint point(vtkMath::Round(pt[0]/imageSpacing),vtkMath::Round(pt[1]/imageSpacing));
            polygon.append(point);
        }
        boundary->append(polygon);
    }
    for(int id=0;id<excludedPointsVector.size();++id){
        const int numPts = excludedPointsVector[id]->GetNumberOfPoints();
        QList<QPoint> polygon;
        double pt[3];
        for(int j=0;j<numPts;++j){
            excludedPointsVector[id]->GetPoint(j,pt);
            QPoint point(vtkMath::Round(pt[0]/imageSpacing),vtkMath::Round(pt[1]/imageSpacing));
            polygon.append(point);
        }
        excluded->append(polygon);
    }
}

