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

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QFileInfo>
#include <QDirIterator>
#include <QDesktopServices>
#include <iostream>

#include <DICe_InputVars.h>
#include <simpleqtvtk.h>

#ifndef DICE_EXEC_PATH
  #error
#endif

DICe::gui::Input_Vars * DICe::gui::Input_Vars::input_vars_ptr_ = NULL;

MainWindow::MainWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // set up the analysis defaults
    ui->translationShapeCheck->setChecked(true);

    // add the initialization methods
    ui->initMethodCombo->addItem("USE_FIELD_VALUES");
    ui->initMethodCombo->addItem("USE_NEIGHBOR_VALUES");
    ui->initMethodCombo->addItem("USE_NEIGHBOR_VALUES_FIRST_STEP");
    ui->initMethodCombo->addItem("USE_PHASE_CORRELATION");
    ui->initMethodCombo->addItem("USE_OPTICAL_FLOW");

    // add the optimization methods
    ui->optMethodCombo->addItem("GRADIENT_BASED");
    ui->optMethodCombo->addItem("SIMPLEX");

    // add the interpolation methods
    ui->interpMethodCombo->addItem("KEYS_FOURTH");
    ui->interpMethodCombo->addItem("BILINEAR");

    // set up the step size and subset size
    ui->subsetSize->setMinimum(10);
    ui->subsetSize->setMaximum(50);
    ui->subsetSize->setValue(25);
    ui->stepSize->setMinimum(1);
    ui->stepSize->setMaximum(100);
    ui->stepSize->setValue(15);

    // reset the progress bar
    //ui->progressBar->setValue(0);

    // reset the default working directory
    ui->workingDirLineEdit->setText(".");
    DICe::gui::Input_Vars::instance()->set_working_dir(QString("."));

    // set up the console output
    qout = new QDebugStream(std::cout, ui->consoleEdit);

    // set up the process that will run DICe
    diceProcess = new QProcess(this);
    connect(diceProcess, SIGNAL(readyReadStandardOutput()), this, SLOT(readOutput()));

    // set the default output fields
    ui->xCheck->setChecked(true);
    ui->yCheck->setChecked(true);
    ui->dispXCheck->setChecked(true);
    ui->dispYCheck->setChecked(true);
    ui->sigmaCheck->setChecked(true);
    ui->gammaCheck->setChecked(true);
    ui->betaCheck->setChecked(true);
    ui->statusCheck->setChecked(true);

    std::cout << "Using DICe from " << DICE_EXEC_PATH << std::endl;

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_refFileButton_clicked()
{
    // open file dialog box to select reference file
    QFileInfo refFileInfo = QFileDialog::getOpenFileName(this,
      tr("Select reference file"), ".",
      tr("Tagged Image File Format (*.tiff *.tif);;Portable Network Graphics (*.png);;Joint Photographic Experts Group (*.jpg *.jpeg)"));
    
    if(refFileInfo.fileName()=="") return;

    // set the reference image in the Input_Vars singleton
    DICe::gui::Input_Vars::instance()->set_ref_file_info(refFileInfo);
    
    // display the name of the file in the reference file box
    ui->simpleQtVTKWidget->readImageFile(refFileInfo.filePath().toStdString());

    // if the deformed images are populated then activate the write and run buttons
    if(DICe::gui::Input_Vars::instance()->has_def_files()){
        ui->writeButton->setEnabled(true);
        ui->runButton->setEnabled(true);
    }
}

void MainWindow::on_defFileButton_clicked()
{
    // get a pointer to the file name list in the Input_Vars singleton
    QStringList * list = DICe::gui::Input_Vars::instance()->get_def_file_list();
    list->clear();
    
    // open a file dialog box to select the deformed images
    QFileDialog defDialog(this);
    defDialog.setFileMode(QFileDialog::ExistingFiles);
    QStringList defFileNames = defDialog.getOpenFileNames(this,
      tr("Select reference file"), ".",
      tr("Tagged Image File Format (*.tiff *.tif);;Portable Network Graphics (*.png);;Joint Photographic Experts Group (*.jpg *.jpeg)"));
    

    if(defFileNames.size()==0) return;

    // clear the widget list
    ui->defListWidget->clear();
    // add the names of the files to a list widget
    for(QStringList::iterator it=defFileNames.begin(); it!=defFileNames.end();++it){
        list->append(*it);
        QFileInfo currentFile = *it;
        QString shortDefFileName = currentFile.fileName();
        ui->defListWidget->addItem(shortDefFileName);
    }

    // update the picture in the def preview
    ui->defListWidget->setCurrentRow(0);
    on_defListWidget_itemClicked(ui->defListWidget->currentItem());

    // reset the def image
    //ui->defFileLabel->setText("");
    //ui->defImageShow->clear();

    // if the reference image is populated then activate the write and run buttons
    if(DICe::gui::Input_Vars::instance()->has_ref_file()){
        ui->writeButton->setEnabled(true);
        ui->runButton->setEnabled(true);
    }

}



void MainWindow::on_defListWidget_itemClicked(QListWidgetItem *item)
{
    // get the index of the item that was clicked
    int row = item->listWidget()->row(item);
    // determine the file name for the clicked item
    QStringList * list = DICe::gui::Input_Vars::instance()->get_def_file_list();
    QFileInfo defFile = list->at(row);
    
    // display the file in the small view below the list
    ui->defFileLabel->setText(defFile.filePath());
    QPixmap pix(defFile.filePath());
    int w = ui->defImageShow->width();
    int h = ui->defImageShow->height();
    ui->defImageShow->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));
}

void MainWindow::on_writeButton_clicked()
{
    writeInputFiles();
}

void MainWindow::on_workingDirButton_clicked()
{
    // open a file dialog and get the directory
    QString dir = QFileDialog::getExistingDirectory(this,tr("Open Directory"),".",
                      QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if(dir=="") return;

    // set the reference image in the Input_Vars singleton
    DICe::gui::Input_Vars::instance()->set_working_dir(dir);

    // display the name of the file in the reference file box
    ui->workingDirLineEdit->setText(dir);

}

void MainWindow::writeInputFiles(){
    std::cout << "Writing input files..." << std::endl << std::endl;
    // set the parameter values:
    DICe::gui::Input_Vars::instance()->set_subset_size(ui->subsetSize->value());
    DICe::gui::Input_Vars::instance()->set_step_size(ui->stepSize->value());
    DICe::gui::Input_Vars::instance()->set_initialization_method(ui->initMethodCombo->currentText().toStdString());
    DICe::gui::Input_Vars::instance()->set_optimization_method(ui->optMethodCombo->currentText().toStdString());
    DICe::gui::Input_Vars::instance()->set_interpolation_method(ui->interpMethodCombo->currentText().toStdString());
    DICe::gui::Input_Vars::instance()->set_enable_translation(ui->translationShapeCheck->isChecked());
    DICe::gui::Input_Vars::instance()->set_enable_rotation(ui->rotationShapeCheck->isChecked());
    DICe::gui::Input_Vars::instance()->set_enable_normal_strain(ui->normalShapeCheck->isChecked());
    DICe::gui::Input_Vars::instance()->set_enable_shear_strain(ui->shearShapeCheck->isChecked());
    // output fields
    std::vector<std::string> output_fields;
    if(ui->xCheck->isChecked()) output_fields.push_back("COORDINATE_X");
    if(ui->yCheck->isChecked()) output_fields.push_back("COORDINATE_Y");
    if(ui->dispXCheck->isChecked()) output_fields.push_back("DISPLACEMENT_X");
    if(ui->dispYCheck->isChecked()) output_fields.push_back("DISPLACEMENT_Y");
    if(ui->rotationCheck->isChecked()) output_fields.push_back("ROTATION_Z");
    if(ui->normalXCheck->isChecked()) output_fields.push_back("NORMAL_STRAIN_X");
    if(ui->normalYCheck->isChecked()) output_fields.push_back("NORMAL_STRAIN_Y");
    if(ui->shearCheck->isChecked()) output_fields.push_back("SHEAR_STRAIN_XY");
    if(ui->noiseCheck->isChecked()) output_fields.push_back("NOISE_LEVEL");
    if(ui->contrastCheck->isChecked()) output_fields.push_back("CONTRAST_LEVEL");
    if(ui->sigmaCheck->isChecked()) output_fields.push_back("SIGMA");
    if(ui->gammaCheck->isChecked()) output_fields.push_back("GAMMA");
    if(ui->betaCheck->isChecked()) output_fields.push_back("BETA");
    if(ui->numActiveCheck->isChecked()) output_fields.push_back("ACTIVE_PIXELS");
    if(ui->statusCheck->isChecked()) output_fields.push_back("STATUS_FLAG");
    DICe::gui::Input_Vars::instance()->set_output_fields(output_fields);

    // export the vertices from the region defs
    ui->simpleQtVTKWidget->exportVertices(DICe::gui::Input_Vars::instance()->boundaryShapes(),DICe::gui::Input_Vars::instance()->excludedShapes());
    DICe::gui::Input_Vars::instance()->write_input_file();
    DICe::gui::Input_Vars::instance()->write_params_file();
    DICe::gui::Input_Vars::instance()->write_subset_file();
}

void MainWindow::prepResultsViewer()
{
    QString dir = DICe::gui::Input_Vars::instance()->get_working_dir();
    std::stringstream working_dir_ss;
#ifdef WIN32
    working_dir_ss << dir.toStdString() << "\\results\\";
#else
    working_dir_ss << dir.toStdString() << "/results/";
#endif
    QStringList resFiles;
    QDirIterator it(QString::fromStdString(working_dir_ss.str()), QStringList() << "DICe_solution_*.txt",
                   QDir::Files,QDirIterator::Subdirectories);
    while (it.hasNext()){
       resFiles.push_back(it.next());
    }
    //for(QStringList::iterator sit=resFiles.begin();sit!=resFiles.end();++sit)
    //    std::cout << " FILE " << sit->toStdString() << std::endl;

    // get the image files
    QStringList defImageFiles = *DICe::gui::Input_Vars::instance()->get_def_file_list();

    QStringList images;
    for(QStringList::iterator sit=defImageFiles.begin();sit!=defImageFiles.end();++sit)
        images.push_back(*sit);

    //for(QStringList::iterator sit=images.begin();sit!=images.end();++sit)
    //    std::cout << " Image FILE " << sit->toStdString() << std::endl;


    if(defImageFiles.size()>0){
        try{
            ui->simpleQtVTKWidget->setFileNames(resFiles,images);
        }catch(std::exception & e){
            std::cout << "Exception was thrown !!" << e.what() << std::endl;
        }
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: there are no deformed images");
        msgBox.exec();
        exit(EXIT_FAILURE);
    }
}

void MainWindow::readOutput(){
    ui->consoleEdit->insertPlainText(diceProcess->readAllStandardOutput());
}

void MainWindow::on_runButton_clicked()
{
    std::cout << "Clearing results directory " << std::endl;
    QString dir = DICe::gui::Input_Vars::instance()->get_working_dir();
    std::stringstream working_dir_ss;
#ifdef WIN32
    working_dir_ss << dir.toStdString() << "\\results\\";
#else
    working_dir_ss << dir.toStdString() << "/results/";
#endif
    QStringList resFiles;
    QDirIterator it(QString::fromStdString(working_dir_ss.str()), QStringList() << "*.txt",
                   QDir::Files,QDirIterator::Subdirectories);
    while (it.hasNext()){
        std::cout << "Removing file " << it.next().toStdString() << std::endl;
        QDir cDir;
        cDir.remove(it.filePath());
    }

    writeInputFiles();
    std::cout << "Running correlation..." << std::endl << std::endl;

    QString diceExec = DICE_EXEC_PATH;
    QStringList args;
    args << "-i" << DICe::gui::Input_Vars::instance()->input_file_name().c_str() << "-v" << "-t";

    try{
        diceProcess->start(diceExec,args);
        diceProcess->waitForFinished();
        diceProcess->close();
    }catch(std::exception & e){
        std::cout << "DICe execution FAILED" << std::endl;
        std::cout << "Exception was thrown !!" << e.what() << std::endl;
    }
    std::cout << "DICe execution SUCCESSFUL" << std::endl;

    // after the run is complete, prepare the results viewer:
    prepResultsViewer();
}

void MainWindow::on_diceButton_clicked()
{
    QDesktopServices::openUrl(QUrl("https://github.com/dicengine/dice", QUrl::TolerantMode));
}
