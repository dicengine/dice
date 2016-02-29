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
#include <DICe_InputVars.h>
#include <qimageroiselector.h>
#include <iostream>

DICe::gui::Input_Vars * DICe::gui::Input_Vars::input_vars_ptr_ = NULL;

MainWindow::MainWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_refFileButton_clicked()
{
    // open file dialog box to select reference file
    QFileInfo refFileInfo = QFileDialog::getOpenFileName(this,
      tr("Select reference file"), "/home",
      tr("Tagged Image File Format (*.tiff *.tif);;Portable Network Graphics (*.png);;Joint Photographic Experts Group (*.jpg *.jpeg)"));
    
    if(refFileInfo.fileName()=="") return;

    // set the reference image in the Input_Vars singleton
    DICe::gui::Input_Vars::instance()->set_ref_file_info(refFileInfo);
    
    // display the name of the file in the reference file box
    QString refFileName = refFileInfo.fileName();
    ui->ROISelector->setImage(refFileInfo);
    ui->refFileLine->setText(refFileName);
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
      tr("Select reference file"), "/home",
      tr("Tagged Image File Format (*.tiff *.tif);;Portable Network Graphics (*.png);;Joint Photographic Experts Group (*.jpg *.jpeg)"));
    
    // clear the widget list
    ui->defListWidget->clear();
    // add the names of the files to a list widget
    for(QStringList::iterator it=defFileNames.begin(); it!=defFileNames.end();++it){
        list->append(*it);
        QFileInfo currentFile = *it;
        QString shortDefFileName = currentFile.fileName();
        ui->defListWidget->addItem(shortDefFileName);
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
