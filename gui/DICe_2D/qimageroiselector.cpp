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

#include "qimageroiselector.h"
#include "ui_qimageroiselector.h"
#include "DICe_InputVars.h"

QImageROISelector::QImageROISelector(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::QImageROISelector)
{
    ui->setupUi(this);
    connect(ui->selectionArea, SIGNAL(mousePos()), this, SLOT(on_selectionAreaMouseMove()));
}

QImageROISelector::~QImageROISelector()
{
    delete ui;
}

void QImageROISelector::setImage(QFileInfo & file)
{
    ui->selectionArea->clearShapesSet();
    ui->imageFileLabel->setText(file.filePath());
    ui->selectionArea->resetImagePan();
    ui->selectionArea->openImage(file.filePath());
    ui->imageDims->setText(QString("W = %1, H = %2").arg(ui->selectionArea->getOriginalImageWidth()).arg(ui->selectionArea->getOriginalImageHeight()));
}

void QImageROISelector::on_boundaryPlus_clicked()
{
    // turn the add boundary tool on or off, depending on the current state swap it
    ui->selectionArea->setAddBoundaryEnabled(!ui->selectionArea->getAddBoundaryEnabled());
    // test if the user is in the process of drawing
    // another line, if so, delet it
    ui->selectionArea->decrementShapeSet(false,true);
    ui->selectionArea->setAddExcludedEnabled(false);
    ui->excludedPlus->setChecked(false);
}

void QImageROISelector::on_boundaryMinus_clicked()
{
  ui->selectionArea->decrementShapeSet();
}

void QImageROISelector::on_excludedPlus_clicked()
{
    // turn the add boundary tool on or off, depending on the current state swap it
    ui->selectionArea->setAddExcludedEnabled(!ui->selectionArea->getAddExcludedEnabled());
    // test if the user is in the process of drawing
    // another line, if so, delet it
    ui->selectionArea->decrementShapeSet(true,true);
    ui->selectionArea->setAddBoundaryEnabled(false);
    ui->boundaryPlus->setChecked(false);
}

void QImageROISelector::on_excludedMinus_clicked()
{
    ui->selectionArea->decrementShapeSet(true);
}


void QImageROISelector::on_resetView_clicked()
{
    ui->selectionArea->resetView();
}

void QImageROISelector::on_selectionAreaMouseMove()
{
    ui->imageCoords->setText(QString("X = %1, Y = %2").arg(ui->selectionArea->getCurrentImageX()).arg(ui->selectionArea->getCurrentImageY()));
}

void QImageROISelector::on_resetShapes_clicked()
{
    ui->selectionArea->clearShapesSet();
}

void QImageROISelector::on_zoomIn_clicked()
{
    // return if there is no image
    if(!ui->selectionArea->activeImage()) return;

    ui->selectionArea->zoom(false);
}

void QImageROISelector::on_zoomOut_clicked()
{
    if(!ui->selectionArea->activeImage())return;
    ui->selectionArea->zoom(true);
}
