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
}

QImageROISelector::~QImageROISelector()
{
    delete ui;
}

bool QImageROISelector::addShapesEnabled(){
    return ui->boundaryPlus->isChecked();
}

bool QImageROISelector::isInSelectionArea(int x, int y)
{
    int offset_x = this->x() + ui->selectionArea->x();
    int offset_y = this->y() + ui->selectionArea->y();
    int limit_x = offset_x + ui->selectionArea->get_image_width();
    int limit_y = offset_y + ui->selectionArea->get_image_height();
    if(x>offset_x && x<limit_x && y>offset_y && y<limit_y)
        return true;
    else
        return false;
}

void QImageROISelector::setImage(QFileInfo & file)
{
    ui->imageFileLabel->setText(file.filePath());
    ui->selectionArea->openImage(file.filePath());
}

void QImageROISelector::drawShapeLine(QPoint &pt)
{
    ui->selectionArea->drawShapeLine(pt);
}

void QImageROISelector::on_boundaryMinus_clicked()
{
    std::cout << "beginning " << std::endl;
    DICe::gui::Input_Vars::instance()->display_roi_vertices();

    // if a shape is in progress, just clear the currnet shape
    bool shape_in_progress = !ui->selectionArea->is_first_point();

    // reset the image
    ui->selectionArea->resetImage();

    // reset the points
    ui->selectionArea->resetLocation();
    ui->selectionArea->clear_current_roi_vertices();

    if(!shape_in_progress){
        // remove the last shape from the set
        DICe::gui::Input_Vars::instance()->decrement_vertex_vector();
    }

    // redraw the other shapes
    for(QList<QList<QPoint> >::iterator it=DICe::gui::Input_Vars::instance()->get_roi_vertex_vectors()->begin();
        it!=DICe::gui::Input_Vars::instance()->get_roi_vertex_vectors()->end();++it){
        ui->selectionArea->drawShape(*it);
    }

    std::cout << "end " << std::endl;
    DICe::gui::Input_Vars::instance()->display_roi_vertices();
}
