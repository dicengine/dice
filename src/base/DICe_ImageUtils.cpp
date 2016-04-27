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

#include <DICe_ImageUtils.h>
#include <DICe_Image.h>

namespace DICe {

void apply_transform(Teuchos::RCP<Image> image_in,
  Teuchos::RCP<Image> image_out,
  const int_t cx,
  const int_t cy,
  Teuchos::RCP<const std::vector<scalar_t> > deformation){
  const int_t width = image_in->width();
  const int_t height = image_in->height();
  TEUCHOS_TEST_FOR_EXCEPTION(width!=image_out->width(),std::runtime_error,"Dimensions must be the same");
  TEUCHOS_TEST_FOR_EXCEPTION(height!=image_out->height(),std::runtime_error,"Dimensions must be the same");
  TEUCHOS_TEST_FOR_EXCEPTION(deformation==Teuchos::null,std::runtime_error,"");
  const scalar_t u = (*deformation)[DISPLACEMENT_X];
  const scalar_t v = (*deformation)[DISPLACEMENT_Y];
  const scalar_t t = (*deformation)[ROTATION_Z];
  const scalar_t ex = (*deformation)[NORMAL_STRAIN_X];
  const scalar_t ey = (*deformation)[NORMAL_STRAIN_Y];
  const scalar_t g = (*deformation)[SHEAR_STRAIN_XY];
  const scalar_t cos_t = std::cos(t);
  const scalar_t sin_t = std::sin(t);
  const scalar_t CX = cx + u;
  const scalar_t CY = cy + v;
  scalar_t dX=0.0, dY=0.0;
  scalar_t Dx=0.0, Dy=0.0;
  scalar_t mapped_x=0.0, mapped_y=0.0;
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x){
      dX = x - CX;
      dY = y - CY;
      Dx = (1.0-ex)*dX - g*dY;
      Dy = (1.0-ey)*dY - g*dX;
      mapped_x = cos_t*Dx - sin_t*Dy - u + CX;
      mapped_y = sin_t*Dx + cos_t*Dy - v + CY;
      image_out->intensities()[y*width+x] = image_in->interpolate_keys_fourth(mapped_x,mapped_y);
    }// x
  }// y
}

}// End DICe Namespace
