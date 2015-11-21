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

void
apply_transform(Teuchos::ArrayRCP<intensity_t> & intensities_from,
  Teuchos::ArrayRCP<intensity_t> & intensities_to,
  const int_t cx,
  const int_t cy,
  const int_t width,
  const int_t height,
  Teuchos::RCP<const std::vector<scalar_t> > deformation){
  assert(deformation!=Teuchos::null);
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

  scalar_t dx=0.0, dy=0.0;
  scalar_t dX=0.0, dY=0.0;
  scalar_t Dx=0.0, Dy=0.0;
  scalar_t mapped_x=0.0, mapped_y=0.0;
  scalar_t dx2=0.0, dx3=0.0;
  scalar_t dy2=0.0, dy3=0.0;
  scalar_t f0x=0.0, f0y=0.0;
  int_t x1=0,x2=0,y1=0,y2=0;
  intensity_t intensity_value = 0.0;
  for(int_t y=0;y<height;++y){
    for(int_t x=0;x<width;++x){

      dX = x - CX;
      dY = y - CY;
      Dx = (1.0-ex)*dX - g*dY;
      Dy = (1.0-ey)*dY - g*dX;
      mapped_x = cos_t*Dx - sin_t*Dy - u + CX;
      mapped_y = sin_t*Dx + cos_t*Dy - v + CY;
      // determine the current pixel the coordinates fall in:
      x1 = (int_t)mapped_x;
      if(mapped_x - x1 >= 0.5) x1++;
      y1 = (int_t)mapped_y;
      if(mapped_y - y1 >= 0.5) y1++;

      // check that the mapped location is inside the image...
      if(mapped_x>2.5&&mapped_x<width-3.5&&mapped_y>2.5&&mapped_y<height-3.5){
        intensity_value = 0.0;
        // convolve all the pixels within + and - pixels of the point in question
        for(int_t j=y1-3;j<=y1+3;++j){
          dy = std::abs(mapped_y - j);
          dy2=dy*dy;
          dy3=dy2*dy;
          f0y = 0.0;
          if(dy <= 1.0){
            f0y = 4.0/3.0*dy3 - 7.0/3.0*dy2 + 1.0;
          }
          else if(dy <= 2.0){
            f0y = -7.0/12.0*dy3 + 3.0*dy2 - 59.0/12.0*dy + 15.0/6.0;
          }
          else if(dy <= 3.0){
            f0y = 1.0/12.0*dy3 - 2.0/3.0*dy2 + 21.0/12.0*dy - 3.0/2.0;
          }
          for(int_t i=x1-3;i<=x1+3;++i){
            // compute the f's of x and y
            dx = std::abs(mapped_x - i);
            dx2=dx*dx;
            dx3=dx2*dx;
            f0x = 0.0;
            if(dx <= 1.0){
              f0x = 4.0/3.0*dx3 - 7.0/3.0*dx2 + 1.0;
            }
            else if(dx <= 2.0){
              f0x = -7.0/12.0*dx3 + 3.0*dx2 - 59.0/12.0*dx + 15.0/6.0;
            }
            else if(dx <= 3.0){
              f0x = 1.0/12.0*dx3 - 2.0/3.0*dx2 + 21.0/12.0*dx - 3.0/2.0;
            }
            intensity_value += intensities_from[j*width+i]*f0x*f0y;
          }
        }
        intensities_to[y*width+x] = intensity_value;
      }
      else if(mapped_x>=0&&mapped_x<width-1.5&&mapped_y>=0&&mapped_y<height-1.5){
        x1 = (int_t)mapped_x;
        x2 = x1+1;
        y1 = (int_t)mapped_y;
        y2  = y1+1;
        intensities_to[y*width+x] =
            (intensities_from[y1*width+x1]*(x2-mapped_x)*(y2-mapped_y)
                +intensities_from[y1*width+x2]*(mapped_x-x1)*(y2-mapped_y)
                +intensities_from[y2*width+x2]*(mapped_x-x1)*(mapped_y-y1)
                +intensities_from[y2*width+x1]*(x2-mapped_x)*(mapped_y-y1));
      }
      else{
        intensities_to[y*width+x] = 0.0;
      }
    }// x
  }// y
}

}// End DICe Namespace
