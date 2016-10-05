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

void SinCos_Image_Deformer::compute_deformation(const scalar_t & coord_x,
  const scalar_t & coord_y,
  scalar_t & bx,
  scalar_t & by){
  // pattern repeats every 500 pixels
  // TODO should this depend on the image dims?
  const scalar_t L = 500.0;
  bx = 0.0;
  by = 0.0;
  if(use_superposition_){
    for(int_t i=0;i<freq_step_;++i){
      const scalar_t beta = (i+1)*DICE_PI/L;
      bx += sin(beta*coord_x)*cos(beta*coord_y)*0.5/(i+1);
      by += -cos(beta*coord_x)*sin(beta*coord_y)*0.5/(i+1);
    }
  }else{
    const scalar_t beta = (freq_step_+1)*DICE_PI/L;
    bx = sin(beta*coord_x)*cos(beta*coord_y)*0.25*(mag_step_+1);
    by = -cos(beta*coord_x)*sin(beta*coord_y)*0.25*(mag_step_+1);
  }
}

void SinCos_Image_Deformer::compute_deriv_deformation(const scalar_t & coord_x,
  const scalar_t & coord_y,
  scalar_t & bxx,
  scalar_t & bxy,
  scalar_t & byx,
  scalar_t & byy){
  // pattern repeats every 500 pixels
  // TODO should this depend on the image dims?
  const scalar_t L = 500.0;
  bxx = 0.0;
  bxy = 0.0;
  byx = 0.0;
  byy = 0.0;
  if(use_superposition_){
    for(int_t i=0;i<freq_step_;++i){
      const scalar_t beta = (i+1)*DICE_PI/L;
      bxx += beta*cos(beta*coord_x)*cos(beta*coord_y)*0.5/(i+1);
      bxy += -beta*sin(beta*coord_x)*sin(beta*coord_y)*0.5/(i+1);
      byx += beta*sin(beta*coord_x)*sin(beta*coord_y)*0.5/(i+1);
      byy += -beta*cos(beta*coord_x)*cos(beta*coord_y)*0.5/(i+1);
    }
  }
  else{
    const scalar_t beta = (freq_step_+1)*DICE_PI/L;
    bxx = beta*cos(beta*coord_x)*cos(beta*coord_y)*0.25*(mag_step_+1);
    bxy = -beta*sin(beta*coord_x)*sin(beta*coord_y)*0.25*(mag_step_+1);
    byx = beta*sin(beta*coord_x)*sin(beta*coord_y)*0.25*(mag_step_+1);
    byy = -beta*cos(beta*coord_x)*cos(beta*coord_y)*0.25*(mag_step_+1);
  }
}

void SinCos_Image_Deformer::compute_displacement_error(const scalar_t & coord_x,
  const scalar_t & coord_y,
  const scalar_t & sol_x,
  const scalar_t & sol_y,
  scalar_t & error_x,
  scalar_t & error_y,
  const bool use_mag,
  const bool relative){

  scalar_t out_x = 0.0;
  scalar_t out_y = 0.0;
  compute_deformation(coord_x,coord_y,out_x,out_y);
  if(use_mag){
    error_x = (sol_x - out_x)*(sol_x - out_x);
    error_y = (sol_y - out_y)*(sol_y - out_y);
  }else{
    error_x = sol_x - out_x;
    error_y = sol_y - out_y;
  }
  if(relative){
    error_x /= 0.25*(mag_step_+1);
    error_y /= 0.25*(mag_step_+1);
  }
}

void SinCos_Image_Deformer::compute_lagrange_strain_error(const scalar_t & coord_x,
  const scalar_t & coord_y,
  const scalar_t & sol_xx,
  const scalar_t & sol_xy,
  const scalar_t & sol_yy,
  scalar_t & error_xx,
  scalar_t & error_xy,
  scalar_t & error_yy,
  const bool use_mag,
  const bool relative){

  scalar_t out_xx = 0.0;
  scalar_t out_xy = 0.0;
  scalar_t out_yx = 0.0;
  scalar_t out_yy = 0.0;
  compute_deriv_deformation(coord_x,coord_y,out_xx,out_xy,out_yx,out_yy);

  const scalar_t strain_xx = 0.5*(2.0*out_xx + out_xx*out_xx + out_yx*out_yx);
  const scalar_t strain_xy = 0.5*(out_xy + out_yx + out_xy*out_xy + out_yx*out_yx);
  const scalar_t strain_yy = 0.5*(2.0*out_yy + out_yy*out_yy + out_xy*out_xy);

  if(use_mag){
    error_xx = (sol_xx - strain_xx)*(sol_xx - strain_xx);
    error_xy = (sol_xy - strain_xy)*(sol_xy - strain_xy);
    error_yy = (sol_yy - strain_yy)*(sol_yy - strain_yy);
  }else{
    error_xx = sol_xx - strain_xx;
    error_xy = sol_xy - strain_xy;
    error_yy = sol_yy - strain_yy;
  }
  if(relative){
    scalar_t rel = (freq_step_+1)*DICE_PI/500.0*(0.25*(mag_step_+1));
    error_xx /= rel;
    error_xy /= rel;
    error_yy /= rel;
  }
}

Teuchos::RCP<Image>
SinCos_Image_Deformer::deform_image(Teuchos::RCP<Image> ref_image){
  const int_t w = ref_image->width();
  const int_t h = ref_image->height();
//  // Note: uses 5 x 5 point sampling grid to evaluate the deformed intensity
//  const int_t num_pts = 5;
//  static scalar_t coeffs[5] = {0.0014,0.1574,0.62825,0.1574,0.0014};
//  // Note: uses 11 x 11 point sampling grid to evaluate the deformed intensity
////  static scalar_t coeffs[11] =
////  {0.0001,0.0017,0.0168,0.0870,
////    0.2328,0.3231,0.2328,
////    0.0870,0.0168,0.0017,0.0001};
//  Teuchos::ArrayRCP<intensity_t> def_intens(w*h,0.0);
//  scalar_t bx=0.0,by=0.0;
//  for(int_t j=0;j<h;++j){
//    for(int_t i=0;i<w;++i){
//      scalar_t avg_intens = 0.0;
//      for(int_t oy=0;oy<num_pts;++oy){
//        const scalar_t sample_y = j - 0.5*(num_pts-1)/num_pts + oy/num_pts;
//        for(int_t ox=0;ox<num_pts;++ox){
//          const scalar_t sample_x = i - 0.5*(num_pts-1)/num_pts + ox/num_pts;
//          const scalar_t weight = coeffs[ox]*coeffs[oy];
//          compute_deformation(sample_x,sample_y,bx,by);
//          scalar_t intens = ref_image->interpolate_keys_fourth(sample_x-bx,sample_y-by);
//          avg_intens += weight*intens;
//        } // end super pixel ox
//      } // end super pixel oy
//      def_intens[j*w+i] = avg_intens;
//    } // end pixel i
//  } // ens pixel j


  // Note: uses 5 x 5 point sampling grid to evaluate the deformed intensity
  const int_t num_pts = 5;
  static scalar_t offsets_x[5] = {0.0,-0.5,0.5,0.5,-0.5};
  static scalar_t offsets_y[5] = {0.0,-0.5,-0.5,0.5,0.5};
  Teuchos::ArrayRCP<intensity_t> def_intens(w*h,0.0);
  scalar_t bx=0.0,by=0.0;
  for(int_t j=0;j<h;++j){
    for(int_t i=0;i<w;++i){
      scalar_t avg_intens = 0.0;
      for(int_t pt=0;pt<num_pts;++pt){
        const scalar_t sample_x = i - offsets_x[pt];
        const scalar_t sample_y = j - offsets_y[pt];
        compute_deformation(sample_x,sample_y,bx,by);
        scalar_t intens = ref_image->interpolate_keys_fourth(sample_x-bx,sample_y-by);
        avg_intens += intens;
      } // end avg points
      def_intens[j*w+i] = avg_intens/num_pts;
    } // end pixel i
  } // ens pixel j

  // no weighted average ...
//  Teuchos::ArrayRCP<intensity_t> def_intens(w*h,0.0);
//  scalar_t bx=0.0,by=0.0;
//  for(int_t j=0;j<h;++j){
//    for(int_t i=0;i<w;++i){
//      compute_deformation(i,j,bx,by);
//      scalar_t intens = ref_image->interpolate_keys_fourth(i-bx,j-by);
//      def_intens[j*w+i] = intens;
//    } // end pixel i
//  } // ens pixel j
  Teuchos::RCP<Image> def_img = Teuchos::rcp(new Image(w,h,def_intens));
  return def_img;
}


}// End DICe Namespace
