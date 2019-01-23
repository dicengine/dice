// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#include <DICe_Image.h>
#include <DICe_ImageUtils.h>
#include <DICe_LocalShapeFunction.h>
#include <DICe_ImageIO.h>
#include <DICe_Shape.h>

#include <cassert>

namespace DICe {

inline scalar_t keys_f0(const scalar_t & s){
  return 1.33333333333333*s*s*s - 2.33333333333333*s*s+ 1.0;
}
inline scalar_t keys_f1(const scalar_t & s){
  return -0.58333333333333*s*s*s + 3.0*s*s - 4.91666666666666*s + 2.5;
}
inline scalar_t keys_f2(const scalar_t & s){
  return 0.08333333333333*s*s*s - 0.66666666666666*s*s + 1.75*s - 1.5;
}

Image::Image(const char * file_name,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  offset_x_(0),
  offset_y_(0),
  intensity_rcp_(Teuchos::null),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_(file_name),
  has_file_name_(true),
  gradient_method_(FINITE_DIFFERENCE)
{
  bool filter_failed = false;
  bool convert_to_8_bit = true;
  if(params!=Teuchos::null){
    filter_failed = params->get<bool>(DICe::filter_failed_cine_pixels,false);
    convert_to_8_bit = params->get<bool>(DICe::convert_cine_to_8_bit,true);
  }
  try{
    utils::read_image_dimensions(file_name,width_,height_);
    TEUCHOS_TEST_FOR_EXCEPTION(width_<=0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(height_<=0,std::runtime_error,"");
    intensities_ = Teuchos::ArrayRCP<intensity_t>(width_*height_,0.0);
    utils::read_image(file_name,intensities_.getRawPtr(),true,convert_to_8_bit,filter_failed);
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, image file read failure");
  }
  // copy the image to the device (no-op for OpenMP, or serial)
  default_constructor_tasks(params);
}

Image::Image(const char * file_name,
  const int_t offset_x,
  const int_t offset_y,
  const int_t width,
  const int_t height,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  width_(width),
  height_(height),
  offset_x_(offset_x),
  offset_y_(offset_y),
  intensity_rcp_(Teuchos::null),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_(file_name),
  has_file_name_(true),
  gradient_method_(FINITE_DIFFERENCE)
{
  bool filter_failed = false;
  bool convert_to_8_bit = true;
  if(params!=Teuchos::null){
    filter_failed = params->get<bool>(DICe::filter_failed_cine_pixels,false);
    convert_to_8_bit = params->get<bool>(DICe::convert_cine_to_8_bit,true);
  }
  // get the image dims
  int_t img_width = 0;
  int_t img_height = 0;
  try{
    utils::read_image_dimensions(file_name,img_width,img_height);
    TEUCHOS_TEST_FOR_EXCEPTION(width_<=0||offset_x_+width_>img_width,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(height_<=0||offset_y_+height_>img_height,std::runtime_error,"");
    // initialize the pixel containers
    intensities_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,0.0);
    // read in the image
    utils::read_image(file_name,
      offset_x,offset_y,
      width_,height_,
      intensities_.getRawPtr(),true,convert_to_8_bit,filter_failed);
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, image file read failure");
  }
  default_constructor_tasks(params);
}

Image::Image(const int_t width,
  const int_t height,
  const intensity_t intensity,
  const int_t offset_x,
  const int_t offset_y):
  width_(width),
  height_(height),
  offset_x_(offset_x),
  offset_y_(offset_y),
  intensity_rcp_(Teuchos::null),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_("(from array)"),
  has_file_name_(false),
  gradient_method_(FINITE_DIFFERENCE)
{
  assert(height_>0);
  assert(width_>0);
  intensities_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,intensity);
  default_constructor_tasks(Teuchos::null);
}

Image::Image(Teuchos::RCP<Image> img,
  const int_t offset_x,
  const int_t offset_y,
  const int_t width,
  const int_t height,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  width_(width),
  height_(height),
  offset_x_(offset_x),
  offset_y_(offset_y),
  intensity_rcp_(Teuchos::null),
  has_gradients_(img->has_gradients()),
  has_gauss_filter_(img->has_gauss_filter()),
  file_name_(img->file_name()),
  has_file_name_(img->has_file_name()),
  gradient_method_(FINITE_DIFFERENCE)
{
  TEUCHOS_TEST_FOR_EXCEPTION(offset_x_<0,std::invalid_argument,"Error, offset_x_ cannot be negative.");
  TEUCHOS_TEST_FOR_EXCEPTION(offset_y_<0,std::invalid_argument,"Error, offset_x_ cannot be negative.");
  if(width_==-1)
    width_ = img->width();
  if(height_==-1)
    height_ = img->height();
  assert(width_>0);
  assert(height_>0);
  const int_t src_width = img->width();
  const int_t src_height = img->height();

  // initialize the pixel containers
  intensities_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,0.0);
  intensities_temp_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,0.0);
  grad_x_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  grad_y_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  mask_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  // deep copy values over
  int_t src_y=0, src_x=0;
  for(int_t y=0;y<height_;++y){
    src_y = y + offset_y_;
    for(int_t x=0;x<width_;++x){
      src_x = x + offset_x_;
      if(src_x>=0&&src_x<src_width&&src_y>=0&&src_y<src_height){
        intensities_[y*width_+x] = (*img)(src_x,src_y);
        grad_x_[y*width_+x] = img->grad_x(src_x,src_y);
        grad_y_[y*width_+x] = img->grad_y(src_x,src_y);
        mask_[y*width_+x] = img->mask(src_x,src_y);
      }
      else{
        intensities_[y*width_+x] = 0.0;
        grad_x_[y*width_+x] = 0.0;
        grad_y_[y*width_+x] = 0.0;
        mask_[y*width_+x] = 1.0;
      }
    }
  }
  grad_c1_ = 1.0/12.0;
  grad_c2_ = -8.0/12.0;
  gauss_filter_mask_size_ = img->gauss_filter_mask_size();
  gauss_filter_half_mask_ = gauss_filter_mask_size_/2+1;
  if(params!=Teuchos::null){
    if(params->isParameter(DICe::gauss_filter_mask_size)){
      gauss_filter_mask_size_ = params->get<int>(DICe::gauss_filter_mask_size,7);
      gauss_filter_half_mask_ = gauss_filter_mask_size_/2+1;
    }
    if(params->isParameter(DICe::gauss_filter_images)){
        if(!img->has_gauss_filter()&&params->get<bool>(DICe::gauss_filter_images,false)){
          // if the image was not filtered, but requested here, filter the image
          DEBUG_MSG("Image filter requested, but origin image does not have gauss filter, applying gauss filter here");
          gauss_filter();
        }
    }
    if(params->isParameter(DICe::compute_image_gradients)){
      if(params->isParameter(DICe::gradient_method)){
        gradient_method_ = params->get<Gradient_Method>(DICe::gradient_method);
      }
      if(!img->has_gradients()&&params->get<bool>(DICe::compute_image_gradients,false)){
        // if gradients have not been computed, but ther are requested here, compute them
        DEBUG_MSG("Image gradients requested, but origin image does not have gradients, computing them here");
        compute_gradients();
      }
    }
  }
}

void
Image::initialize_array_image(intensity_t * intensities){
  assert(width_>0);
  assert(height_>0);
  intensities_ = Teuchos::ArrayRCP<intensity_t>(intensities,0,width_*height_,false);
}

void
Image::default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){
  grad_x_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  grad_y_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  intensities_temp_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,0.0);
  mask_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  if(params!=Teuchos::null){
    if(params->isParameter(DICe::compute_laplacian_image)){
      if(params->get<bool>(DICe::compute_laplacian_image)==true){
        laplacian_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
      }
    }
  }
  // image gradient coefficients
  grad_c1_ = 1.0/12.0;
  grad_c2_ = -8.0/12.0;
  post_allocation_tasks(params);
}

const intensity_t&
Image::operator()(const int_t x, const int_t y) const {
  TEUCHOS_TEST_FOR_EXCEPTION(x<0||x>=width_,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(y<0||y>=height_,std::runtime_error,"");
  return intensities_[y*width_+x];
}

const intensity_t&
Image::operator()(const int_t i) const {
  return intensities_[i];
}

const scalar_t&
Image::grad_x(const int_t x,
  const int_t y) const {
  return grad_x_[y*width_+x];
}

Teuchos::ArrayRCP<scalar_t>
Image::grad_x_array() const {
  return grad_x_;
}

Teuchos::ArrayRCP<scalar_t>
Image::grad_y_array() const {
  return grad_y_;
}

const scalar_t&
Image::grad_y(const int_t x,
  const int_t y) const {
  return grad_y_[y*width_+x];
}

const scalar_t&
Image::mask(const int_t x,
  const int_t y) const {
  return mask_[y*width_+x];
}

const scalar_t&
Image::laplacian(const int_t x,
  const int_t y) const {
  return laplacian_[y*width_+x];
}

Teuchos::ArrayRCP<intensity_t>
Image::intensities()const{
  return intensities_;
}

intensity_t
Image::interpolate_bilinear(const scalar_t & local_x, const scalar_t & local_y){

  if(local_x<0.0||local_x>=width_-1.5||local_y<0.0||local_y>=height_-1.5) return 0.0;
  const int_t x1 = (int_t)local_x;
  const int_t x2 = x1+1;
  const int_t y1 = (int_t)local_y;
  const int_t y2  = y1+1;
  return intensities_[y1*width_+x1]*(x2-local_x)*(y2-local_y)
      +intensities_[y1*width_+x2]*(local_x-x1)*(y2-local_y)
      +intensities_[y2*width_+x2]*(local_x-x1)*(local_y-y1)
      +intensities_[y2*width_+x1]*(x2-local_x)*(local_y-y1);
}

scalar_t
Image::interpolate_grad_x_bilinear(const scalar_t & local_x, const scalar_t & local_y){

  if(local_x<0.0||local_x>=width_-1.5||local_y<0.0||local_y>=height_-1.5) return 0.0;
  const int_t x1 = (int_t)local_x;
  const int_t x2 = x1+1;
  const int_t y1 = (int_t)local_y;
  const int_t y2  = y1+1;
  return grad_x_[y1*width_+x1]*(x2-local_x)*(y2-local_y)
      +grad_x_[y1*width_+x2]*(local_x-x1)*(y2-local_y)
      +grad_x_[y2*width_+x2]*(local_x-x1)*(local_y-y1)
      +grad_x_[y2*width_+x1]*(x2-local_x)*(local_y-y1);
}

scalar_t
Image::interpolate_grad_y_bilinear(const scalar_t & local_x, const scalar_t & local_y){

  if(local_x<0.0||local_x>=width_-1.5||local_y<0.0||local_y>=height_-1.5) return 0.0;
  const int_t x1 = (int_t)local_x;
  const int_t x2 = x1+1;
  const int_t y1 = (int_t)local_y;
  const int_t y2  = y1+1;
  return grad_y_[y1*width_+x1]*(x2-local_x)*(y2-local_y)
      +grad_y_[y1*width_+x2]*(local_x-x1)*(y2-local_y)
      +grad_y_[y2*width_+x2]*(local_x-x1)*(local_y-y1)
      +grad_y_[y2*width_+x1]*(x2-local_x)*(local_y-y1);
}

intensity_t
Image::interpolate_bicubic(const scalar_t & local_x, const scalar_t & local_y){
  if(local_x<1.0||local_x>=width_-2.0||local_y<1.0||local_y>=height_-2.0) return this->interpolate_bilinear(local_x,local_y);

  const int_t x0  = (int_t)local_x;
  const int_t x1  = x0+1;
  const int_t x2  = x1+1;
  const int_t xm1 = x0-1;
  const int_t y0  = (int_t)local_y;
  const int_t y1 = y0+1;
  const int_t y2 = y1+1;
  const int_t ym1 = y0-1;
  const scalar_t x = local_x - x0;
  const scalar_t y = local_y - y0;
  const scalar_t x_2 = x * x;
  const scalar_t x_3 = x_2 * x;
  const scalar_t y_2 = y * y;
  const scalar_t y_3 = y_2 * y;
  const intensity_t fm10  = intensities_[y0*width_+xm1];
  const intensity_t f00   = intensities_[y0*width_+x0];
  const intensity_t f10   = intensities_[y0*width_+x1];
  const intensity_t f20   = intensities_[y0*width_+x2];
  const intensity_t fm11  = intensities_[y1*width_+xm1];
  const intensity_t f01   = intensities_[y1*width_+x0];
  const intensity_t f11   = intensities_[y1*width_+x1];
  const intensity_t f21   = intensities_[y1*width_+x2];
  const intensity_t fm12  = intensities_[y2*width_+xm1];
  const intensity_t f02   = intensities_[y2*width_+x0];
  const intensity_t f12   = intensities_[y2*width_+x1];
  const intensity_t f22   = intensities_[y2*width_+x2];
  const intensity_t fm1m1 = intensities_[ym1*width_+xm1];
  const intensity_t f0m1  = intensities_[ym1*width_+x0];
  const intensity_t f1m1  = intensities_[ym1*width_+x1];
  const intensity_t f2m1  = intensities_[ym1*width_+x2];
#ifdef DICE_USE_DOUBLE
  return f00 + (-0.5*f0m1 + .5*f01)*y + (f0m1 - 2.5*f00 + 2*f01 - .5*f02)*y_2 + (-0.5*f0m1 + 1.5*f00 - 1.5*f01 + .5*f02)*y_3
      + ((-0.5*fm10 + .5*f10) + (0.25*fm1m1 - .25*fm11 - .25*f1m1 + .25*f11)*y + (-0.5*fm1m1 + 1.25*fm10 - fm11 + .25*fm12 +
          0.5*f1m1 - 1.25*f10 + f11 - .25*f12)*y_2 + (0.25*fm1m1 - .75*fm10 + .75*fm11 - .25*fm12 - .25*f1m1 + .75*f10 - .75*f11 + .25*f12)*y_3) * x
      +((fm10 - 2.5*f00 + 2*f10 - .5*f20) + (-0.5*fm1m1 + .5*fm11 + 1.25*f0m1 - 1.25*f01 - f1m1 + f11 + .25*f2m1 - .25*f21)*y + (fm1m1 - 2.5*fm10 + 2*fm11
          - .5*fm12 - 2.5*f0m1 + 6.25*f00 - 5*f01 + 1.25*f02 + 2*f1m1 - 5*f10 + 4*f11 - f12 - .5*f2m1 + 1.25*f20 - f21 + .25*f22)*y_2 +
          (-0.5*fm1m1 + 1.5*fm10 - 1.5*fm11 + .5*fm12 + 1.25*f0m1 - 3.75*f00 + 3.75*f01 - 1.25*f02 - f1m1 + 3*f10 - 3*f11 + f12 + .25*f2m1 - .75*f20 + .75*f21 - .25*f22)*y_3)*x_2
      +((-.5*fm10 + 1.5*f00 - 1.5*f10 + .5*f20) + (0.25*fm1m1 - .25*fm11 - .75*f0m1 + .75*f01 + .75*f1m1 - .75*f11 - .25*f2m1 + .25*f21)*y +
          (-.5*fm1m1 + 1.25*fm10 - fm11 + .25*fm12 + 1.5*f0m1 - 3.75*f00 + 3*f01 - .75*f02 - 1.5*f1m1 + 3.75*f10 - 3*f11 + .75*f12 + .5*f2m1 - 1.25*f20 + f21 - .25*f22)*y_2
          + (0.25*fm1m1 - .75*fm10 + .75*fm11 - .25*fm12 - .75*f0m1 + 2.25*f00 - 2.25*f01 + .75*f02 + .75*f1m1 - 2.25*f10 + 2.25*f11 - .75*f12 - .25*f2m1 + .75*f20 - .75*f21 + .25*f22)*y_3)*x_3;
#else
  return f00 + (-0.5f*f0m1 + .5f*f01)*y + (f0m1 - 2.5f*f00 + 2.0f*f01 - .5f*f02)*y_2 + (-0.5f*f0m1 + 1.5f*f00 - 1.5f*f01 + .5f*f02)*y_3
      + ((-0.5f*fm10 + .5f*f10) + (0.25f*fm1m1 - .25f*fm11 - .25f*f1m1 + .25f*f11)*y + (-0.5f*fm1m1 + 1.25f*fm10 - fm11 + .25f*fm12 +
          0.5f*f1m1 - 1.25f*f10 + f11 - .25f*f12)*y_2 + (0.25f*fm1m1 - .75f*fm10 + .75f*fm11 - .25f*fm12 - .25f*f1m1 + .75f*f10 - .75f*f11 + .25f*f12)*y_3) * x
      +((fm10 - 2.5f*f00 + 2.0f*f10 - .5f*f20) + (-0.5f*fm1m1 + .5f*fm11 + 1.25f*f0m1 - 1.25f*f01 - f1m1 + f11 + .25f*f2m1 - .25f*f21)*y + (fm1m1 - 2.5f*fm10 + 2.0f*fm11
          - .5f*fm12 - 2.5f*f0m1 + 6.25f*f00 - 5.0f*f01 + 1.25f*f02 + 2.0f*f1m1 - 5.0f*f10 + 4.0f*f11 - f12 - .5f*f2m1 + 1.25f*f20 - f21 + .25f*f22)*y_2 +
          (-0.5f*fm1m1 + 1.5f*fm10 - 1.5f*fm11 + .5f*fm12 + 1.25f*f0m1 - 3.75f*f00 + 3.75f*f01 - 1.25f*f02 - f1m1 + 3.0f*f10 - 3.0f*f11 + f12 + .25f*f2m1 - .75f*f20 + .75f*f21 - .25f*f22)*y_3)*x_2
      +((-.5f*fm10 + 1.5f*f00 - 1.5f*f10 + .5f*f20) + (0.25f*fm1m1 - .25f*fm11 - .75f*f0m1 + .75f*f01 + .75f*f1m1 - .75f*f11 - .25f*f2m1 + .25f*f21)*y +
          (-.5f*fm1m1 + 1.25f*fm10 - fm11 + .25f*fm12 + 1.5f*f0m1 - 3.75f*f00 + 3.0f*f01 - .75f*f02 - 1.5f*f1m1 + 3.75f*f10 - 3.0f*f11 + .75f*f12 + .5f*f2m1 - 1.25f*f20 + f21 - .25f*f22)*y_2
          + (0.25f*fm1m1 - .75f*fm10 + .75f*fm11 - .25f*fm12 - .75f*f0m1 + 2.25f*f00 - 2.25f*f01 + .75f*f02 + .75f*f1m1 - 2.25f*f10 + 2.25f*f11 - .75f*f12 - .25f*f2m1 + .75f*f20 - .75f*f21 + .25f*f22)*y_3)*x_3;
#endif
}

scalar_t
Image::interpolate_grad_x_bicubic(const scalar_t & local_x, const scalar_t & local_y){
  if(local_x<1.0||local_x>=width_-2.0||local_y<1.0||local_y>=height_-2.0) return this->interpolate_grad_x_bilinear(local_x,local_y);

  const int_t x0  = (int_t)local_x;
  const int_t x1  = x0+1;
  const int_t x2  = x1+1;
  const int_t xm1 = x0-1;
  const int_t y0  = (int_t)local_y;
  const int_t y1 = y0+1;
  const int_t y2 = y1+1;
  const int_t ym1 = y0-1;
  const scalar_t x = local_x - x0;
  const scalar_t y = local_y - y0;
  const scalar_t x_2 = x * x;
  const scalar_t x_3 = x_2 * x;
  const scalar_t y_2 = y * y;
  const scalar_t y_3 = y_2 * y;
  const scalar_t fm10  = grad_x_[y0*width_+xm1];
  const scalar_t f00   = grad_x_[y0*width_+x0];
  const scalar_t f10   = grad_x_[y0*width_+x1];
  const scalar_t f20   = grad_x_[y0*width_+x2];
  const scalar_t fm11  = grad_x_[y1*width_+xm1];
  const scalar_t f01   = grad_x_[y1*width_+x0];
  const scalar_t f11   = grad_x_[y1*width_+x1];
  const scalar_t f21   = grad_x_[y1*width_+x2];
  const scalar_t fm12  = grad_x_[y2*width_+xm1];
  const scalar_t f02   = grad_x_[y2*width_+x0];
  const scalar_t f12   = grad_x_[y2*width_+x1];
  const scalar_t f22   = grad_x_[y2*width_+x2];
  const scalar_t fm1m1 = grad_x_[ym1*width_+xm1];
  const scalar_t f0m1  = grad_x_[ym1*width_+x0];
  const scalar_t f1m1  = grad_x_[ym1*width_+x1];
  const scalar_t f2m1  = grad_x_[ym1*width_+x2];
#ifdef DICE_USE_DOUBLE
  return f00 + (-0.5*f0m1 + .5*f01)*y + (f0m1 - 2.5*f00 + 2*f01 - .5*f02)*y_2 + (-0.5*f0m1 + 1.5*f00 - 1.5*f01 + .5*f02)*y_3
      + ((-0.5*fm10 + .5*f10) + (0.25*fm1m1 - .25*fm11 - .25*f1m1 + .25*f11)*y + (-0.5*fm1m1 + 1.25*fm10 - fm11 + .25*fm12 +
          0.5*f1m1 - 1.25*f10 + f11 - .25*f12)*y_2 + (0.25*fm1m1 - .75*fm10 + .75*fm11 - .25*fm12 - .25*f1m1 + .75*f10 - .75*f11 + .25*f12)*y_3) * x
      +((fm10 - 2.5*f00 + 2*f10 - .5*f20) + (-0.5*fm1m1 + .5*fm11 + 1.25*f0m1 - 1.25*f01 - f1m1 + f11 + .25*f2m1 - .25*f21)*y + (fm1m1 - 2.5*fm10 + 2*fm11
          - .5*fm12 - 2.5*f0m1 + 6.25*f00 - 5*f01 + 1.25*f02 + 2*f1m1 - 5*f10 + 4*f11 - f12 - .5*f2m1 + 1.25*f20 - f21 + .25*f22)*y_2 +
          (-0.5*fm1m1 + 1.5*fm10 - 1.5*fm11 + .5*fm12 + 1.25*f0m1 - 3.75*f00 + 3.75*f01 - 1.25*f02 - f1m1 + 3*f10 - 3*f11 + f12 + .25*f2m1 - .75*f20 + .75*f21 - .25*f22)*y_3)*x_2
      +((-.5*fm10 + 1.5*f00 - 1.5*f10 + .5*f20) + (0.25*fm1m1 - .25*fm11 - .75*f0m1 + .75*f01 + .75*f1m1 - .75*f11 - .25*f2m1 + .25*f21)*y +
          (-.5*fm1m1 + 1.25*fm10 - fm11 + .25*fm12 + 1.5*f0m1 - 3.75*f00 + 3*f01 - .75*f02 - 1.5*f1m1 + 3.75*f10 - 3*f11 + .75*f12 + .5*f2m1 - 1.25*f20 + f21 - .25*f22)*y_2
          + (0.25*fm1m1 - .75*fm10 + .75*fm11 - .25*fm12 - .75*f0m1 + 2.25*f00 - 2.25*f01 + .75*f02 + .75*f1m1 - 2.25*f10 + 2.25*f11 - .75*f12 - .25*f2m1 + .75*f20 - .75*f21 + .25*f22)*y_3)*x_3;
#else
  return f00 + (-0.5f*f0m1 + .5f*f01)*y + (f0m1 - 2.5f*f00 + 2.0f*f01 - .5f*f02)*y_2 + (-0.5f*f0m1 + 1.5f*f00 - 1.5f*f01 + .5f*f02)*y_3
      + ((-0.5f*fm10 + .5f*f10) + (0.25f*fm1m1 - .25f*fm11 - .25f*f1m1 + .25f*f11)*y + (-0.5f*fm1m1 + 1.25f*fm10 - fm11 + .25f*fm12 +
          0.5f*f1m1 - 1.25f*f10 + f11 - .25f*f12)*y_2 + (0.25f*fm1m1 - .75f*fm10 + .75f*fm11 - .25f*fm12 - .25f*f1m1 + .75f*f10 - .75f*f11 + .25f*f12)*y_3) * x
      +((fm10 - 2.5f*f00 + 2.0f*f10 - .5f*f20) + (-0.5f*fm1m1 + .5f*fm11 + 1.25f*f0m1 - 1.25f*f01 - f1m1 + f11 + .25f*f2m1 - .25f*f21)*y + (fm1m1 - 2.5f*fm10 + 2.0f*fm11
          - .5f*fm12 - 2.5f*f0m1 + 6.25f*f00 - 5.0f*f01 + 1.25f*f02 + 2.0f*f1m1 - 5.0f*f10 + 4.0f*f11 - f12 - .5f*f2m1 + 1.25f*f20 - f21 + .25f*f22)*y_2 +
          (-0.5f*fm1m1 + 1.5f*fm10 - 1.5f*fm11 + .5f*fm12 + 1.25f*f0m1 - 3.75f*f00 + 3.75f*f01 - 1.25f*f02 - f1m1 + 3.0f*f10 - 3.0f*f11 + f12 + .25f*f2m1 - .75f*f20 + .75f*f21 - .25f*f22)*y_3)*x_2
      +((-.5f*fm10 + 1.5f*f00 - 1.5f*f10 + .5f*f20) + (0.25f*fm1m1 - .25f*fm11 - .75f*f0m1 + .75f*f01 + .75f*f1m1 - .75f*f11 - .25f*f2m1 + .25f*f21)*y +
          (-.5f*fm1m1 + 1.25f*fm10 - fm11 + .25f*fm12 + 1.5f*f0m1 - 3.75f*f00 + 3.0f*f01 - .75f*f02 - 1.5f*f1m1 + 3.75f*f10 - 3.0f*f11 + .75f*f12 + .5f*f2m1 - 1.25f*f20 + f21 - .25f*f22)*y_2
          + (0.25f*fm1m1 - .75f*fm10 + .75f*fm11 - .25f*fm12 - .75f*f0m1 + 2.25f*f00 - 2.25f*f01 + .75f*f02 + .75f*f1m1 - 2.25f*f10 + 2.25f*f11 - .75f*f12 - .25f*f2m1 + .75f*f20 - .75f*f21 + .25f*f22)*y_3)*x_3;
#endif
}

scalar_t
Image::interpolate_grad_y_bicubic(const scalar_t & local_x, const scalar_t & local_y){
  if(local_x<1.0||local_x>=width_-2.0||local_y<1.0||local_y>=height_-2.0) return this->interpolate_grad_y_bilinear(local_x,local_y);

  const int_t x0  = (int_t)local_x;
  const int_t x1  = x0+1;
  const int_t x2  = x1+1;
  const int_t xm1 = x0-1;
  const int_t y0  = (int_t)local_y;
  const int_t y1 = y0+1;
  const int_t y2 = y1+1;
  const int_t ym1 = y0-1;
  const scalar_t x = local_x - x0;
  const scalar_t y = local_y - y0;
  const scalar_t x_2 = x * x;
  const scalar_t x_3 = x_2 * x;
  const scalar_t y_2 = y * y;
  const scalar_t y_3 = y_2 * y;
  const scalar_t fm10  = grad_y_[y0*width_+xm1];
  const scalar_t f00   = grad_y_[y0*width_+x0];
  const scalar_t f10   = grad_y_[y0*width_+x1];
  const scalar_t f20   = grad_y_[y0*width_+x2];
  const scalar_t fm11  = grad_y_[y1*width_+xm1];
  const scalar_t f01   = grad_y_[y1*width_+x0];
  const scalar_t f11   = grad_y_[y1*width_+x1];
  const scalar_t f21   = grad_y_[y1*width_+x2];
  const scalar_t fm12  = grad_y_[y2*width_+xm1];
  const scalar_t f02   = grad_y_[y2*width_+x0];
  const scalar_t f12   = grad_y_[y2*width_+x1];
  const scalar_t f22   = grad_y_[y2*width_+x2];
  const scalar_t fm1m1 = grad_y_[ym1*width_+xm1];
  const scalar_t f0m1  = grad_y_[ym1*width_+x0];
  const scalar_t f1m1  = grad_y_[ym1*width_+x1];
  const scalar_t f2m1  = grad_y_[ym1*width_+x2];
#ifdef DICE_USE_DOUBLE
  return f00 + (-0.5*f0m1 + .5*f01)*y + (f0m1 - 2.5*f00 + 2*f01 - .5*f02)*y_2 + (-0.5*f0m1 + 1.5*f00 - 1.5*f01 + .5*f02)*y_3
      + ((-0.5*fm10 + .5*f10) + (0.25*fm1m1 - .25*fm11 - .25*f1m1 + .25*f11)*y + (-0.5*fm1m1 + 1.25*fm10 - fm11 + .25*fm12 +
          0.5*f1m1 - 1.25*f10 + f11 - .25*f12)*y_2 + (0.25*fm1m1 - .75*fm10 + .75*fm11 - .25*fm12 - .25*f1m1 + .75*f10 - .75*f11 + .25*f12)*y_3) * x
      +((fm10 - 2.5*f00 + 2*f10 - .5*f20) + (-0.5*fm1m1 + .5*fm11 + 1.25*f0m1 - 1.25*f01 - f1m1 + f11 + .25*f2m1 - .25*f21)*y + (fm1m1 - 2.5*fm10 + 2*fm11
          - .5*fm12 - 2.5*f0m1 + 6.25*f00 - 5*f01 + 1.25*f02 + 2*f1m1 - 5*f10 + 4*f11 - f12 - .5*f2m1 + 1.25*f20 - f21 + .25*f22)*y_2 +
          (-0.5*fm1m1 + 1.5*fm10 - 1.5*fm11 + .5*fm12 + 1.25*f0m1 - 3.75*f00 + 3.75*f01 - 1.25*f02 - f1m1 + 3*f10 - 3*f11 + f12 + .25*f2m1 - .75*f20 + .75*f21 - .25*f22)*y_3)*x_2
      +((-.5*fm10 + 1.5*f00 - 1.5*f10 + .5*f20) + (0.25*fm1m1 - .25*fm11 - .75*f0m1 + .75*f01 + .75*f1m1 - .75*f11 - .25*f2m1 + .25*f21)*y +
          (-.5*fm1m1 + 1.25*fm10 - fm11 + .25*fm12 + 1.5*f0m1 - 3.75*f00 + 3*f01 - .75*f02 - 1.5*f1m1 + 3.75*f10 - 3*f11 + .75*f12 + .5*f2m1 - 1.25*f20 + f21 - .25*f22)*y_2
          + (0.25*fm1m1 - .75*fm10 + .75*fm11 - .25*fm12 - .75*f0m1 + 2.25*f00 - 2.25*f01 + .75*f02 + .75*f1m1 - 2.25*f10 + 2.25*f11 - .75*f12 - .25*f2m1 + .75*f20 - .75*f21 + .25*f22)*y_3)*x_3;
#else
  return f00 + (-0.5f*f0m1 + .5f*f01)*y + (f0m1 - 2.5f*f00 + 2.0f*f01 - .5f*f02)*y_2 + (-0.5f*f0m1 + 1.5f*f00 - 1.5f*f01 + .5f*f02)*y_3
      + ((-0.5f*fm10 + .5f*f10) + (0.25f*fm1m1 - .25f*fm11 - .25f*f1m1 + .25f*f11)*y + (-0.5f*fm1m1 + 1.25f*fm10 - fm11 + .25f*fm12 +
          0.5f*f1m1 - 1.25f*f10 + f11 - .25f*f12)*y_2 + (0.25f*fm1m1 - .75f*fm10 + .75f*fm11 - .25f*fm12 - .25f*f1m1 + .75f*f10 - .75f*f11 + .25f*f12)*y_3) * x
      +((fm10 - 2.5f*f00 + 2.0f*f10 - .5f*f20) + (-0.5f*fm1m1 + .5f*fm11 + 1.25f*f0m1 - 1.25f*f01 - f1m1 + f11 + .25f*f2m1 - .25f*f21)*y + (fm1m1 - 2.5f*fm10 + 2.0f*fm11
          - .5f*fm12 - 2.5f*f0m1 + 6.25f*f00 - 5.0f*f01 + 1.25f*f02 + 2.0f*f1m1 - 5.0f*f10 + 4.0f*f11 - f12 - .5f*f2m1 + 1.25f*f20 - f21 + .25f*f22)*y_2 +
          (-0.5f*fm1m1 + 1.5f*fm10 - 1.5f*fm11 + .5f*fm12 + 1.25f*f0m1 - 3.75f*f00 + 3.75f*f01 - 1.25f*f02 - f1m1 + 3.0f*f10 - 3.0f*f11 + f12 + .25f*f2m1 - .75f*f20 + .75f*f21 - .25f*f22)*y_3)*x_2
      +((-.5f*fm10 + 1.5f*f00 - 1.5f*f10 + .5f*f20) + (0.25f*fm1m1 - .25f*fm11 - .75f*f0m1 + .75f*f01 + .75f*f1m1 - .75f*f11 - .25f*f2m1 + .25f*f21)*y +
          (-.5f*fm1m1 + 1.25f*fm10 - fm11 + .25f*fm12 + 1.5f*f0m1 - 3.75f*f00 + 3.0f*f01 - .75f*f02 - 1.5f*f1m1 + 3.75f*f10 - 3.0f*f11 + .75f*f12 + .5f*f2m1 - 1.25f*f20 + f21 - .25f*f22)*y_2
          + (0.25f*fm1m1 - .75f*fm10 + .75f*fm11 - .25f*fm12 - .75f*f0m1 + 2.25f*f00 - 2.25f*f01 + .75f*f02 + .75f*f1m1 - 2.25f*f10 + 2.25f*f11 - .75f*f12 - .25f*f2m1 + .75f*f20 - .75f*f21 + .25f*f22)*y_3)*x_3;
#endif
}


intensity_t
Image::interpolate_keys_fourth(const scalar_t & local_x, const scalar_t & local_y){
  static std::vector<scalar_t> coeffs_x(6,0.0);
  static std::vector<scalar_t> coeffs_y(6,0.0);
  static scalar_t dx = 0.0;
  static scalar_t dy = 0.0;
  static int_t ix=0,iy=0;
  static intensity_t value=0.0;
  ix = (int_t)local_x;
  iy = (int_t)local_y;
  if(local_x<=2.5||local_x>=width_-3.5||local_y<=2.5||local_y>=height_-3.5)
    return this->interpolate_bilinear(local_x,local_y);
  dx = local_x - ix;
  dy = local_y - iy;
  coeffs_x[0] = keys_f2(dx+2.0);
  coeffs_x[1] = keys_f1(dx+1.0);
  coeffs_x[2] = keys_f0(dx);
  coeffs_x[3] = keys_f0(1.0-dx);
  coeffs_x[4] = keys_f1(2.0-dx);
  coeffs_x[5] = keys_f2(3.0-dx);
  coeffs_y[0] = keys_f2(dy+2.0);
  coeffs_y[1] = keys_f1(dy+1.0);
  coeffs_y[2] = keys_f0(dy);
  coeffs_y[3] = keys_f0(1.0-dy);
  coeffs_y[4] = keys_f1(2.0-dy);
  coeffs_y[5] = keys_f2(3.0-dy);
  value = 0.0;
  for(int_t m=0;m<6;++m){
    for(int_t n=0;n<6;++n){
      value += coeffs_y[m]*coeffs_x[n]*intensities_[(iy-2+m)*width_ + ix-2+n];
    }
  }
  return value;
}

scalar_t
Image::interpolate_grad_x_keys_fourth(const scalar_t & local_x, const scalar_t & local_y){
  static std::vector<scalar_t> coeffs_x(6,0.0);
  static std::vector<scalar_t> coeffs_y(6,0.0);
  static scalar_t dx = 0.0;
  static scalar_t dy = 0.0;
  static int_t ix=0,iy=0;
  static intensity_t value=0.0;
  ix = (int_t)local_x;
  iy = (int_t)local_y;
  if(local_x<=2.5||local_x>=width_-3.5||local_y<=2.5||local_y>=height_-3.5)
    return this->interpolate_grad_x_bilinear(local_x,local_y);
  dx = local_x - ix;
  dy = local_y - iy;
  coeffs_x[0] = keys_f2(dx+2.0);
  coeffs_x[1] = keys_f1(dx+1.0);
  coeffs_x[2] = keys_f0(dx);
  coeffs_x[3] = keys_f0(1.0-dx);
  coeffs_x[4] = keys_f1(2.0-dx);
  coeffs_x[5] = keys_f2(3.0-dx);
  coeffs_y[0] = keys_f2(dy+2.0);
  coeffs_y[1] = keys_f1(dy+1.0);
  coeffs_y[2] = keys_f0(dy);
  coeffs_y[3] = keys_f0(1.0-dy);
  coeffs_y[4] = keys_f1(2.0-dy);
  coeffs_y[5] = keys_f2(3.0-dy);
  value = 0.0;
  for(int_t m=0;m<6;++m){
    for(int_t n=0;n<6;++n){
      value += coeffs_y[m]*coeffs_x[n]*grad_x_[(iy-2+m)*width_ + ix-2+n];
    }
  }
  return value;
}

scalar_t
Image::interpolate_grad_y_keys_fourth(const scalar_t & local_x, const scalar_t & local_y){
  static std::vector<scalar_t> coeffs_x(6,0.0);
  static std::vector<scalar_t> coeffs_y(6,0.0);
  static scalar_t dx = 0.0;
  static scalar_t dy = 0.0;
  static int_t ix=0,iy=0;
  static intensity_t value=0.0;
  ix = (int_t)local_x;
  iy = (int_t)local_y;
  if(local_x<=2.5||local_x>=width_-3.5||local_y<=2.5||local_y>=height_-3.5)
    return this->interpolate_grad_y_bilinear(local_x,local_y);
  dx = local_x - ix;
  dy = local_y - iy;
  coeffs_x[0] = keys_f2(dx+2.0);
  coeffs_x[1] = keys_f1(dx+1.0);
  coeffs_x[2] = keys_f0(dx);
  coeffs_x[3] = keys_f0(1.0-dx);
  coeffs_x[4] = keys_f1(2.0-dx);
  coeffs_x[5] = keys_f2(3.0-dx);
  coeffs_y[0] = keys_f2(dy+2.0);
  coeffs_y[1] = keys_f1(dy+1.0);
  coeffs_y[2] = keys_f0(dy);
  coeffs_y[3] = keys_f0(1.0-dy);
  coeffs_y[4] = keys_f1(2.0-dy);
  coeffs_y[5] = keys_f2(3.0-dy);
  value = 0.0;
  for(int_t m=0;m<6;++m){
    for(int_t n=0;n<6;++n){
      value += coeffs_y[m]*coeffs_x[n]*grad_y_[(iy-2+m)*width_ + ix-2+n];
    }
  }
  return value;
}

void
Image::compute_gradients(const bool use_hierarchical_parallelism, const int_t team_size){
  if(gradient_method_==FINITE_DIFFERENCE){
    DEBUG_MSG("Image::compute_gradients(): using FINITE_DIFFERENCE");
    compute_gradients_finite_difference();
  }
  else if(gradient_method_==CONVOLUTION_5_POINT){
    DEBUG_MSG("Image::compute_gradients(): using CONVOLUTION_5_POINT");
    compute_gradients_finite_difference();
    smooth_gradients_convolution_5_point();
  }
  has_gradients_ = true;
}

void
Image::smooth_gradients_convolution_5_point(){

  static scalar_t smooth_coeffs[][5] = {{0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
                                        {0.015625,   0.0625,   0.09375,   0.0625,   0.015625},
                                        {0.0234375,  0.09375,  0.140625,  0.09375,  0.0234375},
                                        {0.015625,   0.0625,   0.09375,   0.0625,   0.015625},
                                        {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};
  static int_t smooth_offsets[] =  {-2, -1, 0, 1, 2};

  Teuchos::ArrayRCP<scalar_t> grad_x_temp(width_*height_,0.0);
  Teuchos::ArrayRCP<scalar_t> grad_y_temp(width_*height_,0.0);
  for(int_t i=0;i<width_*height_;++i){
    grad_x_temp[i] = grad_x_[i];
    grad_y_temp[i] = grad_y_[i];
  }
  for(int_t y=2;y<height_-2;++y){
    for(int_t x=2;x<width_-2;++x){
      scalar_t value_x = 0.0, value_y = 0.0;
      for(int_t i=0;i<5;++i){
        for(int_t j=0;j<5;++j){
          value_x += smooth_coeffs[i][j] * grad_x_temp[(y + smooth_offsets[i])*width_ + x + smooth_offsets[j]];
          value_y += smooth_coeffs[i][j] * grad_y_temp[(y + smooth_offsets[i])*width_ + x + smooth_offsets[j]];
        }
      }
      grad_x_[y*width_+x] = value_x;
      grad_y_[y*width_+x] = value_y;
    }
  }
}

void
Image::compute_gradients_finite_difference(){
  for(int_t y=0;y<height_;++y){
    for(int_t x=0;x<width_;++x){
      if(x<2){
        grad_x_[y*width_+x] = intensities_[y*width_+x+1] - intensities_[y*width_+x];
      }
      /// check if this pixel is near the right edge
      else if(x>=width_-2){
        grad_x_[y*width_+x] = intensities_[y*width_+x] - intensities_[y*width_+x-1];
      }
      else{
        grad_x_[y*width_+x] = grad_c1_*intensities_[y*width_+x-2] + grad_c2_*intensities_[y*width_+x-1]
            - grad_c2_*intensities_[y*width_+x+1] - grad_c1_*intensities_[y*width_+x+2];
      }
      /// check if this pixel is near the top edge
      if(y<2){
        grad_y_[y*width_+x] = intensities_[(y+1)*width_+x] - intensities_[y*width_+x];
      }
      /// check if this pixel is near the bottom edge
      else if(y>=height_-2){
        grad_y_[y*width_+x] = intensities_[y*width_+x] - intensities_[(y-1)*width_+x];
      }
      else{
        grad_y_[y*width_+x] = grad_c1_*intensities_[(y-2)*width_+x] + grad_c2_*intensities_[(y-1)*width_+x]
            - grad_c2_*intensities_[(y+1)*width_+x] - grad_c1_*intensities_[(y+2)*width_+x];
      }
    }
  }
}

void
Image::apply_mask(const Conformal_Area_Def & area_def,
  const bool smooth_edges){
  // first create the mask:
  create_mask(area_def,smooth_edges);
  for(int_t i=0;i<num_pixels();++i)
    intensities_[i] = mask_[i]*intensities_[i];
}

void
Image::apply_mask(const bool smooth_edges){
  if(smooth_edges){
    static scalar_t smoothing_coeffs[5][5];
    std::vector<scalar_t> coeffs(5,0.0);
    coeffs[0] = 0.0014;coeffs[1] = 0.1574;coeffs[2] = 0.62825;
    coeffs[3] = 0.1574;coeffs[4] = 0.0014;
    for(int_t j=0;j<5;++j){
      for(int_t i=0;i<5;++i){
        smoothing_coeffs[i][j] = coeffs[i]*coeffs[j];
      }
    }
    Teuchos::ArrayRCP<scalar_t> mask_tmp(mask_.size(),0.0);
    for(int_t i=0;i<mask_tmp.size();++i)
      mask_tmp[i] = mask_[i];
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        if(x>=2&&x<width_-2&&y>=2&&y<height_-2){ // 2 is half the gauss_mask size
          scalar_t value = 0.0;
          for(int_t i=0;i<5;++i){
            for(int_t j=0;j<5;++j){
              // assumes intensity values have already been deep copied into mask_tmp_
              value += smoothing_coeffs[i][j]*mask_tmp[(y+(j-2))*width_+x+(i-2)];
            } //j
          } //i
          mask_[y*width_+x] = value;
        }else{
          mask_[y*width_+x] = mask_tmp[y*width_+x];
        }
      } // x
    } // y
  } // smooth edges
  for(int_t i=0;i<num_pixels();++i)
    intensities_[i] = mask_[i]*intensities_[i];
}

void
Image::create_mask(const Conformal_Area_Def & area_def,
  const bool smooth_edges){
  assert(area_def.has_boundary());
  std::set<std::pair<int_t,int_t> > coords;
  for(size_t i=0;i<area_def.boundary()->size();++i){
    std::set<std::pair<int_t,int_t> > shapeCoords = (*area_def.boundary())[i]->get_owned_pixels();
    coords.insert(shapeCoords.begin(),shapeCoords.end());
  }
  // now remove any excluded regions:
  // now set the inactive bit for the second set of multishapes if they exist.
  if(area_def.has_excluded_area()){
    for(size_t i=0;i<area_def.excluded_area()->size();++i){
      std::set<std::pair<int_t,int_t> > removeCoords = (*area_def.excluded_area())[i]->get_owned_pixels();
      std::set<std::pair<int_t,int_t> >::iterator it = removeCoords.begin();
      for(;it!=removeCoords.end();++it){
        if(coords.find(*it)!=coords.end())
          coords.erase(*it);
      } // end removeCoords loop
    } // end excluded_area loop
  } // end has excluded area
  // NOTE: the pairs are (y,x) not (x,y) so that the ordering is correct in the set
  std::set<std::pair<int_t,int_t> >::iterator set_it = coords.begin();
  for( ; set_it!=coords.end();++set_it){
    mask_[(set_it->first - offset_y_)*width_+set_it->second - offset_x_] = 1.0;
  }
  if(smooth_edges){
    static scalar_t smoothing_coeffs[5][5];
    std::vector<scalar_t> coeffs(5,0.0);
    coeffs[0] = 0.0014;coeffs[1] = 0.1574;coeffs[2] = 0.62825;
    coeffs[3] = 0.1574;coeffs[4] = 0.0014;
    for(int_t j=0;j<5;++j){
      for(int_t i=0;i<5;++i){
        smoothing_coeffs[i][j] = coeffs[i]*coeffs[j];
      }
    }
    Teuchos::ArrayRCP<scalar_t> mask_tmp(mask_.size(),0.0);
    for(int_t i=0;i<mask_tmp.size();++i)
      mask_tmp[i] = mask_[i];
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        if(x>=2&&x<width_-2&&y>=2&&y<height_-2){ // 2 is half the gauss_mask size
          scalar_t value = 0.0;
          for(int_t i=0;i<5;++i){
            for(int_t j=0;j<5;++j){
              // assumes intensity values have already been deep copied into mask_tmp_
              value += smoothing_coeffs[i][j]*mask_tmp[(y+(j-2))*width_+x+(i-2)];
            } //j
          } //i
          mask_[y*width_+x] = value;
        }else{
          mask_[y*width_+x] = mask_tmp[y*width_+x];
        }
      } // x
    } // y
  }
}

Teuchos::RCP<Image>
Image::apply_transformation(Teuchos::RCP<Local_Shape_Function> shape_function,
  const int_t cx,
  const int_t cy,
  const bool apply_in_place){
  Teuchos::RCP<Image> this_img = Teuchos::rcp(this,false);
  if(apply_in_place){
    Teuchos::RCP<Image> temp_img = Teuchos::rcp(new Image(this_img));
    apply_transform(temp_img,this_img,cx,cy,shape_function);
    return Teuchos::null;
  }
  else{
    Teuchos::RCP<Image> result = Teuchos::rcp(new Image(width_,height_));
    apply_transform(this_img,result,cx,cy,shape_function);
    return result;
  }
}

void
Image::gauss_filter(const int_t mask_size,const bool use_hierarchical_parallelism,
  const int_t team_size){
  DEBUG_MSG("Image::gauss_filter: mask_size " << gauss_filter_mask_size_);

  if(mask_size>0){
    gauss_filter_mask_size_=mask_size;
    gauss_filter_half_mask_ = gauss_filter_mask_size_/2+1;
  }

  std::vector<scalar_t> coeffs(13,0.0);

  // make sure the mask size is appropriate
  if(gauss_filter_mask_size_==5){
    coeffs[0] = 0.0014;coeffs[1] = 0.1574;coeffs[2] = 0.62825;
    coeffs[3] = 0.1574;coeffs[4] = 0.0014;
  }
  else if (gauss_filter_mask_size_==7){
    coeffs[0] = 0.0060;coeffs[1] = 0.0606;coeffs[2] = 0.2418;
    coeffs[3] = 0.3831;coeffs[4] = 0.2418;coeffs[5] = 0.0606;
    coeffs[6] = 0.0060;
  }
  else if (gauss_filter_mask_size_==9){
    coeffs[0] = 0.0007;coeffs[1] = 0.0108;coeffs[2] = 0.0748;
    coeffs[3] = 0.2384;coeffs[4] = 0.3505;coeffs[5] = 0.2384;
    coeffs[6] = 0.0748;coeffs[7] = 0.0108;coeffs[8] = 0.0007;
  }
  else if (gauss_filter_mask_size_==11){
    coeffs[0] = 0.0001;coeffs[1] = 0.0017;coeffs[2] = 0.0168;
    coeffs[3] = 0.0870;coeffs[4] = 0.2328;coeffs[5] = 0.3231;
    coeffs[6] = 0.2328;coeffs[7] = 0.0870;coeffs[8] = 0.0168;
    coeffs[9] = 0.0017;coeffs[10] = 0.0001;
  }
  else if (gauss_filter_mask_size_==13){
    coeffs[0] = 0.0001;coeffs[1] = 0.0012;coeffs[2] = 0.0085;
    coeffs[3] = 0.0380;coeffs[4] = 0.1109;coeffs[5] = 0.2108;
    coeffs[6] = 0.2611;coeffs[7] = 0.2108;coeffs[8] = 0.1109;
    coeffs[9] = 0.0380;coeffs[10] = 0.0085;coeffs[11] = 0.0012;
    coeffs[12] = 0.0001;
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,
      "Error, the Gauss filter mask size is invalid (options include 5,7,9,11,13)");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(width_<gauss_filter_mask_size_||height_<gauss_filter_mask_size_,std::runtime_error,
    "Error, image too small (" << width_ << " x " << height_ << ") for gauss filtering with mask size " << gauss_filter_mask_size_);

  // copy over the old intensities
  for(int_t i=0;i<num_pixels();++i)
    intensities_temp_[i] = intensities_[i];

  for(int_t j=0;j<gauss_filter_mask_size_;++j){
    for(int_t i=0;i<gauss_filter_mask_size_;++i){
      gauss_filter_coeffs_[i][j] = coeffs[i]*coeffs[j];
    }
  }
  for(int_t y=0;y<height_;++y){
    for(int_t x=0;x<width_;++x){
      if(x>=gauss_filter_half_mask_&&x<width_-gauss_filter_half_mask_&&y>=gauss_filter_half_mask_&&y<height_-gauss_filter_half_mask_){
        intensity_t value = 0.0;
        for(int_t i=0;i<gauss_filter_mask_size_;++i){
          for(int_t j=0;j<gauss_filter_mask_size_;++j){
            // assumes intensity values have already been deep copied into intensities_temp_
            value += gauss_filter_coeffs_[i][j]*intensities_temp_[(y+(j-gauss_filter_half_mask_+1))*width_+x+(i-gauss_filter_half_mask_+1)];
          } //j
        } //i
        intensities_[y*width_+x] = value;
      }
    }
  }
  has_gauss_filter_ = true;
}

}// End DICe Namespace
