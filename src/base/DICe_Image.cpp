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

#include <Teuchos_ParameterList.hpp>

#include <cassert>

namespace DICe {

inline work_t keys_f0(const work_t & s){
  return 1.33333333333333*s*s*s - 2.33333333333333*s*s+ 1.0;
}
inline work_t keys_f1(const work_t & s){
  return -0.58333333333333*s*s*s + 3.0*s*s - 4.91666666666666*s + 2.5;
}
inline work_t keys_f2(const work_t & s){
  return 0.08333333333333*s*s*s - 0.66666666666666*s*s + 1.75*s - 1.5;
}

template <typename S>
Image_<S>::Image_(const char * file_name,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  width_(0),
  height_(0),
  offset_x_(0),
  offset_y_(0),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_(file_name),
  has_file_name_(true),
  gradient_method_(FINITE_DIFFERENCE)
{
  try{
    utils::read_image_dimensions(file_name,width_,height_);
    subimage_dims_from_params(params);
    // initialize the pixel containers
    utils::read_image(file_name,intensities_,params);
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, image file read failure");
  }
  default_constructor_tasks(params);
  post_allocation_tasks(params);
}

template <typename S>
Image_<S>::Image_(const int_t width,
  const int_t height,
  const S intensity):
  width_(width),
  height_(height),
  offset_x_(0),
  offset_y_(0),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_("(from scalar)"),
  has_file_name_(false),
  gradient_method_(FINITE_DIFFERENCE)
{
  TEUCHOS_TEST_FOR_EXCEPTION(width_<0,std::invalid_argument,"Error, width cannot be negative or zero.");
  TEUCHOS_TEST_FOR_EXCEPTION(height_<0,std::invalid_argument,"Error, height cannot be negative or zero.");
  intensities_ = Teuchos::ArrayRCP<S>(height_*width_,intensity);
  default_constructor_tasks(Teuchos::null);
  post_allocation_tasks(Teuchos::null);
}
template <typename S>
Image_<S>::Image_(Teuchos::RCP<Image_> img,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  width_(img->width()),
  height_(img->height()),
  offset_x_(img->offset_x()),
  offset_y_(img->offset_y()),
  has_gradients_(img->has_gradients()),
  has_gauss_filter_(img->has_gauss_filter()),
  file_name_(img->file_name()),
  has_file_name_(img->has_file_name()),
  gradient_method_(FINITE_DIFFERENCE)
{
  subimage_dims_from_params(params);
  TEUCHOS_TEST_FOR_EXCEPTION(offset_x_<0,std::invalid_argument,"Error, offset_x_ cannot be negative.");
  TEUCHOS_TEST_FOR_EXCEPTION(offset_y_<0,std::invalid_argument,"Error, offset_x_ cannot be negative.");
  const int_t src_width = img->width();
  const int_t src_height = img->height();

  // initialize the pixel containers
  intensities_ = Teuchos::ArrayRCP<S>(height_*width_,0.0);
  intensities_temp_ = Teuchos::ArrayRCP<S>(height_*width_,0.0);
  grad_x_ = Teuchos::ArrayRCP<work_t>(height_*width_,0.0);
  grad_y_ = Teuchos::ArrayRCP<work_t>(height_*width_,0.0);
  mask_ = Teuchos::ArrayRCP<work_t>(height_*width_,0.0);
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
        intensities_[y*width_+x] = 0;
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

template <typename S>
Image_<S>::Image_(const int_t array_width,
  const int_t array_height,
  const Teuchos::ArrayRCP<S> & intensities,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  width_(array_width),
  height_(array_height),
  offset_x_(0),
  offset_y_(0),
  intensities_(intensities),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_("(from array)"),
  has_file_name_(false),
  gradient_method_(FINITE_DIFFERENCE)
{
  if(params!=Teuchos::null){
    // Note for an image constructed from array, the offsets simply set the offset_x_ and offset_y_ values for this instance of the class
    // the full input array is still copied over (by reference) to the local intensity array.
    // This is mostly for propagating offsets if the intesity array RCP came from an image that was already a subimage
    if(params->isParameter(subimage_offset_x))
      offset_x_ = params->get<int_t>(subimage_offset_x);
    if(params->isParameter(subimage_offset_y))
      offset_y_ = params->get<int_t>(subimage_offset_y);
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(subimage_width),std::runtime_error,"cannot create subimage from intensity array");
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(subimage_height),std::runtime_error,"cannot create subimage from intensity array");
  }
  default_constructor_tasks(params);
  post_allocation_tasks(params);
}

template class Image_<storage_t>;
template class Image_<work_t>;

template <typename S>
void
Image_<S>::subimage_dims_from_params(const Teuchos::RCP<Teuchos::ParameterList> & params){
  if(params!=Teuchos::null){
    if(params->isParameter(subimage_offset_x))
      offset_x_ = params->get<int_t>(subimage_offset_x); // note using a default parameter for a param, adds that key to the list and sets it to the default
    if(params->isParameter(subimage_offset_y))
      offset_y_ = params->get<int_t>(subimage_offset_y);
    TEUCHOS_TEST_FOR_EXCEPTION(offset_x_<0||offset_x_>=width_,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(offset_y_<0||offset_y_>=height_,std::runtime_error,"");
    const int_t sub_height = params->isParameter(subimage_height) ? params->get<int_t>(subimage_height) : height_;
    const int_t sub_width = params->isParameter(subimage_width) ? params->get<int_t>(subimage_width) : width_;
    TEUCHOS_TEST_FOR_EXCEPTION(sub_width<=0||offset_x_+sub_width>width_,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(sub_height<=0||offset_y_+sub_height>height_,std::runtime_error,"");
    width_ = sub_width;
    height_ = sub_height;
  }
}

/// post allocation tasks
template <typename S>
void
Image_<S>::post_allocation_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){
  gauss_filter_mask_size_ = 7; // default sizes
  gauss_filter_half_mask_ = 4;
  if(params==Teuchos::null) return;
  gradient_method_ = params->get<Gradient_Method>(DICe::gradient_method,FINITE_DIFFERENCE);
  has_gauss_filter_ =  params->get<bool>(DICe::gauss_filter_images,false);
  gauss_filter_mask_size_ = params->get<int>(DICe::gauss_filter_mask_size,7);
  gauss_filter_half_mask_ = gauss_filter_mask_size_/2+1;
  if(has_gauss_filter_){
    DEBUG_MSG("Image::post_allocation_tasks(): gauss filtering image");
    gauss_filter(-1);
  }
  has_gradients_ = params->get<bool>(DICe::compute_image_gradients,false);
  if(has_gradients_){
    DEBUG_MSG("Image::post_allocation_tasks(): computing image gradients");
    compute_gradients();
  }
  if(params->isParameter(DICe::compute_laplacian_image)){
    if(params->get<bool>(DICe::compute_laplacian_image)==true){
      DEBUG_MSG("Image::post_allocation_tasks(): computing image laplacian");
      TEUCHOS_TEST_FOR_EXCEPTION(laplacian_==Teuchos::null,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(laplacian_.size()!=width_*height_,std::runtime_error,"");
      Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
      imgParams->set(DICe::compute_image_gradients,true); // automatically compute the gradients if the ref image is changed
      Teuchos::RCP<Image_<work_t>> grad_x_img = Teuchos::rcp(new Image_<work_t>(width_,height_,grad_x_,imgParams));
      Teuchos::RCP<Image_<work_t>> grad_y_img = Teuchos::rcp(new Image_<work_t>(width_,height_,grad_y_,imgParams));
      for(int_t y=0;y<height_;++y){
        for(int_t x=0;x<width_;++x){
          laplacian_[y*width_ + x] = grad_x_img->grad_x(x,y) + grad_y_img->grad_y(x,y);
        }
      }
    }
  }
}

template <typename S>
void
Image_<S>::default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){
  DEBUG_MSG("Image::default_contructor_tasks(): allocating image storage");
  grad_x_ = Teuchos::ArrayRCP<work_t>(height_*width_,0.0);
  grad_y_ = Teuchos::ArrayRCP<work_t>(height_*width_,0.0);
  intensities_temp_ = Teuchos::ArrayRCP<S>(height_*width_,0);
  mask_ = Teuchos::ArrayRCP<work_t>(height_*width_,0.0);
  if(params!=Teuchos::null){
    if(params->isParameter(DICe::compute_laplacian_image)){
      if(params->get<bool>(DICe::compute_laplacian_image)==true){
        laplacian_ = Teuchos::ArrayRCP<work_t>(height_*width_,0.0);
      }
    }
  }
  // image gradient coefficients
  grad_c1_ = 1.0/12.0;
  grad_c2_ = -8.0/12.0;
}

template <typename S>
void
Image_<S>::update(const char * file_name,
  const Teuchos::RCP<Teuchos::ParameterList> & params) {
  DEBUG_MSG("Image::update(): update called for image " << file_name);
  // check the dimensions of the incoming intensities
  if(params!=Teuchos::null){
    if(params->isParameter(subimage_offset_x))
      offset_x_ = params->get<int_t>(subimage_offset_x);
    if(params->isParameter(subimage_offset_y))
      offset_y_ = params->get<int_t>(subimage_offset_y);
    if(params->isParameter(subimage_width)){
      int_t tmp_width = params->get<int_t>(subimage_width);
      int_t tmp_height = params->get<int_t>(subimage_height);
      if(tmp_width!=width_||tmp_height!=height_){
        DEBUG_MSG("*** Image::update(): re-allocating image fields becuase image dims have changed ***");
        width_ = tmp_width;
        height_ = tmp_height;
        default_constructor_tasks(params);
      }
    }
  }
  try{
    utils::read_image(file_name,intensities_,params);
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, image file read failure");
  }
  post_allocation_tasks(params);
}

template void Image_<storage_t>::update(const char *,const Teuchos::RCP<Teuchos::ParameterList> &);
template void Image_<work_t>::update(const char *,const Teuchos::RCP<Teuchos::ParameterList> &);

template <typename S>
void
Image_<S>::interpolate_bilinear_all(work_t & intensity_val,
       work_t & grad_x_val, work_t & grad_y_val, const bool compute_gradient,
       const work_t & local_x, const work_t & local_y) {
  if(local_x<0.0||local_x>=width_-1.5||local_y<0.0||local_y>=height_-1.5) {
    intensity_val = 0.0;
    if (compute_gradient) {
      grad_x_val = 0.0;
      grad_y_val = 0.0;
    }
  }
  else {
    const int_t x1 = (int_t)local_x;
    const int_t x2 = x1+1;
    const int_t y1 = (int_t)local_y;
    const int_t y2  = y1+1;
    intensity_val = intensities_[y1*width_+x1]*(x2-local_x)*(y2-local_y)
      +intensities_[y1*width_+x2]*(local_x-x1)*(y2-local_y)
      +intensities_[y2*width_+x2]*(local_x-x1)*(local_y-y1)
      +intensities_[y2*width_+x1]*(x2-local_x)*(local_y-y1);
    if (compute_gradient) {
      grad_x_val = grad_x_[y1*width_+x1]*(x2-local_x)*(y2-local_y)
        +grad_x_[y1*width_+x2]*(local_x-x1)*(y2-local_y)
        +grad_x_[y2*width_+x2]*(local_x-x1)*(local_y-y1)
        +grad_x_[y2*width_+x1]*(x2-local_x)*(local_y-y1);

      grad_y_val = grad_y_[y1*width_+x1]*(x2-local_x)*(y2-local_y)
        +grad_y_[y1*width_+x2]*(local_x-x1)*(y2-local_y)
        +grad_y_[y2*width_+x2]*(local_x-x1)*(local_y-y1)
        +grad_y_[y2*width_+x1]*(x2-local_x)*(local_y-y1);
    }
  }
}

template void Image_<storage_t>::interpolate_bilinear_all(work_t &,work_t &,work_t &,const bool,const work_t &,const work_t &);
template void Image_<work_t>::interpolate_bilinear_all(work_t &,work_t &,work_t &,const bool,const work_t &,const work_t &);

template <typename S>
work_t
Image_<S>::interpolate_bilinear(const work_t & local_x, const work_t & local_y){

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

template work_t Image_<storage_t>::interpolate_bilinear(const work_t &,const work_t &);
template work_t Image_<work_t>::interpolate_bilinear(const work_t &,const work_t &);

template <typename S>
work_t
Image_<S>::interpolate_grad_x_bilinear(const work_t & local_x, const work_t & local_y){

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

template work_t Image_<storage_t>::interpolate_grad_x_bilinear(const work_t &,const work_t &);
template work_t Image_<work_t>::interpolate_grad_x_bilinear(const work_t &,const work_t &);

template <typename S>
work_t
Image_<S>::interpolate_grad_y_bilinear(const work_t & local_x, const work_t & local_y){
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

template work_t Image_<storage_t>::interpolate_grad_y_bilinear(const work_t &,const work_t &);
template work_t Image_<work_t>::interpolate_grad_y_bilinear(const work_t &,const work_t &);

template <typename S>
void
Image_<S>::interpolate_bicubic_all(work_t & intensity_val,
       work_t & grad_x_val, work_t & grad_y_val, const bool compute_gradient,
       const work_t & local_x, const work_t & local_y) {
  if(local_x<1.0||local_x>=width_-2.0||local_y<1.0||local_y>=height_-2.0) {
    intensity_val = this->interpolate_bilinear(local_x,local_y);
    if (compute_gradient) {
      grad_x_val = this->interpolate_grad_x_bilinear(local_x,local_y);
      grad_y_val = this->interpolate_grad_y_bilinear(local_x,local_y);
    }
  }
  const int_t x0  = (int_t)local_x;
  const int_t x1  = x0+1;
  const int_t x2  = x1+1;
  const int_t xm1 = x0-1;
  const int_t y0  = (int_t)local_y;
  const int_t y1 = y0+1;
  const int_t y2 = y1+1;
  const int_t ym1 = y0-1;
  const work_t x = local_x - x0;
  const work_t y = local_y - y0;
  const work_t x_2 = x * x;
  const work_t x_3 = x_2 * x;
  const work_t y_2 = y * y;
  const work_t y_3 = y_2 * y;
  // intensity
  const work_t fm10  = intensities_[y0*width_+xm1];
  const work_t f00   = intensities_[y0*width_+x0];
  const work_t f10   = intensities_[y0*width_+x1];
  const work_t f20   = intensities_[y0*width_+x2];
  const work_t fm11  = intensities_[y1*width_+xm1];
  const work_t f01   = intensities_[y1*width_+x0];
  const work_t f11   = intensities_[y1*width_+x1];
  const work_t f21   = intensities_[y1*width_+x2];
  const work_t fm12  = intensities_[y2*width_+xm1];
  const work_t f02   = intensities_[y2*width_+x0];
  const work_t f12   = intensities_[y2*width_+x1];
  const work_t f22   = intensities_[y2*width_+x2];
  const work_t fm1m1 = intensities_[ym1*width_+xm1];
  const work_t f0m1  = intensities_[ym1*width_+x0];
  const work_t f1m1  = intensities_[ym1*width_+x1];
  const work_t f2m1  = intensities_[ym1*width_+x2];
  #ifdef DICE_USE_DOUBLE
    intensity_val = f00 + (-0.5*f0m1 + .5*f01)*y + (f0m1 - 2.5*f00 + 2.0*f01 - .5*f02)*y_2 + (-0.5*f0m1 + 1.5*f00 - 1.5*f01 + .5*f02)*y_3
      + ((-0.5*fm10 + .5*f10) + (0.25*fm1m1 - .25*fm11 - .25*f1m1 + .25*f11)*y + (-0.5*fm1m1 + 1.25*fm10 - fm11 + .25*fm12 +
          0.5*f1m1 - 1.25*f10 + f11 - .25*f12)*y_2 + (0.25*fm1m1 - .75*fm10 + .75*fm11 - .25*fm12 - .25*f1m1 + .75*f10 - .75*f11 + .25*f12)*y_3) * x
      +((fm10 - 2.5*f00 + 2.0*f10 - .5*f20) + (-0.5*fm1m1 + .5*fm11 + 1.25*f0m1 - 1.25*f01 - f1m1 + f11 + .25*f2m1 - .25*f21)*y + (fm1m1 - 2.5*fm10 + 2.0*fm11
          - .5*fm12 - 2.5*f0m1 + 6.25*f00 - 5.0*f01 + 1.25*f02 + 2.0*f1m1 - 5.0*f10 + 4.0*f11 - f12 - .5*f2m1 + 1.25*f20 - f21 + .25*f22)*y_2 +
          (-0.5*fm1m1 + 1.5*fm10 - 1.5*fm11 + .5*fm12 + 1.25*f0m1 - 3.75*f00 + 3.75*f01 - 1.25*f02 - f1m1 + 3.0*f10 - 3.0*f11 + f12 + .25*f2m1 - .75*f20 + .75*f21 - .25*f22)*y_3)*x_2
      +((-.5*fm10 + 1.5*f00 - 1.5*f10 + .5*f20) + (0.25*fm1m1 - .25*fm11 - .75*f0m1 + .75*f01 + .75*f1m1 - .75*f11 - .25*f2m1 + .25*f21)*y +
          (-.5*fm1m1 + 1.25*fm10 - fm11 + .25*fm12 + 1.5*f0m1 - 3.75*f00 + 3.0*f01 - .75*f02 - 1.5*f1m1 + 3.75*f10 - 3.0*f11 + .75*f12 + .5*f2m1 - 1.25*f20 + f21 - .25*f22)*y_2
          + (0.25*fm1m1 - .75*fm10 + .75*fm11 - .25*fm12 - .75*f0m1 + 2.25*f00 - 2.25*f01 + .75*f02 + .75*f1m1 - 2.25*f10 + 2.25*f11 - .75*f12 - .25*f2m1 + .75*f20 - .75*f21 + .25*f22)*y_3)*x_3;
  #else
    intensity_val = f00 + (-0.5f*f0m1 + .5f*f01)*y + (f0m1 - 2.5f*f00 + 2.0f*f01 - .5f*f02)*y_2 + (-0.5f*f0m1 + 1.5f*f00 - 1.5f*f01 + .5f*f02)*y_3
      + ((-0.5f*fm10 + .5f*f10) + (0.25f*fm1m1 - .25f*fm11 - .25f*f1m1 + .25f*f11)*y + (-0.5f*fm1m1 + 1.25f*fm10 - fm11 + .25f*fm12 +
          0.5f*f1m1 - 1.25f*f10 + f11 - .25f*f12)*y_2 + (0.25f*fm1m1 - .75f*fm10 + .75f*fm11 - .25f*fm12 - .25f*f1m1 + .75f*f10 - .75f*f11 + .25f*f12)*y_3) * x
      +((fm10 - 2.5f*f00 + 2.0f*f10 - .5f*f20) + (-0.5f*fm1m1 + .5f*fm11 + 1.25f*f0m1 - 1.25f*f01 - f1m1 + f11 + .25f*f2m1 - .25f*f21)*y + (fm1m1 - 2.5f*fm10 + 2.0f*fm11
          - .5f*fm12 - 2.5f*f0m1 + 6.25f*f00 - 5.0f*f01 + 1.25f*f02 + 2.0f*f1m1 - 5.0f*f10 + 4.0f*f11 - f12 - .5f*f2m1 + 1.25f*f20 - f21 + .25f*f22)*y_2 +
          (-0.5f*fm1m1 + 1.5f*fm10 - 1.5f*fm11 + .5f*fm12 + 1.25f*f0m1 - 3.75f*f00 + 3.75f*f01 - 1.25f*f02 - f1m1 + 3.0f*f10 - 3.0f*f11 + f12 + .25f*f2m1 - .75f*f20 + .75f*f21 - .25f*f22)*y_3)*x_2
      +((-.5f*fm10 + 1.5f*f00 - 1.5f*f10 + .5f*f20) + (0.25f*fm1m1 - .25f*fm11 - .75f*f0m1 + .75f*f01 + .75f*f1m1 - .75f*f11 - .25f*f2m1 + .25f*f21)*y +
          (-.5f*fm1m1 + 1.25f*fm10 - fm11 + .25f*fm12 + 1.5f*f0m1 - 3.75f*f00 + 3.0f*f01 - .75f*f02 - 1.5f*f1m1 + 3.75f*f10 - 3.0f*f11 + .75f*f12 + .5f*f2m1 - 1.25f*f20 + f21 - .25f*f22)*y_2
          + (0.25f*fm1m1 - .75f*fm10 + .75f*fm11 - .25f*fm12 - .75f*f0m1 + 2.25f*f00 - 2.25f*f01 + .75f*f02 + .75f*f1m1 - 2.25f*f10 + 2.25f*f11 - .75f*f12 - .25f*f2m1 + .75f*f20 - .75f*f21 + .25f*f22)*y_3)*x_3;
  #endif


  if (compute_gradient) {
    // grad_x
    const work_t gxfm10  = grad_x_[y0*width_+xm1];
    const work_t gxf00   = grad_x_[y0*width_+x0];
    const work_t gxf10   = grad_x_[y0*width_+x1];
    const work_t gxf20   = grad_x_[y0*width_+x2];
    const work_t gxfm11  = grad_x_[y1*width_+xm1];
    const work_t gxf01   = grad_x_[y1*width_+x0];
    const work_t gxf11   = grad_x_[y1*width_+x1];
    const work_t gxf21   = grad_x_[y1*width_+x2];
    const work_t gxfm12  = grad_x_[y2*width_+xm1];
    const work_t gxf02   = grad_x_[y2*width_+x0];
    const work_t gxf12   = grad_x_[y2*width_+x1];
    const work_t gxf22   = grad_x_[y2*width_+x2];
    const work_t gxfm1m1 = grad_x_[ym1*width_+xm1];
    const work_t gxf0m1  = grad_x_[ym1*width_+x0];
    const work_t gxf1m1  = grad_x_[ym1*width_+x1];
    const work_t gxf2m1  = grad_x_[ym1*width_+x2];
    // grad_y
    const work_t gyfm10  = grad_y_[y0*width_+xm1];
    const work_t gyf00   = grad_y_[y0*width_+x0];
    const work_t gyf10   = grad_y_[y0*width_+x1];
    const work_t gyf20   = grad_y_[y0*width_+x2];
    const work_t gyfm11  = grad_y_[y1*width_+xm1];
    const work_t gyf01   = grad_y_[y1*width_+x0];
    const work_t gyf11   = grad_y_[y1*width_+x1];
    const work_t gyf21   = grad_y_[y1*width_+x2];
    const work_t gyfm12  = grad_y_[y2*width_+xm1];
    const work_t gyf02   = grad_y_[y2*width_+x0];
    const work_t gyf12   = grad_y_[y2*width_+x1];
    const work_t gyf22   = grad_y_[y2*width_+x2];
    const work_t gyfm1m1 = grad_y_[ym1*width_+xm1];
    const work_t gyf0m1  = grad_y_[ym1*width_+x0];
    const work_t gyf1m1  = grad_y_[ym1*width_+x1];
    const work_t gyf2m1  = grad_y_[ym1*width_+x2];

    #ifdef DICE_USE_DOUBLE
      grad_x_val = gxf00 + (-0.5*gxf0m1 + .5*gxf01)*y + (gxf0m1 - 2.5*gxf00 + 2.0*gxf01 - .5*gxf02)*y_2 + (-0.5*gxf0m1 + 1.5*gxf00 - 1.5*gxf01 + .5*gxf02)*y_3
        + ((-0.5*gxfm10 + .5*gxf10) + (0.25*gxfm1m1 - .25*gxfm11 - .25*gxf1m1 + .25*gxf11)*y + (-0.5*gxfm1m1 + 1.25*gxfm10 - gxfm11 + .25*gxfm12 +
            0.5*gxf1m1 - 1.25*gxf10 + gxf11 - .25*gxf12)*y_2 + (0.25*gxfm1m1 - .75*gxfm10 + .75*gxfm11 - .25*gxfm12 - .25*gxf1m1 + .75*gxf10 - .75*gxf11 + .25*gxf12)*y_3) * x
        +((gxfm10 - 2.5*gxf00 + 2.0*gxf10 - .5*gxf20) + (-0.5*gxfm1m1 + .5*gxfm11 + 1.25*gxf0m1 - 1.25*gxf01 - gxf1m1 + gxf11 + .25*gxf2m1 - .25*gxf21)*y + (gxfm1m1 - 2.5*gxfm10 + 2.0*gxfm11
            - .5*gxfm12 - 2.5*gxf0m1 + 6.25*gxf00 - 5.0*gxf01 + 1.25*gxf02 + 2.0*gxf1m1 - 5.0*gxf10 + 4.0*gxf11 - gxf12 - .5*gxf2m1 + 1.25*gxf20 - gxf21 + .25*gxf22)*y_2 +
            (-0.5*gxfm1m1 + 1.5*gxfm10 - 1.5*gxfm11 + .5*gxfm12 + 1.25*gxf0m1 - 3.75*gxf00 + 3.75*gxf01 - 1.25*gxf02 - gxf1m1 + 3.0*gxf10 - 3.0*gxf11 + gxf12 + .25*gxf2m1 - .75*gxf20 + .75*gxf21 - .25*gxf22)*y_3)*x_2
        +((-.5*gxfm10 + 1.5*gxf00 - 1.5*gxf10 + .5*gxf20) + (0.25*gxfm1m1 - .25*gxfm11 - .75*gxf0m1 + .75*gxf01 + .75*gxf1m1 - .75*gxf11 - .25*gxf2m1 + .25*gxf21)*y +
            (-.5*gxfm1m1 + 1.25*gxfm10 - gxfm11 + .25*gxfm12 + 1.5*gxf0m1 - 3.75*gxf00 + 3.0*gxf01 - .75*gxf02 - 1.5*gxf1m1 + 3.75*gxf10 - 3.0*gxf11 + .75*gxf12 + .5*gxf2m1 - 1.25*gxf20 + gxf21 - .25*gxf22)*y_2
            + (0.25*gxfm1m1 - .75*gxfm10 + .75*gxfm11 - .25*gxfm12 - .75*gxf0m1 + 2.25*gxf00 - 2.25*gxf01 + .75*gxf02 + .75*gxf1m1 - 2.25*gxf10 + 2.25*gxf11 - .75*gxf12 - .25*gxf2m1 + .75*gxf20 - .75*gxf21 + .25*gxf22)*y_3)*x_3;

      grad_y_val = gyf00 + (-0.5*gyf0m1 + .5*gyf01)*y + (gyf0m1 - 2.5*gyf00 + 2.0*gyf01 - .5*gyf02)*y_2 + (-0.5*gyf0m1 + 1.5*gyf00 - 1.5*gyf01 + .5*gyf02)*y_3
        + ((-0.5*gyfm10 + .5*gyf10) + (0.25*gyfm1m1 - .25*gyfm11 - .25*gyf1m1 + .25*gyf11)*y + (-0.5*gyfm1m1 + 1.25*gyfm10 - gyfm11 + .25*gyfm12 +
            0.5*gyf1m1 - 1.25*gyf10 + gyf11 - .25*gyf12)*y_2 + (0.25*gyfm1m1 - .75*gyfm10 + .75*gyfm11 - .25*gyfm12 - .25*gyf1m1 + .75*gyf10 - .75*gyf11 + .25*gyf12)*y_3) * x
        +((gyfm10 - 2.5*gyf00 + 2.0*gyf10 - .5*gyf20) + (-0.5*gyfm1m1 + .5*gyfm11 + 1.25*gyf0m1 - 1.25*gyf01 - gyf1m1 + gyf11 + .25*gyf2m1 - .25*gyf21)*y + (gyfm1m1 - 2.5*gyfm10 + 2.0*gyfm11
            - .5*gyfm12 - 2.5*gyf0m1 + 6.25*gyf00 - 5.0*gyf01 + 1.25*gyf02 + 2.0*gyf1m1 - 5.0*gyf10 + 4.0*gyf11 - gyf12 - .5*gyf2m1 + 1.25*gyf20 - gyf21 + .25*gyf22)*y_2 +
            (-0.5*gyfm1m1 + 1.5*gyfm10 - 1.5*gyfm11 + .5*gyfm12 + 1.25*gyf0m1 - 3.75*gyf00 + 3.75*gyf01 - 1.25*gyf02 - gyf1m1 + 3.0*gyf10 - 3.0*gyf11 + gyf12 + .25*gyf2m1 - .75*gyf20 + .75*gyf21 - .25*gyf22)*y_3)*x_2
        +((-.5*gyfm10 + 1.5*gyf00 - 1.5*gyf10 + .5*gyf20) + (0.25*gyfm1m1 - .25*gyfm11 - .75*gyf0m1 + .75*gyf01 + .75*gyf1m1 - .75*gyf11 - .25*gyf2m1 + .25*gyf21)*y +
            (-.5*gyfm1m1 + 1.25*gyfm10 - gyfm11 + .25*gyfm12 + 1.5*gyf0m1 - 3.75*gyf00 + 3.0*gyf01 - .75*gyf02 - 1.5*gyf1m1 + 3.75*gyf10 - 3.0*gyf11 + .75*gyf12 + .5*gyf2m1 - 1.25*gyf20 + gyf21 - .25*gyf22)*y_2
            + (0.25*gyfm1m1 - .75*gyfm10 + .75*gyfm11 - .25*gyfm12 - .75*gyf0m1 + 2.25*gyf00 - 2.25*gyf01 + .75*gyf02 + .75*gyf1m1 - 2.25*gyf10 + 2.25*gyf11 - .75*gyf12 - .25*gyf2m1 + .75*gyf20 - .75*gyf21 + .25*gyf22)*y_3)*x_3;
    #else
      grad_x_val = gxf00 + (-0.5f*gxf0m1 + .5f*gxf01)*y + (gxf0m1 - 2.5f*gxf00 + 2.0f*gxf01 - .5f*gxf02)*y_2 + (-0.5f*gxf0m1 + 1.5f*gxf00 - 1.5f*gxf01 + .5f*gxf02)*y_3
        + ((-0.5f*gxfm10 + .5f*gxf10) + (0.25f*gxfm1m1 - .25f*gxfm11 - .25f*gxf1m1 + .25f*gxf11)*y + (-0.5f*gxfm1m1 + 1.25f*gxfm10 - gxfm11 + .25f*gxfm12 +
            0.5f*gxf1m1 - 1.25f*gxf10 + gxf11 - .25f*gxf12)*y_2 + (0.25f*gxfm1m1 - .75f*gxfm10 + .75f*gxfm11 - .25f*gxfm12 - .25f*gxf1m1 + .75f*gxf10 - .75f*gxf11 + .25f*gxf12)*y_3) * x
        +((gxfm10 - 2.5f*gxf00 + 2.0f*gxf10 - .5f*gxf20) + (-0.5f*gxfm1m1 + .5f*gxfm11 + 1.25f*gxf0m1 - 1.25f*gxf01 - gxf1m1 + gxf11 + .25f*gxf2m1 - .25f*gxf21)*y + (gxfm1m1 - 2.5f*gxfm10 + 2.0f*gxfm11
            - .5f*gxfm12 - 2.5f*gxf0m1 + 6.25f*gxf00 - 5.0f*gxf01 + 1.25f*gxf02 + 2.0f*gxf1m1 - 5.0f*gxf10 + 4.0f*gxf11 - gxf12 - .5f*gxf2m1 + 1.25f*gxf20 - gxf21 + .25f*gxf22)*y_2 +
            (-0.5f*gxfm1m1 + 1.5f*gxfm10 - 1.5f*gxfm11 + .5f*gxfm12 + 1.25f*gxf0m1 - 3.75f*gxf00 + 3.75f*gxf01 - 1.25f*gxf02 - gxf1m1 + 3.0f*gxf10 - 3.0f*gxf11 + gxf12 + .25f*gxf2m1 - .75f*gxf20 + .75f*gxf21 - .25f*gxf22)*y_3)*x_2
        +((-.5f*gxfm10 + 1.5f*gxf00 - 1.5f*gxf10 + .5f*gxf20) + (0.25f*gxfm1m1 - .25f*gxfm11 - .75f*gxf0m1 + .75f*gxf01 + .75f*gxf1m1 - .75f*gxf11 - .25f*gxf2m1 + .25f*gxf21)*y +
            (-.5f*gxfm1m1 + 1.25f*gxfm10 - gxfm11 + .25f*gxfm12 + 1.5f*gxf0m1 - 3.75f*gxf00 + 3.0f*gxf01 - .75f*gxf02 - 1.5f*gxf1m1 + 3.75f*gxf10 - 3.0f*gxf11 + .75f*gxf12 + .5f*gxf2m1 - 1.25f*gxf20 + gxf21 - .25f*gxf22)*y_2
            + (0.25f*gxfm1m1 - .75f*gxfm10 + .75f*gxfm11 - .25f*gxfm12 - .75f*gxf0m1 + 2.25f*gxf00 - 2.25f*gxf01 + .75f*gxf02 + .75f*gxf1m1 - 2.25f*gxf10 + 2.25f*gxf11 - .75f*gxf12 - .25f*gxf2m1 + .75f*gxf20 - .75f*gxf21 + .25f*gxf22)*y_3)*x_3;

      grad_y_val = gyf00 + (-0.5f*gyf0m1 + .5f*gyf01)*y + (gyf0m1 - 2.5f*gyf00 + 2.0f*gyf01 - .5f*gyf02)*y_2 + (-0.5f*gyf0m1 + 1.5f*gyf00 - 1.5f*gyf01 + .5f*gyf02)*y_3
        + ((-0.5f*gyfm10 + .5f*gyf10) + (0.25f*gyfm1m1 - .25f*gyfm11 - .25f*gyf1m1 + .25f*gyf11)*y + (-0.5f*gyfm1m1 + 1.25f*gyfm10 - gyfm11 + .25f*gyfm12 +
            0.5f*gyf1m1 - 1.25f*gyf10 + gyf11 - .25f*gyf12)*y_2 + (0.25f*gyfm1m1 - .75f*gyfm10 + .75f*gyfm11 - .25f*gyfm12 - .25f*gyf1m1 + .75f*gyf10 - .75f*gyf11 + .25f*gyf12)*y_3) * x
        +((gyfm10 - 2.5f*gyf00 + 2.0f*gyf10 - .5f*gyf20) + (-0.5f*gyfm1m1 + .5f*gyfm11 + 1.25f*gyf0m1 - 1.25f*gyf01 - gyf1m1 + gyf11 + .25f*gyf2m1 - .25f*gyf21)*y + (gyfm1m1 - 2.5f*gyfm10 + 2.0f*gyfm11
            - .5f*gyfm12 - 2.5f*gyf0m1 + 6.25f*gyf00 - 5.0f*gyf01 + 1.25f*gyf02 + 2.0f*gyf1m1 - 5.0f*gyf10 + 4.0f*gyf11 - gyf12 - .5f*gyf2m1 + 1.25f*gyf20 - gyf21 + .25f*gyf22)*y_2 +
            (-0.5f*gyfm1m1 + 1.5f*gyfm10 - 1.5f*gyfm11 + .5f*gyfm12 + 1.25f*gyf0m1 - 3.75f*gyf00 + 3.75f*gyf01 - 1.25f*gyf02 - gyf1m1 + 3.0f*gyf10 - 3.0f*gyf11 + gyf12 + .25f*gyf2m1 - .75f*gyf20 + .75f*gyf21 - .25f*gyf22)*y_3)*x_2
        +((-.5f*gyfm10 + 1.5f*gyf00 - 1.5f*gyf10 + .5f*gyf20) + (0.25f*gyfm1m1 - .25f*gyfm11 - .75f*gyf0m1 + .75f*gyf01 + .75f*gyf1m1 - .75f*gyf11 - .25f*gyf2m1 + .25f*gyf21)*y +
            (-.5f*gyfm1m1 + 1.25f*gyfm10 - gyfm11 + .25f*gyfm12 + 1.5f*gyf0m1 - 3.75f*gyf00 + 3.0f*gyf01 - .75f*gyf02 - 1.5f*gyf1m1 + 3.75f*gyf10 - 3.0f*gyf11 + .75f*gyf12 + .5f*gyf2m1 - 1.25f*gyf20 + gyf21 - .25f*gyf22)*y_2
            + (0.25f*gyfm1m1 - .75f*gyfm10 + .75f*gyfm11 - .25f*gyfm12 - .75f*gyf0m1 + 2.25f*gyf00 - 2.25f*gyf01 + .75f*gyf02 + .75f*gyf1m1 - 2.25f*gyf10 + 2.25f*gyf11 - .75f*gyf12 - .25f*gyf2m1 + .75f*gyf20 - .75f*gyf21 + .25f*gyf22)*y_3)*x_3;
    #endif
  }

}

template void Image_<storage_t>::interpolate_bicubic_all(work_t &,work_t &,work_t &,const bool,const work_t &,const work_t &);
template void Image_<work_t>::interpolate_bicubic_all(work_t &,work_t &,work_t &,const bool,const work_t &,const work_t &);

template <typename S>
work_t
Image_<S>::interpolate_bicubic(const work_t & local_x, const work_t & local_y){
  if(local_x<1.0||local_x>=width_-2.0||local_y<1.0||local_y>=height_-2.0) return this->interpolate_bilinear(local_x,local_y);

  const int_t x0  = (int_t)local_x;
  const int_t x1  = x0+1;
  const int_t x2  = x1+1;
  const int_t xm1 = x0-1;
  const int_t y0  = (int_t)local_y;
  const int_t y1 = y0+1;
  const int_t y2 = y1+1;
  const int_t ym1 = y0-1;
  const work_t x = local_x - x0;
  const work_t y = local_y - y0;
  const work_t x_2 = x * x;
  const work_t x_3 = x_2 * x;
  const work_t y_2 = y * y;
  const work_t y_3 = y_2 * y;
  const work_t fm10  = intensities_[y0*width_+xm1];
  const work_t f00   = intensities_[y0*width_+x0];
  const work_t f10   = intensities_[y0*width_+x1];
  const work_t f20   = intensities_[y0*width_+x2];
  const work_t fm11  = intensities_[y1*width_+xm1];
  const work_t f01   = intensities_[y1*width_+x0];
  const work_t f11   = intensities_[y1*width_+x1];
  const work_t f21   = intensities_[y1*width_+x2];
  const work_t fm12  = intensities_[y2*width_+xm1];
  const work_t f02   = intensities_[y2*width_+x0];
  const work_t f12   = intensities_[y2*width_+x1];
  const work_t f22   = intensities_[y2*width_+x2];
  const work_t fm1m1 = intensities_[ym1*width_+xm1];
  const work_t f0m1  = intensities_[ym1*width_+x0];
  const work_t f1m1  = intensities_[ym1*width_+x1];
  const work_t f2m1  = intensities_[ym1*width_+x2];
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

template work_t Image_<storage_t>::interpolate_bicubic(const work_t &, const work_t &);
template work_t Image_<work_t>::interpolate_bicubic(const work_t &, const work_t &);

template <typename S>
work_t
Image_<S>::interpolate_grad_x_bicubic(const work_t & local_x, const work_t & local_y){
  if(local_x<1.0||local_x>=width_-2.0||local_y<1.0||local_y>=height_-2.0) return this->interpolate_grad_x_bilinear(local_x,local_y);

  const int_t x0  = (int_t)local_x;
  const int_t x1  = x0+1;
  const int_t x2  = x1+1;
  const int_t xm1 = x0-1;
  const int_t y0  = (int_t)local_y;
  const int_t y1 = y0+1;
  const int_t y2 = y1+1;
  const int_t ym1 = y0-1;
  const work_t x = local_x - x0;
  const work_t y = local_y - y0;
  const work_t x_2 = x * x;
  const work_t x_3 = x_2 * x;
  const work_t y_2 = y * y;
  const work_t y_3 = y_2 * y;
  const work_t fm10  = grad_x_[y0*width_+xm1];
  const work_t f00   = grad_x_[y0*width_+x0];
  const work_t f10   = grad_x_[y0*width_+x1];
  const work_t f20   = grad_x_[y0*width_+x2];
  const work_t fm11  = grad_x_[y1*width_+xm1];
  const work_t f01   = grad_x_[y1*width_+x0];
  const work_t f11   = grad_x_[y1*width_+x1];
  const work_t f21   = grad_x_[y1*width_+x2];
  const work_t fm12  = grad_x_[y2*width_+xm1];
  const work_t f02   = grad_x_[y2*width_+x0];
  const work_t f12   = grad_x_[y2*width_+x1];
  const work_t f22   = grad_x_[y2*width_+x2];
  const work_t fm1m1 = grad_x_[ym1*width_+xm1];
  const work_t f0m1  = grad_x_[ym1*width_+x0];
  const work_t f1m1  = grad_x_[ym1*width_+x1];
  const work_t f2m1  = grad_x_[ym1*width_+x2];
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

template work_t Image_<storage_t>::interpolate_grad_x_bicubic(const work_t &, const work_t &);
template work_t Image_<work_t>::interpolate_grad_x_bicubic(const work_t &, const work_t &);

template <typename S>
work_t
Image_<S>::interpolate_grad_y_bicubic(const work_t & local_x, const work_t & local_y){
  if(local_x<1.0||local_x>=width_-2.0||local_y<1.0||local_y>=height_-2.0) return this->interpolate_grad_y_bilinear(local_x,local_y);

  const int_t x0  = (int_t)local_x;
  const int_t x1  = x0+1;
  const int_t x2  = x1+1;
  const int_t xm1 = x0-1;
  const int_t y0  = (int_t)local_y;
  const int_t y1 = y0+1;
  const int_t y2 = y1+1;
  const int_t ym1 = y0-1;
  const work_t x = local_x - x0;
  const work_t y = local_y - y0;
  const work_t x_2 = x * x;
  const work_t x_3 = x_2 * x;
  const work_t y_2 = y * y;
  const work_t y_3 = y_2 * y;
  const work_t fm10  = grad_y_[y0*width_+xm1];
  const work_t f00   = grad_y_[y0*width_+x0];
  const work_t f10   = grad_y_[y0*width_+x1];
  const work_t f20   = grad_y_[y0*width_+x2];
  const work_t fm11  = grad_y_[y1*width_+xm1];
  const work_t f01   = grad_y_[y1*width_+x0];
  const work_t f11   = grad_y_[y1*width_+x1];
  const work_t f21   = grad_y_[y1*width_+x2];
  const work_t fm12  = grad_y_[y2*width_+xm1];
  const work_t f02   = grad_y_[y2*width_+x0];
  const work_t f12   = grad_y_[y2*width_+x1];
  const work_t f22   = grad_y_[y2*width_+x2];
  const work_t fm1m1 = grad_y_[ym1*width_+xm1];
  const work_t f0m1  = grad_y_[ym1*width_+x0];
  const work_t f1m1  = grad_y_[ym1*width_+x1];
  const work_t f2m1  = grad_y_[ym1*width_+x2];
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

template work_t Image_<storage_t>::interpolate_grad_y_bicubic(const work_t &, const work_t &);
template work_t Image_<work_t>::interpolate_grad_y_bicubic(const work_t &, const work_t &);

template <typename S>
void
Image_<S>::interpolate_keys_fourth_all(work_t & intensity_val,
       work_t & grad_x_val, work_t & grad_y_val, const bool compute_gradient,
       const work_t & local_x, const work_t & local_y) {
  intensity_val = 0.0;
  if (compute_gradient) {
    grad_x_val = 0.0;
    grad_y_val = 0.0;
  }
  static std::vector<work_t> coeffs_x(6,0.0);
  static std::vector<work_t> coeffs_y(6,0.0);
  static work_t dx = 0.0;
  static work_t dy = 0.0;
  static int_t ix=0,iy=0;
  //static intensity_t value=0.0;
  static work_t cc = 0.0;
  ix = (int_t)local_x;
  iy = (int_t)local_y;
  if(local_x<=2.5||local_x>=width_-3.5||local_y<=2.5||local_y>=height_-3.5) {
    intensity_val  =  this->interpolate_bilinear(local_x,local_y);
    if (compute_gradient) {
      grad_x_val = this->interpolate_grad_x_bilinear(local_x,local_y);
      grad_y_val = this->interpolate_grad_y_bilinear(local_x,local_y);
    }
  }
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
  //value = 0.0;
  for(int_t m=0;m<6;++m){
    for(int_t n=0;n<6;++n){
      cc = coeffs_y[m]*coeffs_x[n];
      intensity_val += cc*intensities_[(iy-2+m)*width_ + ix-2+n];
      if (compute_gradient) {
        grad_x_val += cc*grad_x_[(iy-2+m)*width_ + ix-2+n];
        grad_y_val += cc*grad_y_[(iy-2+m)*width_ + ix-2+n];
      }
    }
  }
}

template void Image_<storage_t>::interpolate_keys_fourth_all(work_t &, work_t &, work_t &, const bool, const work_t &, const work_t &);
template void Image_<work_t>::interpolate_keys_fourth_all(work_t &, work_t &, work_t &, const bool, const work_t &, const work_t &);

template <typename S>
work_t
Image_<S>::interpolate_keys_fourth(const work_t & local_x, const work_t & local_y){
  static std::vector<work_t> coeffs_x(6,0.0);
  static std::vector<work_t> coeffs_y(6,0.0);
  static work_t dx = 0.0;
  static work_t dy = 0.0;
  static int_t ix=0,iy=0;
  static work_t value=0.0;
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

template work_t Image_<storage_t>::interpolate_keys_fourth(const work_t &, const work_t &);
template work_t Image_<work_t>::interpolate_keys_fourth(const work_t &, const work_t &);

template <typename S>
work_t
Image_<S>::interpolate_grad_x_keys_fourth(const work_t & local_x, const work_t & local_y){
  static std::vector<work_t> coeffs_x(6,0.0);
  static std::vector<work_t> coeffs_y(6,0.0);
  static work_t dx = 0.0;
  static work_t dy = 0.0;
  static int_t ix=0,iy=0;
  static work_t value=0.0;
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

template work_t Image_<storage_t>::interpolate_grad_x_keys_fourth(const work_t &, const work_t &);
template work_t Image_<work_t>::interpolate_grad_x_keys_fourth(const work_t &, const work_t &);

template <typename S>
work_t
Image_<S>::interpolate_grad_y_keys_fourth(const work_t & local_x, const work_t & local_y){
  static std::vector<work_t> coeffs_x(6,0.0);
  static std::vector<work_t> coeffs_y(6,0.0);
  static work_t dx = 0.0;
  static work_t dy = 0.0;
  static int_t ix=0,iy=0;
  static work_t value=0.0;
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

template work_t Image_<storage_t>::interpolate_grad_y_keys_fourth(const work_t &, const work_t &);
template work_t Image_<work_t>::interpolate_grad_y_keys_fourth(const work_t &, const work_t &);

template <typename S>
void
Image_<S>::compute_gradients(){
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

template <typename S>
void
Image_<S>::smooth_gradients_convolution_5_point(){

  static work_t smooth_coeffs[][5] = {{0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
                                        {0.015625,   0.0625,   0.09375,   0.0625,   0.015625},
                                        {0.0234375,  0.09375,  0.140625,  0.09375,  0.0234375},
                                        {0.015625,   0.0625,   0.09375,   0.0625,   0.015625},
                                        {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};
  static int_t smooth_offsets[] =  {-2, -1, 0, 1, 2};

  Teuchos::ArrayRCP<work_t> grad_x_temp(width_*height_,0.0);
  Teuchos::ArrayRCP<work_t> grad_y_temp(width_*height_,0.0);
  for(int_t i=0;i<width_*height_;++i){
    grad_x_temp[i] = grad_x_[i];
    grad_y_temp[i] = grad_y_[i];
  }
  for(int_t y=2;y<height_-2;++y){
    for(int_t x=2;x<width_-2;++x){
      work_t value_x = 0.0, value_y = 0.0;
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

template <typename S>
void
Image_<S>::compute_gradients_finite_difference(){
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

template <typename S>
void
Image_<S>::apply_mask(const Conformal_Area_Def & area_def,
  const bool smooth_edges){
  // first create the mask:
  create_mask(area_def,smooth_edges);
  for(int_t i=0;i<num_pixels();++i)
    intensities_[i] = static_cast<S>(mask_[i]*intensities_[i]);
}

template void Image_<storage_t>::apply_mask(const Conformal_Area_Def &,const bool);
template void Image_<work_t>::apply_mask(const Conformal_Area_Def &,const bool);

template <typename S>
void
Image_<S>::apply_mask(const bool smooth_edges){
  if(smooth_edges){
    static work_t smoothing_coeffs[5][5];
    std::vector<work_t> coeffs(5,0.0);
    coeffs[0] = 0.0014;coeffs[1] = 0.1574;coeffs[2] = 0.62825;
    coeffs[3] = 0.1574;coeffs[4] = 0.0014;
    for(int_t j=0;j<5;++j){
      for(int_t i=0;i<5;++i){
        smoothing_coeffs[i][j] = coeffs[i]*coeffs[j];
      }
    }
    Teuchos::ArrayRCP<work_t> mask_tmp(mask_.size(),0.0);
    for(int_t i=0;i<mask_tmp.size();++i)
      mask_tmp[i] = mask_[i];
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        if(x>=2&&x<width_-2&&y>=2&&y<height_-2){ // 2 is half the gauss_mask size
          work_t value = 0.0;
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
    intensities_[i] = static_cast<S>(mask_[i]*intensities_[i]);
}

template void Image_<storage_t>::apply_mask(const bool);
template void Image_<work_t>::apply_mask(const bool);

template <typename S>
void
Image_<S>::create_mask(const Conformal_Area_Def & area_def,
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
    static work_t smoothing_coeffs[5][5];
    std::vector<work_t> coeffs(5,0.0);
    coeffs[0] = 0.0014;coeffs[1] = 0.1574;coeffs[2] = 0.62825;
    coeffs[3] = 0.1574;coeffs[4] = 0.0014;
    for(int_t j=0;j<5;++j){
      for(int_t i=0;i<5;++i){
        smoothing_coeffs[i][j] = coeffs[i]*coeffs[j];
      }
    }
    Teuchos::ArrayRCP<work_t> mask_tmp(mask_.size(),0.0);
    for(int_t i=0;i<mask_tmp.size();++i)
      mask_tmp[i] = mask_[i];
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        if(x>=2&&x<width_-2&&y>=2&&y<height_-2){ // 2 is half the gauss_mask size
          work_t value = 0.0;
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

template <typename S>
Teuchos::RCP<Image_<S>>
Image_<S>::apply_transformation(Teuchos::RCP<Local_Shape_Function> shape_function,
  const int_t cx,
  const int_t cy,
  const bool apply_in_place){
  Teuchos::RCP<Image_> this_img = Teuchos::rcp(this,false);
  if(apply_in_place){
    Teuchos::RCP<Image_> temp_img = Teuchos::rcp(new Image_(this_img));
    apply_transform(temp_img,this_img,cx,cy,shape_function);
    return Teuchos::null;
  }
  else{
    Teuchos::RCP<Image_> result = Teuchos::rcp(new Image_(width_,height_));
    apply_transform(this_img,result,cx,cy,shape_function);
    return result;
  }
}

template
Teuchos::RCP<Image_<storage_t>>
Image_<storage_t>::apply_transformation(Teuchos::RCP<Local_Shape_Function>,const int_t,const int_t,const bool);
template
Teuchos::RCP<Image_<work_t>>
Image_<work_t>::apply_transformation(Teuchos::RCP<Local_Shape_Function>,const int_t,const int_t,const bool);

template <typename S>
void
Image_<S>::gauss_filter(const int_t mask_size){
  DEBUG_MSG("Image::gauss_filter: mask_size " << gauss_filter_mask_size_);

  if(mask_size>0){
    gauss_filter_mask_size_=mask_size;
    gauss_filter_half_mask_ = gauss_filter_mask_size_/2+1;
  }

  std::vector<work_t> coeffs(13,0.0);

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
        work_t value = 0.0;
        for(int_t i=0;i<gauss_filter_mask_size_;++i){
          for(int_t j=0;j<gauss_filter_mask_size_;++j){
            // assumes intensity values have already been deep copied into intensities_temp_
            value += gauss_filter_coeffs_[i][j]*intensities_temp_[(y+(j-gauss_filter_half_mask_+1))*width_+x+(i-gauss_filter_half_mask_+1)];
          } //j
        } //i
        intensities_[y*width_+x] = static_cast<S>(value);
      }
    }
  }
  has_gauss_filter_ = true;
}

/// returns true if the image is a frame from a video sequence cine or netcdf file
template <typename S>
bool
Image_<S>::is_video_frame()const{
  Image_File_Type type = utils::image_file_type(file_name_.c_str());
  if(type==CINE||type==NETCDF)
    return true;
  else
    return false;
}


template <typename S>
work_t
Image_<S>::diff(Teuchos::RCP<Image_> rhs) const{
  if(rhs->width()!=width_||rhs->height()!=height_)
    return -1.0;
  work_t diff = 0.0;
  work_t diff_ = 0.0;
  for(int_t i=0;i<width_*height_;++i){
    diff_ = (*this)(i) - (*rhs)(i);
    diff += diff_*diff_;
  }
  return std::sqrt(diff);
}

template work_t Image_<storage_t>::diff(Teuchos::RCP<Image_>)const;
template work_t Image_<work_t>::diff(Teuchos::RCP<Image_<work_t>>)const;

/// normalize the image intensity values
template <typename S>
Teuchos::RCP<Image_<S>>
Image_<S>::normalize(const Teuchos::RCP<Teuchos::ParameterList> & params){
  Teuchos::ArrayRCP<S> normalized_intens(width_*height_,0);
  // TODO make the normalization more selective (only include the ROI)
  work_t mean = 0.0;
  int_t num_pixels = 0;
  const int_t buffer = 10;
  for(int_t y=buffer;y<height_-buffer;++y){
    for(int_t x=buffer;x<width_-buffer;++x){
      mean += (*this)(x,y);
      //if(num_pixels<10)
      //std::cout << " x " << x << " y " << y << " " << (*this)(x,y) << " p1 " << (*this)(x-1,y-1) << std::endl;
      num_pixels++;
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(num_pixels<=0,std::runtime_error,"");
  mean/=num_pixels;
  DEBUG_MSG("Image::normalize() mean " << mean << " num pixels in mean " << num_pixels);

  work_t mean_sum = 0.0;
  for(int_t y=buffer;y<height_-buffer;++y){
    for(int_t x=buffer;x<width_-buffer;++x){
      mean_sum += ((*this)(x,y) - mean)*((*this)(x,y) - mean);
    }
  }
  mean_sum = std::sqrt(mean_sum);
  DEBUG_MSG("Image::normalize() mean sum " << mean_sum);
  TEUCHOS_TEST_FOR_EXCEPTION(mean_sum==0.0,std::runtime_error,
    "Error, mean sum should not be zero");
  for(int_t i=0;i<width_*height_;++i)
    normalized_intens[i] = static_cast<S>(((*this)(i) - mean) / mean_sum);
   Teuchos::RCP<Image_> result = Teuchos::rcp(new Image_(width_,height_,normalized_intens,params));
   return result;
}

template <typename S>
work_t
Image_<S>::mean()const{
  work_t mean_value = 0.0;
  for(int_t i=0;i<width_*height_;++i)
    mean_value += (*this)(i);
  return mean_value / (width_*height_);
}

template work_t Image_<storage_t>::mean()const;
template work_t Image_<work_t>::mean()const;

template <typename S>
void
Image_<S>::write(const std::string & file_name){
  try{
    utils::write_image(file_name.c_str(),width_,height_,intensities().getRawPtr(),default_is_layout_right());
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, write image failure.");
  }
}

template void Image_<storage_t>::write(const std::string &);
template void Image_<work_t>::write(const std::string &);

template <typename S>
void
Image_<S>::write_overlap_image(const std::string & file_name,
  Teuchos::RCP<Image_> top_img){
  TEUCHOS_TEST_FOR_EXCEPTION(top_img->height()!=height_||top_img->width()!=width_,std::runtime_error,"Error, dimensions must match for top and bottom image");
  try{
    utils::write_color_overlap_image(file_name.c_str(),width_,height_,intensities().getRawPtr(),top_img->intensities().getRawPtr());
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, write color overlap image failure.");
  }
}

template void Image_<storage_t>::write_overlap_image(const std::string &, Teuchos::RCP<Image_>);
template void Image_<work_t>::write_overlap_image(const std::string &, Teuchos::RCP<Image_>);

template <typename S>
void
Image_<S>::write_grad_x(const std::string & file_name){
  try{
    utils::write_image(file_name.c_str(),width_,height_,grad_x_array().getRawPtr(),default_is_layout_right());
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, write image grad_x failure.");
  }
}

template void Image_<storage_t>::write_grad_x(const std::string & file_name);
template void Image_<work_t>::write_grad_x(const std::string & file_name);

template <typename S>
void
Image_<S>::write_grad_y(const std::string & file_name){
  try{
    utils::write_image(file_name.c_str(),width_,height_,grad_y_array().getRawPtr(),default_is_layout_right());
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, write image grad_y failure.");
  }
}

template void Image_<storage_t>::write_grad_y(const std::string & file_name);
template void Image_<work_t>::write_grad_y(const std::string & file_name);

template <typename S>
Teuchos::RCP<Image_<S>>
Image_<S>::apply_rotation(const Rotation_Value rotation,
  const Teuchos::RCP<Teuchos::ParameterList> & params){
  // don't re-filter images that have already been filtered:
  if(params!=Teuchos::null){
    params->set(DICe::gauss_filter_images,false);
  }
  Teuchos::RCP<Image_> result;
  Teuchos::ArrayRCP<S> new_intensities(width_*height_,0);
  if(rotation==NINTY_DEGREES){
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        new_intensities[(width_-1-x)*height_+y] = (*this)(x,y);
      }
    }
    // note the height and width are swapped in the constructor call on purpose due to the transformation
    result = Teuchos::rcp(new Image_(height_,width_,new_intensities,params));
  }else if(rotation==ONE_HUNDRED_EIGHTY_DEGREES){
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        new_intensities[y*width_+x] = (*this)(width_-1-x,height_-1-y);
      }
    }
    result = Teuchos::rcp(new Image_(width_,height_,new_intensities,params));
  }else if(rotation==TWO_HUNDRED_SEVENTY_DEGREES){
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        new_intensities[x*height_+(height_-1-y)] = (*this)(x,y);
      }
    }
    // note the height and width are swapped in the constructor call on purpose due to the transformation
    result = Teuchos::rcp(new Image_(height_,width_,new_intensities,params));
  }else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Error, unknown rotation requested.");
  }
  return result;
}

template Teuchos::RCP<Image_<storage_t>> Image_<storage_t>::apply_rotation(const Rotation_Value,const Teuchos::RCP<Teuchos::ParameterList> &);
template Teuchos::RCP<Image_<work_t>> Image_<work_t>::apply_rotation(const Rotation_Value,const Teuchos::RCP<Teuchos::ParameterList> &);

}// End DICe Namespace
