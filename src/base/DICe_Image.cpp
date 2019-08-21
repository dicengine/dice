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
#include <DICe_ImageIO.h>
#include <DICe_Shape.h>
#if DICE_KOKKOS
  #include <DICe_Kokkos.h>
#endif

#include <Teuchos_ParameterList.hpp>

#include <cassert>

namespace DICe {

Image::Image(intensity_t * intensities,
  const int_t width,
  const int_t height,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  width_(width),
  height_(height),
  offset_x_(0),
  offset_y_(0),
  intensity_rcp_(Teuchos::null),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_("(from raw array)"),
  has_file_name_(false),
  gradient_method_(FINITE_DIFFERENCE)
{
  initialize_array_image(intensities);
  default_constructor_tasks(params);
}

Image::Image(const int_t width,
  const int_t height,
  Teuchos::ArrayRCP<intensity_t> intensities,
  const Teuchos::RCP<Teuchos::ParameterList> & params,
  const int_t offset_x,
  const int_t offset_y):
  width_(width),
  height_(height),
  offset_x_(offset_x),
  offset_y_(offset_y),
  intensity_rcp_(intensities),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_("(from array)"),
  has_file_name_(false),
  gradient_method_(FINITE_DIFFERENCE)
{
  initialize_array_image(intensities.getRawPtr());
  default_constructor_tasks(params);
}

/// post allocation tasks
void
Image::post_allocation_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){
  gauss_filter_mask_size_ = 7; // default sizes
  gauss_filter_half_mask_ = 4;
  if(params==Teuchos::null) return;
  gradient_method_ = params->get<Gradient_Method>(DICe::gradient_method,FINITE_DIFFERENCE);
  const bool gauss_filter_image =  params->get<bool>(DICe::gauss_filter_images,false);
  const bool gauss_filter_use_hierarchical_parallelism = params->get<bool>(DICe::gauss_filter_use_hierarchical_parallelism,false);
  const int gauss_filter_team_size = params->get<int>(DICe::gauss_filter_team_size,256);
  gauss_filter_mask_size_ = params->get<int>(DICe::gauss_filter_mask_size,7);
  gauss_filter_half_mask_ = gauss_filter_mask_size_/2+1;
  if(gauss_filter_image){
    gauss_filter(-1,gauss_filter_use_hierarchical_parallelism,gauss_filter_team_size);
  }
  const bool compute_image_gradients = params->get<bool>(DICe::compute_image_gradients,false);
  DEBUG_MSG("Image::post_allocation_tasks(): compute_image_gradients is " << compute_image_gradients);
  const bool image_grad_use_hierarchical_parallelism = params->get<bool>(DICe::image_grad_use_hierarchical_parallelism,false);
  const int image_grad_team_size = params->get<int>(DICe::image_grad_team_size,256);
  if(compute_image_gradients)
    compute_gradients(image_grad_use_hierarchical_parallelism,image_grad_team_size);
  if(params->isParameter(DICe::compute_laplacian_image)){
    if(params->get<bool>(DICe::compute_laplacian_image)==true){
      TEUCHOS_TEST_FOR_EXCEPTION(laplacian_==Teuchos::null,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(laplacian_.size()!=width_*height_,std::runtime_error,"");
      Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
      imgParams->set(DICe::compute_image_gradients,true); // automatically compute the gradients if the ref image is changed
      Teuchos::RCP<Image> grad_x_img = Teuchos::rcp(new Image(width_,height_,grad_x_,imgParams));
      Teuchos::RCP<Image> grad_y_img = Teuchos::rcp(new Image(width_,height_,grad_y_,imgParams));
      for(int_t y=0;y<height_;++y){
        for(int_t x=0;x<width_;++x){
          laplacian_[y*width_ + x] = grad_x_img->grad_x(x,y) + grad_y_img->grad_y(x,y);
        }
      }
    }
  }
}

/// returns true if the image is a frame from a video sequence cine or netcdf file
bool
Image::is_video_frame()const{
  Image_File_Type type = utils::image_file_type(file_name_.c_str());
  if(type==CINE||type==NETCDF)
    return true;
  else
    return false;
}


scalar_t
Image::diff(Teuchos::RCP<Image> rhs) const{
  if(rhs->width()!=width_||rhs->height()!=height_)
    return -1.0;
  scalar_t diff = 0.0;
  scalar_t diff_ = 0.0;
  for(int_t i=0;i<width_*height_;++i){
    diff_ = (*this)(i) - (*rhs)(i);
    diff += diff_*diff_;
  }
  return std::sqrt(diff);
}

/// normalize the image intensity values
Teuchos::RCP<Image>
Image::normalize(const Teuchos::RCP<Teuchos::ParameterList> & params){
  Teuchos::ArrayRCP<intensity_t> normalized_intens(width_*height_,0.0);
  // TODO make the normalization more selective (only include the ROI)
  intensity_t mean = 0.0;
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

  intensity_t mean_sum = 0.0;
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
    normalized_intens[i] = ((*this)(i) - mean) / mean_sum;
   Teuchos::RCP<Image> result = Teuchos::rcp(new Image(width_,height_,normalized_intens,params));
   return result;
}

void
Image::replace_intensities(Teuchos::ArrayRCP<intensity_t> intensities){
  assert(intensities.size()==width_*height_);
  intensity_rcp_ = intensities; // copy the pointer so the arrayRCP doesn't get deallocated;
  // automatically re-compute the image gradients:
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::compute_image_gradients,true);
  initialize_array_image(intensities.getRawPtr());
#if DICE_KOKKOS
  intensities_.modify<host_space>(); // The template is where the modification took place
  intensities_.sync<device_space>(); // The template is what needs to be synced
#endif
  default_constructor_tasks(params);
}

scalar_t
Image::mean()const{
  scalar_t mean_value = 0.0;
  for(int_t i=0;i<width_*height_;++i)
    mean_value += (*this)(i);
  return mean_value / (width_*height_);
}

void
Image::write(const std::string & file_name){
  try{
    utils::write_image(file_name.c_str(),width_,height_,intensities().getRawPtr(),default_is_layout_right());
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, write image failure.");
  }
}

void
Image::write_overlap_image(const std::string & file_name,
  Teuchos::RCP<Image> top_img){
  TEUCHOS_TEST_FOR_EXCEPTION(top_img->height()!=height_||top_img->width()!=width_,std::runtime_error,"Error, dimensions must match for top and bottom image");
  try{
    utils::write_color_overlap_image(file_name.c_str(),width_,height_,intensities().getRawPtr(),top_img->intensities().getRawPtr());
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, write color overlap image failure.");
  }
}


#if DICE_KOKKOS
#else
void
Image::write_grad_x(const std::string & file_name){
  try{
    utils::write_image(file_name.c_str(),width_,height_,grad_x_array().getRawPtr(),default_is_layout_right());
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, write image grad_x failure.");
  }
}

void
Image::write_grad_y(const std::string & file_name){
  try{
    utils::write_image(file_name.c_str(),width_,height_,grad_y_array().getRawPtr(),default_is_layout_right());
  }
  catch(...){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, write image grad_y failure.");
  }
}
#endif

Teuchos::RCP<Image>
Image::apply_rotation(const Rotation_Value rotation,
  const Teuchos::RCP<Teuchos::ParameterList> & params){
  // don't re-filter images that have already been filtered:
  if(params!=Teuchos::null){
    params->set(DICe::gauss_filter_images,false);
  }
  Teuchos::RCP<Image> result;
  Teuchos::ArrayRCP<intensity_t> new_intensities(width_*height_,0.0);
  if(rotation==NINTY_DEGREES){
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        new_intensities[(width_-1-x)*height_+y] = (*this)(x,y);
      }
    }
    // note the height and width are swapped in the constructor call on purpose due to the transformation
    result = Teuchos::rcp(new Image(height_,width_,new_intensities,params));
  }else if(rotation==ONE_HUNDRED_EIGHTY_DEGREES){
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        new_intensities[y*width_+x] = (*this)(width_-1-x,height_-1-y);
      }
    }
    result = Teuchos::rcp(new Image(width_,height_,new_intensities,params));
  }else if(rotation==TWO_HUNDRED_SEVENTY_DEGREES){
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        new_intensities[x*height_+(height_-1-y)] = (*this)(x,y);
      }
    }
    // note the height and width are swapped in the constructor call on purpose due to the transformation
    result = Teuchos::rcp(new Image(height_,width_,new_intensities,params));
  }else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Error, unknown rotation requested.");
  }
  return result;
}



}// End DICe Namespace
