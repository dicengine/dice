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

#include <DICe_Image.h>
#include <DICe_Tiff.h>
#include <DICe_Rawi.h>
#include <DICe_Jpeg.h>
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
  file_name_("(from raw array)")
{
  initialize_array_image(intensities);
  default_constructor_tasks(params);
}

Image::Image(const int_t width,
  const int_t height,
  Teuchos::ArrayRCP<intensity_t> intensities,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  width_(width),
  height_(height),
  offset_x_(0),
  offset_y_(0),
  intensity_rcp_(intensities),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_("(from array)")
{
  initialize_array_image(intensities.getRawPtr());
  default_constructor_tasks(params);
}

// TODO add an option to normalize this by intensity mean
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
Image::write_tiff(const std::string & file_name){
  utils::write_tiff_image(file_name.c_str(),width_,height_,intensities().getRawPtr(),default_is_layout_right());
}

void
Image::write_jpeg(const std::string & file_name){
  utils::write_jpeg_image(file_name.c_str(),width_,height_,intensities().getRawPtr(),default_is_layout_right());
}

void
Image::write_rawi(const std::string & file_name){
  utils::write_rawi_image(file_name.c_str(),width_,height_,intensities().getRawPtr(),default_is_layout_right());
}

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
