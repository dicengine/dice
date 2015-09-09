// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact:
//              Dan Turner   (danielzturner@gmail.com)
//
// ************************************************************************
// @HEADER

#include <DICe_Image.h>
#include <DICe_Tiff.h>

#include <cassert>

namespace DICe {

Image::Image(const std::string & file_name,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  offset_x_(0),
  offset_y_(0),
  has_gradients_(false)
{
  // get the image dims
  read_tiff_image_dimensions(file_name,width_,height_);
  assert(width_>0);
  assert(height_>0);
  // initialize the pixel containers
  intensities_ = intensity_2d_t("intensities",height_,width_);
  // read in the image
  read_tiff_image(file_name,
    intensities_.h_view.ptr_on_device(),
    default_is_layout_right());
  // copy the image to the device (no-op for OpenMP)
  default_constructor_tasks(params);
}

Image::Image(const std::string & file_name,
  const size_t offset_x,
  const size_t offset_y,
  const size_t width,
  const size_t height,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  offset_x_(offset_x),
  offset_y_(offset_y),
  width_(width),
  height_(height),
  has_gradients_(false)
{
  // get the image dims
  size_t img_width = 0;
  size_t img_height = 0;
  read_tiff_image_dimensions(file_name,img_width,img_height);
  assert(width_>0&&offset_x_+width_<img_width);
  assert(height_>0&&offset_y_+height_<img_height);
  // initialize the pixel containers
  intensities_ = intensity_2d_t("intensities",height_,width_);
  // read in the image
  read_tiff_image(file_name,
    offset_x,offset_y,
    width_,height_,
    intensities_.h_view.ptr_on_device(),
    default_is_layout_right());
  // copy the image to the device (no-op for OpenMP)
  default_constructor_tasks(params);
}

Image::Image(intensity_t * intensities,
  const size_t width,
  const size_t height,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  width_(width),
  height_(height),
  offset_x_(0),
  offset_y_(0),
  has_gradients_(false)
{
  assert(width_>0);
  assert(height_>0);
  // initialize the pixel containers
  // if the default layout is LayoutRight, then no permutation is necessary
  if(default_is_layout_right()){
    intensity_device_view_t intensities_dev(intensities,height_,width_);
    intensity_host_view_t intensities_host  = Kokkos::create_mirror_view(intensities_dev);
    intensities_ = intensity_2d_t(intensities_dev,intensities_host);
  }
  // else the data has to be copied to coalesce with the default layout
  else{
    assert(default_is_layout_left());
    intensities_ = intensity_2d_t("intensities",height_,width_);
    for(size_t y=0;y<height_;++y){
      for(size_t x=0;x<width_;++x){
        intensities_.h_view(y,x) = intensities[y*width_+x];
      }
    }
  }
  default_constructor_tasks(params);
}

void
Image::default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){
  grad_x_ = scalar_2d_t("grad_x",height_,width_);
  grad_y_ = scalar_2d_t("grad_y",height_,width_);
  // copy the image to the device (no-op for OpenMP)
  intensities_.modify<host_space>(); // The template is where the modification took place
  intensities_.sync<device_space>(); // The template is what needs to be synced
  // image gradient coefficients
  grad_c1_ = 1.0/12.0;
  grad_c2_ = -8.0/12.0;
  const bool compute_image_gradients = params!=Teuchos::null ?
      params->get<bool>(DICe::compute_image_gradients,false) : false;
  if(compute_image_gradients)
    compute_gradients();
}

void
Image::write(const std::string & file_name){
  write_tiff_image(file_name,width_,height_,intensities_.h_view.ptr_on_device(),default_is_layout_right());
}

KOKKOS_INLINE_FUNCTION
void
Image::operator()(const Grad_Flat_Tag &, const size_t pixel_index)const{
  const size_t y = pixel_index / width_;
  const size_t x = pixel_index - y*width_;
  const scalar_t grad_c1_ = 1.0/12.0;
  const scalar_t grad_c2_ = -8/12.0;
  if(x<2){
    grad_x_.d_view(y,x) = intensities_.d_view(y,x+1) - intensities_.d_view(y,x);
  }
  /// check if this pixel is near the right edge
  else if(x>=width_-2){
    grad_x_.d_view(y,x) = intensities_.d_view(y,x) - intensities_.d_view(y,x-1);
  }
  else{
    grad_x_.d_view(y,x) = grad_c1_*intensities_.d_view(y,x-2) + grad_c2_*intensities_.d_view(y,x-1)
        - grad_c2_*intensities_.d_view(y,x+1) - grad_c1_*intensities_.d_view(y,x+2);
  }
  /// check if this pixel is near the top edge
  if(y<2){
    grad_y_.d_view(y,x) = intensities_.d_view(y+1,x) - intensities_.d_view(y,x);
  }
  /// check if this pixel is near the bottom edge
  else if(y>=height_-2){
    grad_y_.d_view(y,x) = intensities_.d_view(y,x) - intensities_.d_view(y-1,x);
  }
  else{
    grad_y_.d_view(y,x) = grad_c1_*intensities_.d_view(y-2,x) + grad_c2_*intensities_.d_view(y-1,x)
        - grad_c2_*intensities_.d_view(y+1,x) - grad_c1_*intensities_.d_view(y+2,x);
  }
}

void
Image::compute_gradients(){
  Kokkos::parallel_for(Kokkos::RangePolicy<Grad_Flat_Tag>(0,num_pixels()),*this);
  grad_x_.modify<device_space>();
  grad_x_.sync<host_space>();
  grad_y_.modify<device_space>();
  grad_y_.sync<host_space>();
  has_gradients_ = true;
}

}// End DICe Namespace
