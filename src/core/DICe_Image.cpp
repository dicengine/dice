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
  grad_x_ = scalar_2d_t("grad_x",height_,width_);
  grad_y_ = scalar_2d_t("grad_y",height_,width_);
  // read in the image
  read_tiff_image(file_name,intensities_.h_view.ptr_on_device());
  // copy the image to the device (no-op for OpenMP)
  intensities_.modify<host_space>(); // The template is where the modification took place
  intensities_.sync<execution_space>(); // The template is what needs to be synced
  const bool compute_image_gradients = params!=Teuchos::null ?
      params->get<bool>(DICe::compute_image_gradients,false) : false;
  if(compute_image_gradients)
    compute_gradients();
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
  grad_x_ = scalar_2d_t("grad_x",height_,width_);
  grad_y_ = scalar_2d_t("grad_y",height_,width_);
  // read in the image
  read_tiff_image(file_name,offset_x,offset_y,width_,height_,intensities_.h_view.ptr_on_device());
  // copy the image to the device (no-op for OpenMP)
  intensities_.modify<host_space>(); // The template is where the modification took place
  intensities_.sync<execution_space>(); // The template is what needs to be synced
  const bool compute_image_gradients = params!=Teuchos::null ?
      params->get<bool>(DICe::compute_image_gradients,false) : false;
  if(compute_image_gradients)
    compute_gradients();
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
  intensity_device_view_t intensities_dev(intensities,height_,width_);
  intensity_device_view_t::HostMirror intensities_host(intensities_dev);
  intensities_ = intensity_2d_t(intensities_dev,intensities_host);
  grad_x_ = scalar_2d_t("grad_x",height_,width_);
  grad_y_ = scalar_2d_t("grad_y",height_,width_);
  // assumes that the host array passed in to the constructor is already populated
  // copy the image to the device (no-op for OpenMP)
  intensities_.modify<host_space>(); // The template is where the modification took place
  intensities_.sync<execution_space>(); // The template is what needs to be synced
  const bool compute_image_gradients = params!=Teuchos::null ?
      params->get<bool>(DICe::compute_image_gradients,false) : false;
  if(compute_image_gradients)
    compute_gradients();
}

void
Image::write(const std::string & file_name){
  write_tiff_image(file_name,width_,height_,intensities_.h_view.ptr_on_device());
}

//struct Image_Gradient_Functor {
//  intensity_device_view_t intensities_;
//  scalar_device_view_2d_t grad_x_;
//  scalar_device_view_2d_t grad_y_;
//  size_t width_;
//  size_t height_;
//  const scalar_t c1;
//  const scalar_t c2;
//  Image_Gradient_Functor(const size_t width,
//    const size_t height,
//    intensity_device_view_t intensities,
//    scalar_device_view_2d_t grad_x,
//    scalar_device_view_2d_t grad_y):
//      width_(width),
//      height_(height),
//      intensities_(intensities),
//      grad_x_(grad_x),
//      grad_y_(grad_y),
//      c1(1.0/12.0),
//      c2(-8.0/12.0){}
//  KOKKOS_INLINE_FUNCTION
//  void operator()(const size_t pixel_index) const {
//    const size_t y = pixel_index / width_;
//    const size_t x = pixel_index - y*width_;
//    /// check if this pixel is near the left edge
//    if(x<2){
//      grad_x_(y,x) = intensities_(y,x+1) - intensities_(y,x);
//    }
//    /// check if this pixel is near the right edge
//    else if(x>=width_-2){
//      grad_x_(y,x) = intensities_(y,x) - intensities_(y,x-1);
//    }
//    else{
//      grad_x_(y,x) = c1*intensities_(y,x-2) + c2*intensities_(y,x-1) - c2*intensities_(y,x+1) - c1*intensities_(y,x+2);
//    }
//    /// check if this pixel is near the top edge
//    if(y<2){
//      grad_y_(y,x) = intensities_(y+1,x) - intensities_(y,x);
//    }
//    /// check if this pixel is near the bottom edge
//    else if(y>=height_-2){
//      grad_y_(y,x) = intensities_(y,x) - intensities_(y-1,x);
//    }
//    else{
//      grad_y_(y,x) = c1*intensities_(y-2,x) + c2*intensities_(y-1,x) - c2*intensities_(y+1,x) -c1*intensities_(y+2,x);
//    }
//  }
//};

void
Image::compute_gradients(){
//  Image_Gradient_Functor igf(width_,height_,intensities_dev_,grad_x_dev_,grad_y_dev_);
//  Kokkos::parallel_for(num_pixels(),igf);
////
//  // coefficients used to compute image gradients
//  // five point finite difference stencil:
//  const scalar_t c1 = 1.0/12.0;
//  const scalar_t c2 = -8.0/12.0;
//  const scalar_t c4 = -c2;
//  const scalar_t c5 = -c1;
//
//  // lambda for computing the gradients
//  // TODO make the image intensity array random access
//  // TODO better FD along the edges
//  Kokkos::parallel_for(num_pixels(),
//    KOKKOS_LAMBDA (const size_t i) {
//    // get the x and y coordinates of this point
//    const size_t y = i / width_;
//    const size_t x = i - y*width_;
//    /// check if this pixel is near the left edge
//    if(x<2){
//      grad_x_dev_(y,x) = intensities_dev_(y,x+1) - intensities_dev_(y,x);
//    }
//    /// check if this pixel is near the right edge
//    else if(x>=width_-2){
//      grad_x_dev_(y,x) = intensities_dev_(y,x) - intensities_dev_(y,x-1);
//    }
//    else{
//      grad_x_dev_(y,x) = c1*intensities_dev_(y,x-2) + c2*intensities_dev_(y,x-1) + c4*intensities_dev_(y,x+1) + c5*intensities_dev_(y,x+2);
//    }
//    /// check if this pixel is near the top edge
//    if(y<2){
//      grad_y_dev_(y,x) = intensities_dev_(y+1,x) - intensities_dev_(y,x);
//    }
//    /// check if this pixel is near the bottom edge
//    else if(y>=height_-2){
//      grad_y_dev_(y,x) = intensities_dev_(y,x) - intensities_dev_(y-1,x);
//    }
//    else{
//      grad_y_dev_(y,x) = c1*intensities_dev_(y-2,x) + c2*intensities_dev_(y-1,x) + c4*intensities_dev_(y+1,x) + c5*intensities_dev_(y+2,x);
//    }
//  });
 // Kokkos::deep_copy(intensities_host_,intensities_dev_);

  has_gradients_ = true;
}


}// End DICe Namespace
