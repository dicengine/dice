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
  intensities_dev_ = intensity_device_view_t("intensities_dev",height_,width_);
  intensities_host_ = Kokkos::create_mirror_view(intensities_dev_);
  grad_x_dev_ = scalar_device_view_2d_t("grad_x_dev",height_,width_);
  grad_y_dev_ = scalar_device_view_2d_t("grad_y_dev",height_,width_);
  grad_x_host_ = Kokkos::create_mirror_view(grad_x_dev_);
  grad_y_host_ = Kokkos::create_mirror_view(grad_y_dev_);
  // read in the image
  read_tiff_image(file_name,intensities_host_);
  // copy the image to the device (no-op for OpenMP)
  Kokkos::deep_copy(intensities_dev_,intensities_host_);
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
  assert(offset_x>=0);
  assert(offset_y>=0);
  // get the image dims
  size_t img_width = 0;
  size_t img_height = 0;
  read_tiff_image_dimensions(file_name,img_width,img_height);
  assert(width_>0&&offset_x_+width_<img_width);
  assert(height_>0&&offset_y_+height_<img_height);
  // initialize the pixel containers
  intensities_dev_ = intensity_device_view_t("intensities_dev",height_,width_);
  intensities_host_ = Kokkos::create_mirror_view(intensities_dev_);
  grad_x_dev_ = scalar_device_view_2d_t("grad_x_dev",height_,width_);
  grad_y_dev_ = scalar_device_view_2d_t("grad_y_dev",height_,width_);
  grad_x_host_ = Kokkos::create_mirror_view(grad_x_dev_);
  grad_y_host_ = Kokkos::create_mirror_view(grad_y_dev_);
  // read in the image
  read_tiff_image(file_name,offset_x,offset_y,width_,height_,intensities_host_);
  // copy the image to the device (no-op for OpenMP)
  Kokkos::deep_copy(intensities_dev_,intensities_host_);
  const bool compute_image_gradients = params!=Teuchos::null ?
      params->get<bool>(DICe::compute_image_gradients,false) : false;
  if(compute_image_gradients)
    compute_gradients();
}

Image::Image(scalar_t * intensities,
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
  intensities_dev_ = intensity_device_view_t(intensities,height_,width_);
  intensities_host_ = Kokkos::create_mirror_view(intensities_dev_);
  grad_x_dev_ = scalar_device_view_2d_t("grad_x_dev",height_,width_);
  grad_y_dev_ = scalar_device_view_2d_t("grad_y_dev",height_,width_);
  grad_x_host_ = Kokkos::create_mirror_view(grad_x_dev_);
  grad_y_host_ = Kokkos::create_mirror_view(grad_y_dev_);
  // assumes that the host array passed in to the constructor is already populated
  // copy the image to the device (no-op for OpenMP)
  Kokkos::deep_copy(intensities_dev_,intensities_host_);
  const bool compute_image_gradients = params!=Teuchos::null ?
      params->get<bool>(DICe::compute_image_gradients,false) : false;
  if(compute_image_gradients)
    compute_gradients();
}

void
Image::write(const std::string & file_name){
  write_tiff_image(file_name,width_,height_,intensities_host_);
}

void
Image::compute_gradients(){
  // coefficients used to compute image gradients
  // five point finite difference stencil:
  static scalar_t c1 = 1.0/12.0;
  static scalar_t c2 = -8.0/12.0;
  static scalar_t c4 = -c2;
  static scalar_t c5 = -c1;

  // lambda for computing the gradients
  const unsigned int team_size = 256;
  const unsigned int num_teams = height_;

  // TODO make the image intensity array random access
  // TODO better FD along the edges
  Kokkos::parallel_for(num_pixels(),
    KOKKOS_LAMBDA (const size_t i) {
    // get the x and y coordinates of this point
    const size_t y = i / width_;
    const size_t x = i - y*width_;
    /// check if this pixel is near the edges
    if(x<2){
      grad_x_dev_(y,x) = intensities_dev_(y,x+1) - intensities_dev_(y,x);
    }
    else if(x>=width_-2){
      grad_x_dev_(y,x) = intensities_dev_(y,x) - intensities_dev_(y,x-1);
    }
    else{
      grad_x_dev_(y,x) = c1*intensities_dev_(y,x-2) + c2*intensities_dev_(y,x-1) + c4*intensities_dev_(y,x+1) + c5*intensities_dev_(y,x+2);
    }
    if(y<2){
      grad_y_dev_(y,x) = intensities_dev_(y+1,x) - intensities_dev_(y,x);
    }
    else if(y>=height_-2){
      grad_y_dev_(y,x) = intensities_dev_(y,x) - intensities_dev_(y-1,x);
    }
    else{
      grad_y_dev_(y,x) = c1*intensities_dev_(y-2,x) + c2*intensities_dev_(y-1,x) + c4*intensities_dev_(y+1,x) + c5*intensities_dev_(y+2,x);
    }
  });
  Kokkos::deep_copy(intensities_host_,intensities_dev_);

  has_gradients_ = true;
}


}// End DICe Namespace
