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

#include <DICe_Subset.h>
#include <DICe_Tiff.h>

#include <cassert>

namespace DICe {

Subset::Subset(size_t cx,
  size_t cy,
  Teuchos::ArrayRCP<size_t> x,
  Teuchos::ArrayRCP<size_t> y):
  num_pixels_(x.size()),
  cx_(cx),
  cy_(cy)
{
  assert(num_pixels_>0);
  assert(x.size()==y.size());

  // initialize the coordinate views
  size_1d_device_view_t x_dev(x.getRawPtr(),num_pixels_);
  size_1d_host_view_t x_host  = Kokkos::create_mirror_view(x_dev);
  x_ = size_1d_t(x_dev,x_host);
  size_1d_device_view_t y_dev(y.getRawPtr(),num_pixels_);
  size_1d_host_view_t y_host  = Kokkos::create_mirror_view(y_dev);
  y_ = size_1d_t(y_dev,y_host);
  // copy the x and y coords image to the device (no-op for OpenMP)
  // once the x and y views are synced
  // this information does not need to change for the life of the subset
  x_.modify<host_space>();
  y_.modify<host_space>();
  x_.sync<device_space>();
  y_.sync<device_space>();

  // initialize the pixel containers
  // the values will get populated later
  ref_intensities_ = intensity_1d_t("ref_intensities",num_pixels_);
  def_intensities_ = intensity_1d_t("def_intensities",num_pixels_);

}

Subset::Subset(const size_t cx,
  const size_t cy,
  const size_t width,
  const size_t height):
 cx_(cx),
 cy_(cy)
{
  assert(width>0);
  assert(height>0);
  const size_t half_width = width/2;
  const size_t half_height = height/2;
  // if the width and height arguments are not odd, the next larges odd size is used:
  num_pixels_ = (2*half_width+1)*(2*half_height+1);
  assert(num_pixels_>0);

  // initialize the coordinate views
  x_ = size_1d_t("x",num_pixels_);
  y_ = size_1d_t("y",num_pixels_);
  size_t index = 0;
  for(size_t y=cy_-half_height;y<=cy_+half_height;++y){
    for(size_t x=cx_-half_width;x<=cx_+half_width;++x){
      x_.h_view(index) = x;
      y_.h_view(index) = y;
      index++;
    }
  }
  // copy the x and y coords image to the device (no-op for OpenMP)
  // once the x and y views are synced
  // this information does not need to change for the life of the subset
  x_.modify<host_space>();
  y_.modify<host_space>();
  x_.sync<device_space>();
  y_.sync<device_space>();

  // initialize the pixel container
  // the values will get populated later
  ref_intensities_ = intensity_1d_t("ref_intensities",num_pixels_);
  def_intensities_ = intensity_1d_t("def_intensities",num_pixels_);
}

void
Subset::write_tif(const std::string & file_name,
  const bool use_def_intensities){
  // determine the extents of the subset and the offsets
  size_t max_x = 0;
  size_t min_x = x_.h_view(0);
  size_t max_y = 0;
  size_t min_y = y_.h_view(0);
  for(size_t i=0;i<num_pixels_;++i){
    if(x_.h_view(i) > max_x) max_x = x_.h_view(i);
    if(x_.h_view(i) < min_x) min_x = x_.h_view(i);
    if(y_.h_view(i) > max_y) max_y = y_.h_view(i);
    if(y_.h_view(i) < min_y) min_y = y_.h_view(i);
  }
  //create a square image that fits the extents of the subet
  const size_t w = max_x - min_x + 1;
  const size_t h = max_y - min_y + 1;
  intensity_t * intensities = new intensity_t[w*h];
  for(size_t i=0;i<w*h;++i)
    intensities[i] = 0.0;
  for(size_t i=0;i<num_pixels_;++i)
    intensities[(y_.h_view(i)-min_y)*w+(x_.h_view(i)-min_x)] = use_def_intensities ?
        def_intensities_.h_view(i) : ref_intensities_.h_view(i);
  write_tiff_image(file_name,w,h,intensities,true);
  delete[] intensities;
}

void
Subset::initialize(Teuchos::RCP<Image> image,
  Teuchos::RCP<Def_Map> map,
  const Subset_Init_Mode init_mode){
  const Subset_Init_Functor init_functor(this,image.getRawPtr(),map,init_mode);
  // assume if the map is null, use the no_map_tag in the parrel for call of the functor
  if(map==Teuchos::null){
    Kokkos::parallel_for(Kokkos::RangePolicy<Subset_Init_Functor::No_Map_Tag>(0,num_pixels_),init_functor);
  }
  else{
    Kokkos::parallel_for(Kokkos::RangePolicy<Subset_Init_Functor::Map_Bilinear_Tag>(0,num_pixels_),init_functor);
  }
  // now sync up the intensities:
  if(init_mode==FILL_REF_INTENSITIES){
    ref_intensities_.modify<device_space>();
    ref_intensities_.sync<host_space>();
  }
  else{
    def_intensities_.modify<device_space>();
    def_intensities_.sync<host_space>();
  }
}

// functor to populate the values of a subset
// by using the mapping
KOKKOS_INLINE_FUNCTION
void
Subset_Init_Functor::operator()(const Map_Bilinear_Tag&, const size_t pixel_index)const{
  subset_intensities_(pixel_index) = 1.0;
  // map the point to the def
  const scalar_t dx = x_(pixel_index) - cx_;
  const scalar_t dy = y_(pixel_index) - cy_;
  const scalar_t Dx = (1.0+ex_)*dx + g_*dy;
  const scalar_t Dy = (1.0+ey_)*dy + g_*dx;
  // mapped location
  scalar_t mapped_x = cos_t_*Dx - sin_t_*Dy + u_ + cx_;
  scalar_t mapped_y = sin_t_*Dx + cos_t_*Dy + v_ + cy_;
  // check that the mapped location is inside the image...
  if(mapped_x>=0&&mapped_x<image_w_-2&&mapped_y>=0&&mapped_y<image_h_-2){

    size_t x1 = (size_t)mapped_x;
    size_t x2 = x1+1;
    size_t y2 = (size_t)mapped_y;
    size_t y1  = y2+1;
    subset_intensities_(pixel_index) =
        (-image_intensities_(x1,y1)*(x2-mapped_x)*(y2-mapped_y)
         -image_intensities_(x2,y1)*(mapped_x-x1)*(y2-mapped_y)
         -image_intensities_(x1,y2)*(x2-mapped_x)*(mapped_y-y1)
         -image_intensities_(x2,y2)*(mapped_x-x1)*(mapped_y-y1));
  }
  else{
    // out of bounds pixels are black
    subset_intensities_(pixel_index) = 0;
  }
}
// functor to populate the values of a subset
// without using the mapping
KOKKOS_INLINE_FUNCTION
void
Subset_Init_Functor::operator()(const No_Map_Tag&, const size_t pixel_index)const{
  subset_intensities_(pixel_index) = image_intensities_(y_(pixel_index),x_(pixel_index));
}


}// End DICe Namespace
