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

Subset::Subset(int_t cx,
  int_t cy,
  Teuchos::ArrayRCP<int_t> x,
  Teuchos::ArrayRCP<int_t> y):
  num_pixels_(x.size()),
  cx_(cx),
  cy_(cy)
{
  assert(num_pixels_>0);
  assert(x.size()==y.size());

  // initialize the coordinate views
  pixel_coord_device_view_1d x_dev(x.getRawPtr(),num_pixels_);
  pixel_coord_host_view_1d x_host  = Kokkos::create_mirror_view(x_dev);
  x_ = pixel_coord_dual_view_1d(x_dev,x_host);
  pixel_coord_device_view_1d y_dev(y.getRawPtr(),num_pixels_);
  pixel_coord_host_view_1d y_host  = Kokkos::create_mirror_view(y_dev);
  y_ = pixel_coord_dual_view_1d(y_dev,y_host);
  // copy the x and y coords image to the device (no-op for OpenMP)
  // once the x and y views are synced
  // this information does not need to change for the life of the subset
  x_.modify<host_space>();
  y_.modify<host_space>();
  x_.sync<device_space>();
  y_.sync<device_space>();

  // initialize the pixel containers
  // the values will get populated later
  ref_intensities_ = intensity_dual_view_1d("ref_intensities",num_pixels_);
  def_intensities_ = intensity_dual_view_1d("def_intensities",num_pixels_);
  is_active_ = bool_dual_view_1d("is_active",num_pixels_);
  reset_is_active();
}

Subset::Subset(const int_t cx,
  const int_t cy,
  const int_t width,
  const int_t height):
 cx_(cx),
 cy_(cy)
{
  assert(width>0);
  assert(height>0);
  const int_t half_width = width/2;
  const int_t half_height = height/2;
  // if the width and height arguments are not odd, the next larges odd size is used:
  num_pixels_ = (2*half_width+1)*(2*half_height+1);
  assert(num_pixels_>0);

  // initialize the coordinate views
  x_ = pixel_coord_dual_view_1d("x",num_pixels_);
  y_ = pixel_coord_dual_view_1d("y",num_pixels_);
  int_t index = 0;
  for(int_t y=cy_-half_height;y<=cy_+half_height;++y){
    for(int_t x=cx_-half_width;x<=cx_+half_width;++x){
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
  ref_intensities_ = intensity_dual_view_1d("ref_intensities",num_pixels_);
  def_intensities_ = intensity_dual_view_1d("def_intensities",num_pixels_);
  is_active_ = bool_dual_view_1d("is_active",num_pixels_);
  reset_is_active();
}

Subset::Subset(const int_t cx,
  const int_t cy,
  const Conformal_Area_Def & subset_def):
  cx_(cx),
  cy_(cy)
{
  assert(subset_def.has_boundary());
  std::set<std::pair<int_t,int_t> > coords;
  for(int_t i=0;i<subset_def.boundary()->size();++i){
    std::set<std::pair<int_t,int_t> > shapeCoords = (*subset_def.boundary())[i]->get_owned_pixels();
    coords.insert(shapeCoords.begin(),shapeCoords.end());
  }
  // at this point all the coordinate pairs are in the set
  num_pixels_ = coords.size();

  // resize the storage arrays now that the num_pixels is known
  x_ = pixel_coord_dual_view_1d("x",num_pixels_);
  y_ = pixel_coord_dual_view_1d("y",num_pixels_);
  int_t index = 0;
  // NOTE: the pairs are (y,x) not (x,y) so that the ordering is correct in the set
  typename std::set<std::pair<int_t,int_t> >::iterator set_it = coords.begin();
  for( ; set_it!=coords.end();++set_it){
    x_.h_view(index) = set_it->second;
    y_.h_view(index) = set_it->first;
    index++;
  }
  x_.modify<host_space>();
  y_.modify<host_space>();
  x_.sync<device_space>();
  y_.sync<device_space>();

  // initialize the pixel container
  // the values will get populated later
  ref_intensities_ = intensity_dual_view_1d("ref_intensities",num_pixels_);
  def_intensities_ = intensity_dual_view_1d("def_intensities",num_pixels_);
  is_active_ = bool_dual_view_1d("is_active",num_pixels_);
  reset_is_active();

  // now set the inactive bit for the second set of multishapes if they exist.
  if(subset_def.has_excluded_area()){
    for(int_t i=0;i<subset_def.excluded_area()->size();++i){
      (*subset_def.excluded_area())[i]->deactivate_pixels(num_pixels_,is_active_.h_view.ptr_on_device(),
        x_.h_view.ptr_on_device(),y_.h_view.ptr_on_device());
    }
  }
  is_active_.modify<host_space>();
  is_active_.sync<device_space>();


  // TODO TODO TODO
//
//  if(subset_def.has_obstructed_area()){
//    for(Size i=0;i<subset_def.obstructed_area()->size();++i){
//      std::set<std::pair<Size,Size> > obstructedArea = (*subset_def.obstructed_area())[i]->get_owned_pixels();
//      obstructed_coords_.insert(obstructedArea.begin(),obstructedArea.end());
//    }
//  }
}

void
Subset::reset_is_active(){
  for(int_t i=0;i<num_pixels_;++i)
    is_active_.h_view(i) = true;
  is_active_.modify<host_space>();
  is_active_.sync<device_space>();
}

void
Subset::write_subset_on_image(const std::string & file_name,
  Teuchos::RCP<Image> image,
  Teuchos::RCP<const std::vector<scalar_t> > deformation){
  //create a square image that fits the extents of the subet
  const int_t w = image->width();
  const int_t h = image->height();
  const scalar_t u = (*deformation)[DISPLACEMENT_X];
  const scalar_t v = (*deformation)[DISPLACEMENT_Y];
  const scalar_t t = (*deformation)[ROTATION_Z];
  const scalar_t ex = (*deformation)[NORMAL_STRAIN_X];
  const scalar_t ey = (*deformation)[NORMAL_STRAIN_Y];
  const scalar_t g = (*deformation)[SHEAR_STRAIN_XY];

  intensity_t * intensities = new intensity_t[w*h];
  for(int_t y=0;y<h;++y){
    for(int_t x=0;x<w;++x){
      intensities[y*w+x] = image->intensities().h_view(y,x);
    }
  }
  if(deformation!=Teuchos::null){
    scalar_t dx=0.0,dy=0.0;
    scalar_t Dx=0.0,Dy=0.0;
    scalar_t mapped_x=0.0,mapped_y=0.0;
    int_t px=0,py=0;
    for(int_t i=0;i<num_pixels_;++i){
      // compute the deformed shape:
      // need to cast the x_ and y_ values since the resulting value could be negative
      dx = (scalar_t)(x_.h_view(i)) - cx_;
      dy = (scalar_t)(y_.h_view(i)) - cy_;
      Dx = (1.0+ex)*dx + g*dy;
      Dy = (1.0+ey)*dy + g*dx;
      // mapped location
      mapped_x = std::cos(t)*Dx - std::sin(t)*Dy + u + cx_;
      mapped_y = std::sin(t)*Dx + std::cos(t)*Dy + v + cy_;
      // get the nearest pixel location:
      px = (int_t)mapped_x;
      if(mapped_x - (int_t)mapped_x >= 0.5) px++;
      py = (int_t)mapped_y;
      if(mapped_y - (int_t)mapped_y >= 0.5) py++;
      intensities[py*w+px] = 255;
    }
  }
  else{ // write the original shape of the subset
    for(int_t i=0;i<num_pixels_;++i)
      intensities[y_.h_view(i)*w+x_.h_view(i)] = 255;
  }
  utils::write_tiff_image(file_name.c_str(),w,h,intensities,true);
  delete[] intensities;
}

void
Subset::write_tiff(const std::string & file_name,
  const bool use_def_intensities){
  // determine the extents of the subset and the offsets
  int_t max_x = 0;
  int_t min_x = x_.h_view(0);
  int_t max_y = 0;
  int_t min_y = y_.h_view(0);
  for(int_t i=0;i<num_pixels_;++i){
    if(x_.h_view(i) > max_x) max_x = x_.h_view(i);
    if(x_.h_view(i) < min_x) min_x = x_.h_view(i);
    if(y_.h_view(i) > max_y) max_y = y_.h_view(i);
    if(y_.h_view(i) < min_y) min_y = y_.h_view(i);
  }
  //create a square image that fits the extents of the subet
  const int_t w = max_x - min_x + 1;
  const int_t h = max_y - min_y + 1;
  intensity_t * intensities = new intensity_t[w*h];
  for(int_t i=0;i<w*h;++i)
    intensities[i] = 0.0;
  for(int_t i=0;i<num_pixels_;++i){
    if(!is_active_.h_view(i)){
      intensities[(y_.h_view(i)-min_y)*w+(x_.h_view(i)-min_x)] = 100; // color the inactive areas gray
    }
    else{
      intensities[(y_.h_view(i)-min_y)*w+(x_.h_view(i)-min_x)] = use_def_intensities ?
          def_intensities_.h_view(i) : ref_intensities_.h_view(i);
    }
  }
  utils::write_tiff_image(file_name.c_str(),w,h,intensities,true);
  delete[] intensities;
}

scalar_t
Subset::mean(const Subset_View_Target target){
  scalar_t mean = 0.0;
  if(target==REF_INTENSITIES){
    Intensity_Sum_Functor sum_func(ref_intensities_.d_view);
    Kokkos::parallel_reduce(num_pixels_,sum_func,mean);
  }else{
    Intensity_Sum_Functor sum_func(def_intensities_.d_view);
    Kokkos::parallel_reduce(num_pixels_,sum_func,mean);
  }
  return mean/num_pixels_;
}

scalar_t
Subset::mean(const Subset_View_Target target,
  scalar_t & sum){
  scalar_t mean = 0.0;
  if(target==REF_INTENSITIES){
    Intensity_Sum_Functor sum_func(ref_intensities_.d_view);
    Kokkos::parallel_reduce(num_pixels_,sum_func,mean);
  }else{
    Intensity_Sum_Functor sum_func(def_intensities_.d_view);
    Kokkos::parallel_reduce(num_pixels_,sum_func,mean);
  }
  mean/=num_pixels_;
  sum = 0.0;
  if(target==REF_INTENSITIES){
    Intensity_Sum_Minus_Mean_Functor sum_minus_mean_func(ref_intensities_.d_view,mean);
    Kokkos::parallel_reduce(num_pixels_,sum_minus_mean_func,sum);
  }else{
    Intensity_Sum_Minus_Mean_Functor sum_minus_mean_func(def_intensities_.d_view,mean);
    Kokkos::parallel_reduce(num_pixels_,sum_minus_mean_func,sum);
  }
  sum = std::sqrt(sum);
  return mean;
}

scalar_t
Subset::gamma(){
  // TODO turn off obstructed pixels
  scalar_t mean_sum_ref = 0.0;
  const scalar_t mean_ref = mean(REF_INTENSITIES,mean_sum_ref);
  scalar_t mean_sum_def = 0.0;
  const scalar_t mean_def = mean(DEF_INTENSITIES,mean_sum_def);
  scalar_t gamma = 0.0;
  ZNSSD_Gamma_Functor gamma_func(ref_intensities_.d_view,
     def_intensities_.d_view,
     mean_ref,mean_def,
     mean_sum_ref,mean_sum_def);
  Kokkos::parallel_reduce(num_pixels_,gamma_func,gamma);
  return gamma;
}

void
Subset::initialize(Teuchos::RCP<Image> image,
  const Subset_View_Target target,
  Teuchos::RCP<const std::vector<scalar_t> > deformation,
  const Interpolation_Method interp){
  const Subset_Init_Functor init_functor(this,image,deformation,target);
  // assume if the map is null, use the no_map_tag in the parrel for call of the functor
  if(deformation==Teuchos::null){
    Kokkos::parallel_for(Kokkos::RangePolicy<Subset_Init_Functor::No_Map_Tag>(0,num_pixels_),init_functor);
  }
  else{
    if(interp==BILINEAR)
      Kokkos::parallel_for(Kokkos::RangePolicy<Subset_Init_Functor::Map_Bilinear_Tag>(0,num_pixels_),init_functor);
    else if(interp==KEYS_FOURTH)
      Kokkos::parallel_for(Kokkos::RangePolicy<Subset_Init_Functor::Map_Keys_Tag>(0,num_pixels_),init_functor);
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,
        "Error, unknown interpolation method requested");
    }
  }
  // now sync up the intensities:
  if(target==REF_INTENSITIES){
    ref_intensities_.modify<device_space>();
    ref_intensities_.sync<host_space>();
  }
  else{
    def_intensities_.modify<device_space>();
    def_intensities_.sync<host_space>();
  }
}

// functor to populate the values of a subset
// by using the mapping and bilinear interpolation
KOKKOS_INLINE_FUNCTION
void
Subset_Init_Functor::operator()(const Map_Bilinear_Tag&,
  const int_t pixel_index)const{
  // map the point to the def
  // have to cast x_ and y_ to scalar type since the result could be negative
  const scalar_t dx = (scalar_t)(x_(pixel_index)) - cx_;
  const scalar_t dy = (scalar_t)(y_(pixel_index)) - cy_;
  const scalar_t Dx = (1.0+ex_)*dx + g_*dy;
  const scalar_t Dy = (1.0+ey_)*dy + g_*dx;
  // mapped location
  scalar_t mapped_x = cos_t_*Dx - sin_t_*Dy + u_ + cx_;
  scalar_t mapped_y = sin_t_*Dx + cos_t_*Dy + v_ + cy_;
  // check that the mapped location is inside the image...
  if(mapped_x>=0&&mapped_x<image_w_-2&&mapped_y>=0&&mapped_y<image_h_-2){
    int_t x1 = (int_t)mapped_x;
    int_t x2 = x1+1;
    int_t y1 = (int_t)mapped_y;
    int_t y2  = y1+1;
    subset_intensities_(pixel_index) =
        (image_intensities_(y1,x1)*(x2-mapped_x)*(y2-mapped_y)
         +image_intensities_(y1,x2)*(mapped_x-x1)*(y2-mapped_y)
         +image_intensities_(y2,x2)*(mapped_x-x1)*(mapped_y-y1)
         +image_intensities_(y2,x1)*(x2-mapped_x)*(mapped_y-y1));
  }
  else{
    // out of bounds pixels are black
    subset_intensities_(pixel_index) = 0;
  }
}

// functor to populate the values of a subset
// by using the mapping and Keys fourth-order interpolation
KOKKOS_INLINE_FUNCTION
void
Subset_Init_Functor::operator()(const Map_Keys_Tag&,
  const int_t pixel_index)const{

  // map the point to the def
  // have to cast x_ and y_ to scalar type since the result could be negative
  const scalar_t dx = (scalar_t)(x_(pixel_index)) - cx_;
  const scalar_t dy = (scalar_t)(y_(pixel_index)) - cy_;
  const scalar_t Dx = (1.0+ex_)*dx + g_*dy;
  const scalar_t Dy = (1.0+ey_)*dy + g_*dx;
  // mapped location
  scalar_t mapped_x = cos_t_*Dx - sin_t_*Dy + u_ + cx_;
  scalar_t mapped_y = sin_t_*Dx + cos_t_*Dy + v_ + cy_;
  // check that the mapped location is inside the image...
  if(mapped_x>=3&&mapped_x<image_w_-4&&mapped_y>=3&&mapped_y<image_h_-4){

    // determine the current pixel the coordinates fall in:
    int_t px = (int_t)mapped_x;
    if(mapped_x - px >= 0.5) px++;
    int_t py = (int_t)mapped_y;
    if(mapped_y - py >= 0.5) py++;

    // check if the location is close enough to the pixel location
    // to not need interpolation
    if((mapped_x - px)<tol_&&(mapped_y-py<tol_)){
      subset_intensities_(pixel_index) = image_intensities_(py,px);
    }
    else{
      intensity_t intensity_value = 0.0;
      // convolve all the pixels within + and - pixels of the point in question
      scalar_t dx=0.0, dy=0.0;
      scalar_t dx2=0.0, dx3=0.0;
      scalar_t dy2=0.0, dy3=0.0;
      scalar_t f0x=0.0, f0y=0.0;
      for(int_t y=py-3;y<=py+3;++y){
        dy = std::abs(mapped_y - y);
        dy2=dy*dy;
        dy3=dy2*dy;
        f0y = 0.0;
        if(dy <= 1.0){
          f0y = 1.3333333333*dy3 - 2.3333333333*dy2 + 1.0;
        }
        else if(dy <= 2.0){
          f0y = -0.5833333333*dy3 + 3.0*dy2 - 4.9166666666*dy + 2.5;
        }
        else if(dy <= 3.0){
          f0y = 0.0833333333*dy3 - 0.6666666666*dy2 + 1.75*dy - 1.5;
        }
        for(int_t x=px-3;x<=px+3;++x){
          // compute the f's of x and y
          dx = std::abs(mapped_x - x);
          dx2=dx*dx;
          dx3=dx2*dx;
          f0x = 0.0;
          if(dx <= 1.0){
            f0x = 1.3333333333*dx3 - 2.3333333333*dx2 + 1.0;
          }
          else if(dx <= 2.0){
            f0x = -0.5833333333*dx3 + 3.0*dx2 - 4.9166666666*dx + 2.5;
          }
          else if(dx <= 3.0){
            f0x = 0.0833333333*dx3 - 0.6666666666*dx2 + 1.75*dx - 1.5;
          }
          intensity_value += image_intensities_(y,x)*f0x*f0y;
        }
      }
      subset_intensities_(pixel_index) = intensity_value;
    }
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
Subset_Init_Functor::operator()(const No_Map_Tag&,
  const int_t pixel_index)const{
  subset_intensities_(pixel_index) = image_intensities_(y_(pixel_index),x_(pixel_index));
}


}// End DICe Namespace
