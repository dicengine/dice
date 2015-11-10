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
  cy_(cy),
  has_gradients_(false),
  is_conformal_(false)
{
  assert(num_pixels_>0);
  assert(x.size()==y.size());

#ifdef DICE_KOKKOS
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
  grad_x_ = scalar_dual_view_1d("grad_x",num_pixels_);
  grad_y_ = scalar_dual_view_1d("grad_x",num_pixels_);
  is_active_ = bool_dual_view_1d("is_active",num_pixels_);
  is_deactivated_this_step_ = bool_dual_view_1d("is_deactivated_this_step",num_pixels_);
#else
  x_ = x;
  y_ = y;
  ref_intensities_ = Teuchos::ArrayRCP<intensity_t>(num_pixels_,0.0);
  def_intensities_ = Teuchos::ArrayRCP<intensity_t>(num_pixels_,0.0);
  grad_x_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  grad_y_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  is_active_ = Teuchos::ArrayRCP<bool>(num_pixels_,true);
  is_deactivated_this_step_ = Teuchos::ArrayRCP<bool>(num_pixels_,false);
#endif

  reset_is_active();
  reset_is_deactivated_this_step();
}

Subset::Subset(const int_t cx,
  const int_t cy,
  const int_t width,
  const int_t height):
 cx_(cx),
 cy_(cy),
 has_gradients_(false),
 is_conformal_(false)
{
  assert(width>0);
  assert(height>0);
  const int_t half_width = width/2;
  const int_t half_height = height/2;
  // if the width and height arguments are not odd, the next larges odd size is used:
  num_pixels_ = (2*half_width+1)*(2*half_height+1);
  assert(num_pixels_>0);

#ifdef DICE_KOKKOS
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
  grad_x_ = scalar_dual_view_1d("grad_x",num_pixels_);
  grad_y_ = scalar_dual_view_1d("grad_x",num_pixels_);
  is_active_ = bool_dual_view_1d("is_active",num_pixels_);
  is_deactivated_this_step_ = bool_dual_view_1d("is_deactivated_this_step",num_pixels_);
#else
  x_ = Teuchos::ArrayRCP<int_t>(num_pixels_,0);
  y_ = Teuchos::ArrayRCP<int_t>(num_pixels_,0);
  int_t index = 0;
  for(int_t y=cy_-half_height;y<=cy_+half_height;++y){
    for(int_t x=cx_-half_width;x<=cx_+half_width;++x){
      x_[index] = x;
      y_[index] = y;
      index++;
    }
  }
  ref_intensities_ = Teuchos::ArrayRCP<intensity_t>(num_pixels_,0.0);
  def_intensities_ = Teuchos::ArrayRCP<intensity_t>(num_pixels_,0.0);
  grad_x_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  grad_y_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  is_active_ = Teuchos::ArrayRCP<bool>(num_pixels_,true);
  is_deactivated_this_step_ = Teuchos::ArrayRCP<bool>(num_pixels_,false);
#endif

  reset_is_active();
  reset_is_deactivated_this_step();
}

Subset::Subset(const int_t cx,
  const int_t cy,
  const Conformal_Area_Def & subset_def):
  cx_(cx),
  cy_(cy),
  has_gradients_(false),
  conformal_subset_def_(subset_def),
  is_conformal_(true)
{
  assert(subset_def.has_boundary());
  std::set<std::pair<int_t,int_t> > coords;
  for(size_t i=0;i<subset_def.boundary()->size();++i){
    std::set<std::pair<int_t,int_t> > shapeCoords = (*subset_def.boundary())[i]->get_owned_pixels();
    coords.insert(shapeCoords.begin(),shapeCoords.end());
  }
  // at this point all the coordinate pairs are in the set
  num_pixels_ = coords.size();

#ifdef DICE_KOKKOS
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
  grad_x_ = scalar_dual_view_1d("grad_x",num_pixels_);
  grad_y_ = scalar_dual_view_1d("grad_x",num_pixels_);
  is_active_ = bool_dual_view_1d("is_active",num_pixels_);
  is_deactivated_this_step_ = bool_dual_view_1d("is_deactivated_this_step",num_pixels_);
#else
  x_ = Teuchos::ArrayRCP<int_t>(num_pixels_,0);
  y_ = Teuchos::ArrayRCP<int_t>(num_pixels_,0);
  int_t index = 0;
  // NOTE: the pairs are (y,x) not (x,y) so that the ordering is correct in the set
  typename std::set<std::pair<int_t,int_t> >::iterator set_it = coords.begin();
  for( ; set_it!=coords.end();++set_it){
    x_[index] = set_it->second;
    y_[index] = set_it->first;
    index++;
  }
  ref_intensities_ = Teuchos::ArrayRCP<intensity_t>(num_pixels_,0.0);
  def_intensities_ = Teuchos::ArrayRCP<intensity_t>(num_pixels_,0.0);
  grad_x_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  grad_y_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  is_active_ = Teuchos::ArrayRCP<bool>(num_pixels_,true);
  is_deactivated_this_step_ = Teuchos::ArrayRCP<bool>(num_pixels_,false);
#endif
  reset_is_active();
  reset_is_deactivated_this_step();

  // now set the inactive bit for the second set of multishapes if they exist.
  if(subset_def.has_excluded_area()){
    for(size_t i=0;i<subset_def.excluded_area()->size();++i){
#ifdef DICE_KOKKOS
      (*subset_def.excluded_area())[i]->deactivate_pixels(num_pixels_,is_active_.h_view.ptr_on_device(),
        x_.h_view.ptr_on_device(),y_.h_view.ptr_on_device());
#else
      (*subset_def.excluded_area())[i]->deactivate_pixels(num_pixels_,is_active_,x_,y_);
#endif
    }
  }
#ifdef DICE_KOKKOS
  is_active_.modify<host_space>();
  is_active_.sync<device_space>();
#endif
  if(subset_def.has_obstructed_area()){
    for(size_t i=0;i<subset_def.obstructed_area()->size();++i){
      std::set<std::pair<int_t,int_t> > obstructedArea = (*subset_def.obstructed_area())[i]->get_owned_pixels();
      obstructed_coords_.insert(obstructedArea.begin(),obstructedArea.end());
    }
  }
}

void
Subset::reset_is_active(){
#ifdef DICE_KOKKOS
  for(int_t i=0;i<num_pixels_;++i)
    is_active_.h_view(i) = true;
  is_active_.modify<host_space>();
  is_active_.sync<device_space>();
#else
  for(int_t i=0;i<num_pixels_;++i)
    is_active_[i] = true;
#endif
}

void
Subset::reset_is_deactivated_this_step(){
#ifdef DICE_KOKKOS
  for(int_t i=0;i<num_pixels_;++i)
    is_deactivated_this_step_.h_view(i) = false;
  is_deactivated_this_step_.modify<host_space>();
  is_deactivated_this_step_.sync<device_space>();
#else
  for(int_t i=0;i<num_pixels_;++i)
    is_deactivated_this_step_[i] = false;
#endif
}

bool
Subset::is_obstructed_pixel(const scalar_t & coord_x,
  const scalar_t & coord_y) const {
  // determine which pixel the coordinates fall in:
  int_t c_x = (int_t)coord_x;
  if(coord_x - (int_t)coord_x >= 0.5) c_x++;
  int_t c_y = (int_t)coord_y;
  if(coord_y - (int_t)coord_y >= 0.5) c_y++;
  // now check if c_x and c_y are obstructed
  // Note the x and y coordinates are switched because that is how they live in the set
  // this was done for performance in loops over y then x
  std::pair<int_t,int_t> point(c_y,c_x);
  const bool obstructed = obstructed_coords_.find(point)!=obstructed_coords_.end();
  return obstructed;
}

std::set<std::pair<int_t,int_t> >
Subset::deformed_shapes(Teuchos::RCP<const std::vector<scalar_t> > deformation,
  const int_t cx,
  const int_t cy,
  const scalar_t & skin_factor){
  std::set<std::pair<int_t,int_t> > coords;
  if(!is_conformal_) return coords;
  for(size_t i=0;i<conformal_subset_def_.boundary()->size();++i){
    std::set<std::pair<int_t,int_t> > shapeCoords =
        (*conformal_subset_def_.boundary())[i]->get_owned_pixels(deformation,cx,cy,skin_factor);
    coords.insert(shapeCoords.begin(),shapeCoords.end());
  }
  return coords;
}

void
Subset::turn_off_obstructed_pixels(Teuchos::RCP<const std::vector<scalar_t> > deformation){
  assert(deformation!=Teuchos::null);
  const scalar_t u     = (*deformation)[DICe::DISPLACEMENT_X];
  const scalar_t v     = (*deformation)[DICe::DISPLACEMENT_Y];
  const scalar_t theta = (*deformation)[DICe::ROTATION_Z];
  const scalar_t dudx  = (*deformation)[DICe::NORMAL_STRAIN_X];
  const scalar_t dvdy  = (*deformation)[DICe::NORMAL_STRAIN_Y];
  const scalar_t gxy   = (*deformation)[DICe::SHEAR_STRAIN_XY];
  scalar_t Dx=0.0,Dy=0.0;
  scalar_t dx=0.0, dy=0.0;
  scalar_t X=0.0,Y=0.0;
  int_t px=0,py=0;
  scalar_t cos_t = std::cos(theta);
  scalar_t sin_t = std::sin(theta);
  reset_is_deactivated_this_step();
  const bool has_blocks = !pixels_blocked_by_other_subsets_.empty();
  for(int_t i=0;i<num_pixels_;++i){

    dx = (scalar_t)(x(i)) - cx_;
    dy = (scalar_t)(y(i)) - cy_;
    Dx = (1.0+dudx)*dx + gxy*dy;
    Dy = (1.0+dvdy)*dy + gxy*dx;
    // mapped location
    X = cos_t*Dx - sin_t*Dy + u + cx_;
    Y = sin_t*Dx + cos_t*Dy + v + cy_;

    if(is_obstructed_pixel(X,Y)){
#ifdef DICE_KOKKOS
      is_deactivated_this_step_.h_view(i) = true;
#else
      is_deactivated_this_step_[i] = true;
#endif
    }
    else{
#ifdef DICE_KOKKOS
      is_deactivated_this_step_.h_view(i) = false;
#else
      is_deactivated_this_step_[i] = false;
#endif
    }
    // pixels blocked by other subsets:
    if(has_blocks){
      px = ((int_t)(X + 0.5) == (int_t)(X)) ? (int_t)(X) : (int_t)(X) + 1;
      py = ((int_t)(Y + 0.5) == (int_t)(Y)) ? (int_t)(Y) : (int_t)(Y) + 1;
#ifdef DICE_KOKKOS
      if(pixels_blocked_by_other_subsets_.find(std::pair<int_t,int_t>(py,px))
          !=pixels_blocked_by_other_subsets_.end())
        is_deactivated_this_step_.h_view(i) = true;
#else
      if(pixels_blocked_by_other_subsets_.find(std::pair<int_t,int_t>(py,px))
          !=pixels_blocked_by_other_subsets_.end())
        is_deactivated_this_step_[i] = true;
#endif
    }
  } // pixel loop
#ifdef DICE_KOKKOS
  is_deactivated_this_step_.modify<host_space>();
  is_deactivated_this_step_.sync<device_space>();
#endif
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
#ifdef DICE_KOKKOS
      dx = (scalar_t)(x_.h_view(i)) - cx_;
      dy = (scalar_t)(y_.h_view(i)) - cy_;
#else
      dx = (scalar_t)(x_[i]) - cx_;
      dy = (scalar_t)(y_[i]) - cy_;
#endif
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
#ifdef DICE_KOKKOS
      intensities[py*w+px] = !is_active_.h_view(i) ? 255
          : is_deactivated_this_step_.h_view(i) ?  0
          : std::abs((def_intensities_.h_view(i) - ref_intensities_.h_view(i))*2);
#else
      intensities[py*w+px] = !is_active_[i] ? 255
          : is_deactivated_this_step_[i] ?  0
          : std::abs((def_intensities_[i] - ref_intensities_[i])*2);
#endif
    }
  }
  else{ // write the original shape of the subset
    for(int_t i=0;i<num_pixels_;++i)
#ifdef DICE_KOKKOS
      intensities[y_.h_view(i)*w+x_.h_view(i)] = 255;
#else
      intensities[y_[i]*w+x_[i]] = 255;
#endif
  }
  utils::write_tiff_image(file_name.c_str(),w,h,intensities,true);
  delete[] intensities;
}

void
Subset::write_tiff(const std::string & file_name,
  const bool use_def_intensities){
  // determine the extents of the subset and the offsets
  int_t max_x = 0;
  int_t max_y = 0;
#ifdef DICE_KOKKOS
  int_t min_x = x_.h_view(0);
  int_t min_y = y_.h_view(0);
  for(int_t i=0;i<num_pixels_;++i){
    if(x_.h_view(i) > max_x) max_x = x_.h_view(i);
    if(x_.h_view(i) < min_x) min_x = x_.h_view(i);
    if(y_.h_view(i) > max_y) max_y = y_.h_view(i);
    if(y_.h_view(i) < min_y) min_y = y_.h_view(i);
  }
#else
  int_t min_x = x_[0];
  int_t min_y = y_[0];
  for(int_t i=0;i<num_pixels_;++i){
    if(x_[i] > max_x) max_x = x_[i];
    if(x_[i] < min_x) min_x = x_[i];
    if(y_[i] > max_y) max_y = y_[i];
    if(y_[i] < min_y) min_y = y_[i];
  }
#endif
  //create a square image that fits the extents of the subet
  const int_t w = max_x - min_x + 1;
  const int_t h = max_y - min_y + 1;
  intensity_t * intensities = new intensity_t[w*h];
  for(int_t i=0;i<w*h;++i)
    intensities[i] = 0.0;
#ifdef DICE_KOKKOS
  for(int_t i=0;i<num_pixels_;++i){
    if(!is_active_.h_view(i)){
      intensities[(y_.h_view(i)-min_y)*w+(x_.h_view(i)-min_x)] = 100; // color the inactive areas gray
    }
    else{
      intensities[(y_.h_view(i)-min_y)*w+(x_.h_view(i)-min_x)] = use_def_intensities ?
          def_intensities_.h_view(i) : ref_intensities_.h_view(i);
    }
  }
#else
  for(int_t i=0;i<num_pixels_;++i){
    if(!is_active_[i]){
      intensities[(y_[i]-min_y)*w+(x_[i]-min_x)] = 100; // color the inactive areas gray
    }
    else{
      intensities[(y_[i]-min_y)*w+(x_[i]-min_x)] = use_def_intensities ?
          def_intensities_[i] : ref_intensities_[i];
    }
  }
#endif

  utils::write_tiff_image(file_name.c_str(),w,h,intensities,true);
  delete[] intensities;
}

scalar_t
Subset::mean(const Subset_View_Target target){
  scalar_t mean = 0.0;
#ifdef DICE_KOKKOS
  if(target==REF_INTENSITIES){
    Intensity_Sum_Functor sum_func(ref_intensities_.d_view);
    Kokkos::parallel_reduce(num_pixels_,sum_func,mean);
  }else{
    Intensity_Sum_Functor sum_func(def_intensities_.d_view);
    Kokkos::parallel_reduce(num_pixels_,sum_func,mean);
  }
#else
  if(target==REF_INTENSITIES){
    for(size_t i=0;i<num_pixels_;++i)
      mean += ref_intensities_[i];
  }else{
    for(size_t i=0;i<num_pixels_;++i)
      mean += def_intensities_[i];
  }
#endif
  return mean/num_pixels_;
}

scalar_t
Subset::mean(const Subset_View_Target target,
  scalar_t & sum){
  scalar_t mean_ = mean(target);
  sum = 0.0;
#ifdef DICE_KOKKOS
  if(target==REF_INTENSITIES){
    Intensity_Sum_Minus_Mean_Functor sum_minus_mean_func(ref_intensities_.d_view,mean_);
    Kokkos::parallel_reduce(num_pixels_,sum_minus_mean_func,sum);
  }else{
    Intensity_Sum_Minus_Mean_Functor sum_minus_mean_func(def_intensities_.d_view,mean_);
    Kokkos::parallel_reduce(num_pixels_,sum_minus_mean_func,sum);
  }
#else
  if(target==REF_INTENSTITIES){
    for(size_t i=0;i<num_pixels_;++i)
      sum += (ref_intensities_[i]-mean_)*(ref_intensities_[i]-mean_);
  }else{
    for(size_t i=0;i<num_pixels_;++i)
      sum += (def_intensities_[i]-mean_)*(def_intensities_[i]-mean_);
  }
#endif
  sum = std::sqrt(sum);
  return mean_;
}

scalar_t
Subset::gamma(){
  // assumes obstructed pixels are already turned off
  scalar_t mean_sum_ref = 0.0;
  const scalar_t mean_ref = mean(REF_INTENSITIES,mean_sum_ref);
  scalar_t mean_sum_def = 0.0;
  const scalar_t mean_def = mean(DEF_INTENSITIES,mean_sum_def);
  TEUCHOS_TEST_FOR_EXCEPTION(mean_sum_ref==0.0||mean_sum_def==0.0,std::runtime_error," invalid mean sum (cannot be 0.0, ZNSSD is then undefined)" <<
    mean_sum_ref << " " << mean_sum_def);
  scalar_t gamma = 0.0;
#ifdef DICE_KOKKOS
  ZNSSD_Gamma_Functor gamma_func(ref_intensities_.d_view,
     def_intensities_.d_view,
     is_active_.d_view,
     is_deactivated_this_step_.d_view,
     mean_ref,mean_def,
     mean_sum_ref,mean_sum_def);
  Kokkos::parallel_reduce(num_pixels_,gamma_func,gamma);
#else
  scalar_t value = 0.0;
  for(size_t i=0;i<num_pixels_;++i){
    if(is_active_[i]&!is_deactivated_this_step_[i]){
      value = (def_intensities_[i]-mean_d_)/mean_sum_d_ - (ref_intensities_[i]-mean_r_)/mean_sum_r_;
      gamma += value*value;
    }
  }
#endif
  return gamma;
}

Teuchos::ArrayRCP<scalar_t>
Subset::grad_x_array()const{
  // note: the Kokkos version returns a copy of the array, the serial version returns the array itself (providing access to modify)
#ifdef DICE_KOKKOS
  Teuchos::ArrayRCP<scalar_t> array(num_pixels_,0.0);
  for(int_t i=0;i<num_pixels_;++i)
    array[i] = grad_x_.h_view.ptr_on_device()[i];
  return array;
#else
  return grad_x_;
#endif
}

Teuchos::ArrayRCP<scalar_t>
Subset::grad_y_array()const{
#ifdef DICE_KOKKOS
  Teuchos::ArrayRCP<scalar_t> array(num_pixels_,0.0);
  for(int_t i=0;i<num_pixels_;++i)
    array[i] = grad_y_.h_view.ptr_on_device()[i];
  return array;
#else
  return grad_y_;
#endif
}

void
Subset::initialize(Teuchos::RCP<Image> image,
  const Subset_View_Target target,
  Teuchos::RCP<const std::vector<scalar_t> > deformation,
  const Interpolation_Method interp){

  // coordinates for points x and y are always in global coordinates
  // if the input image is a sub-image i.e. it has offsets, then these need to be taken into account
  const int_t offset_x = image->offset_x();
  const int_t offset_y = image->offset_y();

#ifdef DICE_KOKKOS
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
    if(image->has_gradients()){
      // TODO thread this:
      // copy over the image gradients:
      for(int_t px=0;px<num_pixels_;++px){
        grad_x_.h_view(px) = image->grad_x().h_view(y_.h_view(px)-offset_y,x_.h_view(px)-offset_x);
        grad_y_.h_view(px) = image->grad_y().h_view(y_.h_view(px)-offset_y,x_.h_view(px)-offset_x);
      }
      has_gradients_ = true;
    }
    grad_x_.modify<host_space>();
    grad_x_.sync<device_space>();
    grad_y_.modify<host_space>();
    grad_y_.sync<device_space>();
  }
  else{
    def_intensities_.modify<device_space>();
    def_intensities_.sync<host_space>();
  }
#else
  const int_t image_w = image->width();
   const int_t image_h = image->height();
   Teuchos::ArrayRCP<intensity_t> intensities_ = target==REF_INTENSITIES ? ref_intensities_ : def_intensities_;
   // assume if the map is null, use the no_map_tag in the parrel for call of the functor
   if(deformation==Teuchos::null){
     for(size_t i=0;i<num_pixels_;++i)
       intensities_[i] = (*image)(x_[i]-offset_x,y_[i]-offset_y);
   }
   else{
     const scalar_t u = (*deformation)[DISPLACEMENT_X];
     const scalar_t v = (*deformation)[DISPLACEMENT_Y];
     const scalar_t t = (*deformation)[ROTATION_Z];
     const scalar_t ex = (*deformation)[NORMAL_STRAIN_X];
     const scalar_t ey = (*deformation)[NORMAL_STRAIN_Y];
     const scalar_t g = (*deformation)[SHEAR_STRAIN_XY];
     const scalar_t cos_t = std::cos(t);
     const scalar_t sin_t = std::sin(t);
     // initialize the work variables
     scalar_t mapped_x = 0.0;
     scalar_t mapped_y = 0.0;
     scalar_t dx = 0.0,dy=0.0,Dx=0.0,Dy=0.0;
     int_t x1=0,x2=0,y1=0,y2=0;
     const scalar_t ox=(scalar_t)offset_x,oy=(scalar_t)offset_y;
     scalar_t dx2=0.0, dx3=0.0;
     scalar_t dy2=0.0, dy3=0.0;
     scalar_t f0x=0.0, f0y=0.0;
     scalar_t intensity_value = 0.0;
     for(size_t i=0;i<num_pixels_;++i){
       dx = (scalar_t)(x_[i]) - cx_;
       dy = (scalar_t)(y_[i]) - cy_;
       Dx = (1.0+ex)*dx + g*dy;
       Dy = (1.0+ey)*dy + g*dx;
       // mapped location
       mapped_x = cos_t*Dx - sin_t*Dy + u + cx_ - ox;
       mapped_y = sin_t*Dx + cos_t*Dy + v + cy_ - oy;
       if(interp==BILINEAR){
         // check that the mapped location is inside the image...
         if(mapped_x>=0&&mapped_x<image_w-1.5&&mapped_y>=0&&mapped_y<image_h-1.5){
           x1 = (int_t)mapped_x;
           x2 = x1+1;
           y1 = (int_t)mapped_y;
           y2  = y1+1;
           intensities_[i] =
               ((*image)(x1,y1)*(x2-mapped_x)*(y2-mapped_y)
                   +(*image)(x2,y1)*(mapped_x-x1)*(y2-mapped_y)
                   +(*image)(x2,y2)*(mapped_x-x1)*(mapped_y-y1)
                   +(*image)(x1,y2)*(x2-mapped_x)*(mapped_y-y1));
         }
         else{
           // out of bounds pixels are black
           intensities_[i] = 0;
         }
       }
       else if(interp==KEYS_FOURTH){
         x1 = (int_t)mapped_x;
         if(mapped_x - x1 >= 0.5) x1++;
         y1 = (int_t)mapped_y;
         if(mapped_y - y1 >= 0.5) y1++;
         // check that the mapped location is inside the image...
         if(mapped_x>2.5&&mapped_x<image_w-3.5&&mapped_y>2.5&&mapped_y<image_h-3.5){
           intensity_value = 0.0;
           // convolve all the pixels within + and - pixels of the point in question
           for(int_t y=y1-3;y<=y1+3;++y){
             dy = std::abs(mapped_y - y);
             dy2=dy*dy;
             dy3=dy2*dy;
             f0y = 0.0;
             if(dy <= 1.0){
               f0y = 4.0/3.0*dy3 - 7.0/3.0*dy2 + 1.0;
             }
             else if(dy <= 2.0){
               f0y = -7.0/12.0*dy3 + 3.0*dy2 - 59.0/12.0*dy + 15.0/6.0;
             }
             else if(dy <= 3.0){
               f0y = 1.0/12.0*dy3 - 2.0/3.0*dy2 + 21.0/12.0*dy - 3.0/2.0;
             }
             for(int_t x=x1-3;x<=x1+3;++x){
               // compute the f's of x and y
               dx = std::abs(mapped_x - x);
               dx2=dx*dx;
               dx3=dx2*dx;
               f0x = 0.0;
               if(dx <= 1.0){
                 f0x = 4.0/3.0*dx3 - 7.0/3.0*dx2 + 1.0;
               }
               else if(dx <= 2.0){
                 f0x = -7.0/12.0*dx3 + 3.0*dx2 - 59.0/12.0*dx + 15.0/6.0;
               }
               else if(dx <= 3.0){
                 f0x = 1.0/12.0*dx3 - 2.0/3.0*dx2 + 21.0/12.0*dx - 3.0/2.0;
               }
               intensity_value += (*image)(x,y)*f0x*f0y;
             }
           }
           intensities_[i] = intensity_value;
         }
         // bilinear as a fall back near the edges
         else if(mapped_x>=0&&mapped_x<image_w-1.5&&mapped_y>=0&&mapped_y<image_h-1.5){
           x1 = (int_t)mapped_x;
           x2 = x1+1;
           y1 = (int_t)mapped_y;
           y2  = y1+1;
           intensities_[i] =
               ((*image)(x1,y1)*(x2-mapped_x)*(y2-mapped_y)
                +(*image)(x2,y1)*(mapped_x-x1)*(y2-mapped_y)
                +(*image)(x2,y2)*(mapped_x-x1)*(mapped_y-y1)
                +(*image)(x1,y2)*(x2-mapped_x)*(mapped_y-y1));
         }
         else{
           intensities_[i] = 0;
         }
       }
       else{
         TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,
           "Error, unknown interpolation method requested");
       }
     }
   }
   // now sync up the intensities:
   if(target==REF_INTENSITIES){
     if(image->has_gradients()){
       // copy over the image gradients:
       for(int_t px=0;px<num_pixels_;++px){
         grad_x_[px] = image->grad_x()[y_[px]-offset_y,x_[px]-offset_x];
         grad_y_[px] = image->grad_y()[y_[px]-offset_y,x_[px]-offset_x];
       }
       has_gradients_ = true;
     }
   }
#endif
  // turn off the obstructed pixels if the deformation vector is not null
  if(deformation!=Teuchos::null){
    // assumes that the check for obstructions has already been done
    turn_off_obstructed_pixels(deformation);
  }
}

#ifdef DICE_KOKKOS
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
  scalar_t mapped_x = cos_t_*Dx - sin_t_*Dy + u_ + cx_ - (scalar_t)offset_x_;
  scalar_t mapped_y = sin_t_*Dx + cos_t_*Dy + v_ + cy_ - (scalar_t)offset_y_;

  // check that the mapped location is inside the image...
  if(mapped_x>=0&&mapped_x<image_w_-1.5&&mapped_y>=0&&mapped_y<image_h_-1.5){
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
  scalar_t mapped_x = cos_t_*Dx - sin_t_*Dy + u_ + cx_ - (scalar_t)offset_x_;
  scalar_t mapped_y = sin_t_*Dx + cos_t_*Dy + v_ + cy_ - (scalar_t)offset_y_;
  // determine the current pixel the coordinates fall in:
  int_t px = (int_t)mapped_x;
  if(mapped_x - px >= 0.5) px++;
  int_t py = (int_t)mapped_y;
  if(mapped_y - py >= 0.5) py++;

  // check that the mapped location is inside the image...
  if(mapped_x>2.5&&mapped_x<image_w_-3.5&&mapped_y>2.5&&mapped_y<image_h_-3.5){
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
        f0y = 4.0/3.0*dy3 - 7.0/3.0*dy2 + 1.0;
      }
      else if(dy <= 2.0){
        f0y = -7.0/12.0*dy3 + 3.0*dy2 - 59.0/12.0*dy + 15.0/6.0;
      }
      else if(dy <= 3.0){
        f0y = 1.0/12.0*dy3 - 2.0/3.0*dy2 + 21.0/12.0*dy - 3.0/2.0;
      }
      for(int_t x=px-3;x<=px+3;++x){
        // compute the f's of x and y
        dx = std::abs(mapped_x - x);
        dx2=dx*dx;
        dx3=dx2*dx;
        f0x = 0.0;
        if(dx <= 1.0){
          f0x = 4.0/3.0*dx3 - 7.0/3.0*dx2 + 1.0;
        }
        else if(dx <= 2.0){
          f0x = -7.0/12.0*dx3 + 3.0*dx2 - 59.0/12.0*dx + 15.0/6.0;
        }
        else if(dx <= 3.0){
          f0x = 1.0/12.0*dx3 - 2.0/3.0*dx2 + 21.0/12.0*dx - 3.0/2.0;
        }
        intensity_value += image_intensities_(y,x)*f0x*f0y;
      }
    }
    subset_intensities_(pixel_index) = intensity_value;
  }
  // bilinear as a fall back near the edges
  else if(mapped_x>=0&&mapped_x<image_w_-1.5&&mapped_y>=0&&mapped_y<image_h_-1.5){
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
    subset_intensities_(pixel_index) = 0;
  }
}

// functor to populate the values of a subset
// without using the mapping
KOKKOS_INLINE_FUNCTION
void
Subset_Init_Functor::operator()(const No_Map_Tag&,
  const int_t pixel_index)const{
  subset_intensities_(pixel_index) = image_intensities_(y_(pixel_index)-offset_y_,x_(pixel_index)-offset_x_);
}
#endif

}// End DICe Namespace
