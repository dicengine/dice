// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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

#include <DICe_Subset.h>
#include <DICe_ImageIO.h>

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
  is_conformal_(false),
  sub_image_id_(0)
{
  assert(num_pixels_>0);
  assert(x.size()==y.size());
  x_ = x;
  y_ = y;
  for(int_t i=0;i<x_.size();++i){
    if(x_[i]<0) x_[i]=0;
    if(y_[i]<0) y_[i]=0;
  }
  ref_intensities_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  def_intensities_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  grad_x_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  grad_y_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  is_active_ = Teuchos::ArrayRCP<bool>(num_pixels_,true);
  is_deactivated_this_step_ = Teuchos::ArrayRCP<bool>(num_pixels_,false);
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
 is_conformal_(false),
 sub_image_id_(0)
{
  assert(width>0);
  assert(height>0);
  const int_t half_width = width/2;
  const int_t half_height = height/2;
  // if the width and height arguments are not odd, the next larges odd size is used:
  num_pixels_ = (2*half_width+1)*(2*half_height+1);
  assert(num_pixels_>0);
  x_ = Teuchos::ArrayRCP<int_t>(num_pixels_,0);
  y_ = Teuchos::ArrayRCP<int_t>(num_pixels_,0);
  int_t index = 0;
  for(int_t y=cy_-half_height;y<=cy_+half_height;++y){
    for(int_t x=cx_-half_width;x<=cx_+half_width;++x){
      x_[index] = x<0?0:x;
      y_[index] = y<0?0:y;
      index++;
    }
  }
  ref_intensities_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  def_intensities_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  grad_x_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  grad_y_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  is_active_ = Teuchos::ArrayRCP<bool>(num_pixels_,true);
  is_deactivated_this_step_ = Teuchos::ArrayRCP<bool>(num_pixels_,false);
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
  is_conformal_(true),
  sub_image_id_(0)
{
  assert(subset_def.has_boundary());
  std::set<std::pair<int_t,int_t> > coords;
  for(size_t i=0;i<subset_def.boundary()->size();++i){
    std::set<std::pair<int_t,int_t> > shapeCoords = (*subset_def.boundary())[i]->get_owned_pixels();
    coords.insert(shapeCoords.begin(),shapeCoords.end());
  }
  // at this point all the coordinate pairs are in the set
  num_pixels_ = coords.size();
  x_ = Teuchos::ArrayRCP<int_t>(num_pixels_,0);
  y_ = Teuchos::ArrayRCP<int_t>(num_pixels_,0);
  int_t index = 0;
  // NOTE: the pairs are (y,x) not (x,y) so that the ordering is correct in the set
  std::set<std::pair<int_t,int_t> >::iterator set_it = coords.begin();
  for( ; set_it!=coords.end();++set_it){
    x_[index] = set_it->second < 0 ? 0 : set_it->second;
    y_[index] = set_it->first < 0 ? 0: set_it->first;
    index++;
  }
  // warn the user if the centroid is outside the subset
  std::pair<int_t,int_t> centroid_pair = std::pair<int_t,int_t>(cy_,cx_);
  if(coords.find(centroid_pair)==coords.end())
    std::cout << "*** Warning: centroid " << cx_ << " " << cy_ << " is outside the subset boundary" << std::endl;
  ref_intensities_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  def_intensities_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  grad_x_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  grad_y_ = Teuchos::ArrayRCP<scalar_t>(num_pixels_,0.0);
  is_active_ = Teuchos::ArrayRCP<bool>(num_pixels_,true);
  is_deactivated_this_step_ = Teuchos::ArrayRCP<bool>(num_pixels_,false);
  reset_is_active();
  reset_is_deactivated_this_step();

  // now set the inactive bit for the second set of multishapes if they exist.
  if(subset_def.has_excluded_area()){
    for(size_t i=0;i<subset_def.excluded_area()->size();++i){
      (*subset_def.excluded_area())[i]->deactivate_pixels(num_pixels_,is_active_.getRawPtr(),x_.getRawPtr(),y_.getRawPtr());
    }
  }
  if(subset_def.has_obstructed_area()){
    for(size_t i=0;i<subset_def.obstructed_area()->size();++i){
      std::set<std::pair<int_t,int_t> > obstructedArea = (*subset_def.obstructed_area())[i]->get_owned_pixels();
      obstructed_coords_.insert(obstructedArea.begin(),obstructedArea.end());
    }
  }
}

void
Subset::update_centroid(const int_t cx, const int_t cy){
  const int_t delta_x = cx - cx_;
  const int_t delta_y = cy - cy_;
  cx_ = cx;
  cy_ = cy;
  for(int_t i=0;i<num_pixels_;++i){
    x_[i] += delta_x; if(x_[i]<0) x_[i]=0;
    y_[i] += delta_y; if(y_[i]<0) y_[i]=0;
  }
}

const int_t&
Subset::x(const int_t pixel_index)const{
  return x_[pixel_index];
}

const int_t&
Subset::y(const int_t pixel_index)const{
  return y_[pixel_index];
}

const scalar_t&
Subset::grad_x(const int_t pixel_index)const{
  return grad_x_[pixel_index];
}

const scalar_t&
Subset::grad_y(const int_t pixel_index)const{
  return grad_y_[pixel_index];
}

scalar_t&
Subset::ref_intensities(const int_t pixel_index){
  return ref_intensities_[pixel_index];
}

scalar_t&
Subset::def_intensities(const int_t pixel_index){
  return def_intensities_[pixel_index];
}

/// returns true if this pixel is active
bool &
Subset::is_active(const int_t pixel_index){
  return is_active_[pixel_index];
}

/// returns true if this pixel is deactivated for this particular frame
bool &
Subset::is_deactivated_this_step(const int_t pixel_index){
  return is_deactivated_this_step_[pixel_index];
}

void
Subset::reset_is_active(){
  for(int_t i=0;i<num_pixels_;++i)
    is_active_[i] = true;
}

void
Subset::reset_is_deactivated_this_step(){
  for(int_t i=0;i<num_pixels_;++i)
    is_deactivated_this_step_[i] = false;
}

scalar_t
Subset::max(const Subset_View_Target target){
  scalar_t max = std::numeric_limits<scalar_t>::min();
  if(target==REF_INTENSITIES){
    for(int_t i=0;i<num_pixels_;++i){
      if(is_active_[i]&!is_deactivated_this_step_[i])
        if(ref_intensities_[i]>max)
          max = ref_intensities_[i];
    }
  }else{
    for(int_t i=0;i<num_pixels_;++i)
      if(is_active_[i]&!is_deactivated_this_step_[i]){
        if(def_intensities_[i]>max)
          max = def_intensities_[i];
      }
  }
  return max;
}

scalar_t
Subset::min(const Subset_View_Target target){
  scalar_t min = std::numeric_limits<scalar_t>::max();
  if(target==REF_INTENSITIES){
    for(int_t i=0;i<num_pixels_;++i){
      if(is_active_[i]&!is_deactivated_this_step_[i])
        if(ref_intensities_[i]<min)
          min = ref_intensities_[i];
    }
  }else{
    for(int_t i=0;i<num_pixels_;++i)
      if(is_active_[i]&!is_deactivated_this_step_[i]){
        if(def_intensities_[i]<min)
          min = def_intensities_[i];
      }
  }
  return min;
}

void
Subset::round(const Subset_View_Target target){
  if(target==REF_INTENSITIES){
    for(int_t i=0;i<num_pixels_;++i)
      ref_intensities_[i] = std::round(ref_intensities_[i]);
  }else{
    for(int_t i=0;i<num_pixels_;++i)
      def_intensities_[i] = std::round(def_intensities_[i]);
  }
}

scalar_t
Subset::mean(const Subset_View_Target target){
  scalar_t mean = 0.0;
  int_t num_active = num_active_pixels();
  if(target==REF_INTENSITIES){
    for(int_t i=0;i<num_pixels_;++i){
      if(is_active_[i]&!is_deactivated_this_step_[i])
        mean += ref_intensities_[i];
    }
  }else{
    for(int_t i=0;i<num_pixels_;++i)
      if(is_active_[i]&!is_deactivated_this_step_[i]){
        mean += def_intensities_[i];
      }
  }
  return num_active != 0 ? mean/num_active : 0.0;
}

scalar_t
Subset::mean(const Subset_View_Target target,
  scalar_t & sum){
  scalar_t mean_ = mean(target);
  sum = 0.0;
  if(target==REF_INTENSITIES){
    for(int_t i=0;i<num_pixels_;++i){
      if(is_active_[i]&!is_deactivated_this_step_[i])
        sum += (ref_intensities_[i]-mean_)*(ref_intensities_[i]-mean_);
    }
  }else{
    for(int_t i=0;i<num_pixels_;++i){
      if(is_active_[i]&!is_deactivated_this_step_[i])
        sum += (def_intensities_[i]-mean_)*(def_intensities_[i]-mean_);
    }
  }
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
  if(mean_sum_ref==0.0||mean_sum_def==0.0) return -1.0;
  scalar_t gamma = 0.0;
  scalar_t value = 0.0;
  for(int_t i=0;i<num_pixels_;++i){
    if(is_active_[i]&!is_deactivated_this_step_[i]){
      value = (def_intensities_[i]-mean_def)/mean_sum_def - (ref_intensities_[i]-mean_ref)/mean_sum_ref;
      gamma += value*value;
    }
  }
  return gamma;
}

scalar_t
Subset::diff_ref_def() const{
  scalar_t diff = 0.0;
  for(int_t i=0;i<num_pixels_;++i)
    diff += (ref_intensities_[i]-def_intensities_[i])*(ref_intensities_[i]-def_intensities_[i]);
  diff = std::sqrt(diff);
  return diff;
}


Teuchos::ArrayRCP<scalar_t>
Subset::grad_x_array()const{
  return grad_x_;
}

Teuchos::ArrayRCP<scalar_t>
Subset::grad_y_array()const{
  return grad_y_;
}

scalar_t
Subset::sssig(){
  // assumes obstructed pixels are already turned off
  scalar_t sssig = 0.0;
  for(int_t i=0;i<num_pixels_;++i){
    sssig += grad_x_[i]*grad_x_[i] + grad_y_[i]*grad_y_[i];
  }
  sssig /= num_pixels_==0.0?1.0:num_pixels_;
  return sssig;
}

template <typename S>
void
Subset::initialize(Teuchos::RCP<Image_<S>> image,
  const Subset_View_Target target,
  Teuchos::RCP<Local_Shape_Function> shape_function,
  const Interpolation_Method interp){
//  DEBUG_MSG("Subset::initialize():  initializing subset from image " << image->file_name());
  // coordinates for points x and y are always in global coordinates
  // if the input image is a sub-image i.e. it has offsets, then these need to be taken into account
  const int_t offset_x = image->offset_x();
  const int_t offset_y = image->offset_y();
  const int_t w = image->width();
  const int_t h = image->height();
  Teuchos::ArrayRCP<scalar_t> intensities = target==REF_INTENSITIES ? ref_intensities_ : def_intensities_;
  // assume if the map is null, use the no_map_tag in the parrel for call of the functor
  if(shape_function==Teuchos::null){
    for(int_t i=0;i<num_pixels_;++i){
      intensities[i] = (*image)(x_[i]-offset_x,y_[i]-offset_y);
    }
  }
  else{
    int_t px,py;
    const bool has_blocks = !pixels_blocked_by_other_subsets_.empty();
    // initialize the work variables
    scalar_t mapped_x = 0.0;
    scalar_t mapped_y = 0.0;
    const scalar_t ox=(scalar_t)offset_x,oy=(scalar_t)offset_y;
    const bool has_gradients = image->has_gradients();
    // function pointer to avoid having to set the interpolation method for each pixel
    void (Image_<S>::*interp_func)(scalar_t&,scalar_t&,scalar_t&,const bool,const scalar_t&,const scalar_t&) const = image->get_interpolant(interp);
    for(int_t i=0;i<num_pixels_;++i){
      shape_function->map(x_[i],y_[i],cx_,cy_,mapped_x,mapped_y);
      px = ((int_t)(mapped_x + 0.5) == (int_t)(mapped_x)) ? (int_t)(mapped_x) : (int_t)(mapped_x) + 1;
      py = ((int_t)(mapped_y + 0.5) == (int_t)(mapped_y)) ? (int_t)(mapped_y) : (int_t)(mapped_y) + 1;
      // out of image bounds ( 4 pixel buffer to ensure enough room to interpolate away from the sub image boundary)
      if(px<offset_x+4||px>=offset_x+w-4||py<offset_y+4||py>=offset_y+h-4){
        is_deactivated_this_step(i) = true;
        continue;
      }
      if(is_obstructed_pixel(mapped_x,mapped_y)){
        is_deactivated_this_step(i) = true;
        continue;
      }
      if(has_blocks){
        if(pixels_blocked_by_other_subsets_.find(std::pair<int_t,int_t>(py,px))
            !=pixels_blocked_by_other_subsets_.end()){
          is_deactivated_this_step(i) = true;
          continue;
        }
      }
      // if the code got here, the pixel is not deactivated
      is_deactivated_this_step(i) = false;
      (*image.*interp_func)(intensities[i], grad_x_[i], grad_y_[i],has_gradients, mapped_x-ox, mapped_y-oy);
    }
  }
  // sync up the intensities:
  if(target==REF_INTENSITIES){
    if(image->has_gradients()){
      // copy over the image gradients:
      for(int_t px=0;px<num_pixels_;++px){
        grad_x_[px] = image->grad_x(x_[px]-offset_x,y_[px]-offset_y);
        grad_y_[px] = image->grad_y(x_[px]-offset_x,y_[px]-offset_y);
      }
      has_gradients_ = true;
    }
  }
//  DEBUG_MSG("Subset::initialize():  initializing complete");
}
template DICE_LIB_DLL_EXPORT void Subset::initialize(Teuchos::RCP<Image_<scalar_t>>,const Subset_View_Target,Teuchos::RCP<Local_Shape_Function>,const Interpolation_Method);
#ifndef STORAGE_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void Subset::initialize(Teuchos::RCP<Image_<storage_t>>,const Subset_View_Target,Teuchos::RCP<Local_Shape_Function>,const Interpolation_Method);
#endif

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
Subset::deformed_shapes(Teuchos::RCP<Local_Shape_Function> shape_function,
  const int_t cx,
  const int_t cy,
  const scalar_t & skin_factor){
  std::set<std::pair<int_t,int_t> > coords;
  if(!is_conformal_) return coords;
  for(size_t i=0;i<conformal_subset_def_.boundary()->size();++i){
    std::set<std::pair<int_t,int_t> > shapeCoords =
        (*conformal_subset_def_.boundary())[i]->get_owned_pixels(shape_function,cx,cy,skin_factor);
    coords.insert(shapeCoords.begin(),shapeCoords.end());
  }
  return coords;
}

void
Subset::turn_off_obstructed_pixels(Teuchos::RCP<Local_Shape_Function> shape_function){
  assert(shape_function!=Teuchos::null);

  scalar_t X=0.0,Y=0.0;
  int_t px=0,py=0;
  const bool has_blocks = !pixels_blocked_by_other_subsets_.empty();
  reset_is_deactivated_this_step();

  for(int_t i=0;i<num_pixels_;++i){
    shape_function->map(x(i),y(i),cx_,cy_,X,Y);
    if(is_obstructed_pixel(X,Y)){
      is_deactivated_this_step(i) = true;
    }
    else{
      is_deactivated_this_step(i) = false;
    }
    // pixels blocked by other subsets:
    if(has_blocks){
      px = ((int_t)(X + 0.5) == (int_t)(X)) ? (int_t)(X) : (int_t)(X) + 1;
      py = ((int_t)(Y + 0.5) == (int_t)(Y)) ? (int_t)(Y) : (int_t)(Y) + 1;
      if(pixels_blocked_by_other_subsets_.find(std::pair<int_t,int_t>(py,px))
          !=pixels_blocked_by_other_subsets_.end()){
        is_deactivated_this_step(i) = true;
      }
    }
  } // pixel loop
}

void
Subset::turn_on_previously_obstructed_pixels(){
  // this assumes that the is_deactivated_this_step_ flags have already been set correctly prior
  // to calling this method.
  for(int_t px=0;px<num_pixels_;++px){
    // it's not obstructed this step, but was inactive to begin with
    if(!is_deactivated_this_step(px) && !is_active(px)){
      // take the pixel value from the deformed subset
      ref_intensities(px) = def_intensities(px);
      // set the active bit to true
      is_active(px) = true;
    }
  }
}

template <typename S>
void
Subset::write_subset_on_image(const std::string & file_name,
  Teuchos::RCP<Image_<S>> image,
  Teuchos::RCP<Local_Shape_Function> shape_function){
  //create a square image that fits the extents of the subet
  const int_t w = image->width();
  const int_t h = image->height();
  const int_t ox = image->offset_x();
  const int_t oy = image->offset_y();
  Teuchos::ArrayRCP<S> intensities(w*h,0);
  for(int_t m=0;m<h;++m){
    for(int_t n=0;n<w;++n){
      intensities[m*w+n] = (*image)(n,m);
    }
  }
  scalar_t mapped_x=0.0,mapped_y=0.0;
  int_t px=0,py=0;

  if(shape_function!=Teuchos::null){
    for(int_t i=0;i<num_pixels_;++i){
      shape_function->map(x(i),y(i),cx_,cy_,mapped_x,mapped_y);
      // get the nearest pixel location:
      px = (int_t)mapped_x;
      if(mapped_x - (int_t)mapped_x >= 0.5) px++;
      py = (int_t)mapped_y;
      if(mapped_y - (int_t)mapped_y >= 0.5) py++;
      intensities[py*w+px] = !is_active(i) ? 255
          : is_deactivated_this_step(i) ?  0
          : std::abs((def_intensities(i) - ref_intensities(i))*2);
    }
  }
  else{ // write the original shape of the subset
    for(int_t i=0;i<num_pixels_;++i)
      intensities[(y(i)-oy)*w+(x(i)-ox)] = 255;
  }
  utils::write_image(file_name.c_str(),w,h,intensities.getRawPtr(),true);
}

template DICE_LIB_DLL_EXPORT void Subset::write_subset_on_image(const std::string &,Teuchos::RCP<Image_<scalar_t>>,Teuchos::RCP<Local_Shape_Function>);
#ifndef STORAGE_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void Subset::write_subset_on_image(const std::string &,Teuchos::RCP<Image_<storage_t>>,Teuchos::RCP<Local_Shape_Function>);
#endif

void
Subset::write_image(const std::string & file_name,
  const bool use_def_intensities){
  // determine the extents of the subset and the offsets
  int_t max_x = 0;
  int_t max_y = 0;
  int_t min_x = x(0);
  int_t min_y = y(0);
  for(int_t i=0;i<num_pixels_;++i){
    if(x(i) > max_x) max_x = x(i);
    if(x(i) < min_x) min_x = x(i);
    if(y(i) > max_y) max_y = y(i);
    if(y(i) < min_y) min_y = y(i);
  }
  //create a square image that fits the extents of the subet
  const int_t w = max_x - min_x + 1;
  const int_t h = max_y - min_y + 1;
  Teuchos::ArrayRCP<scalar_t> intensities(w*h,0);
  for(int_t i=0;i<w*h;++i)
    intensities[i] = 0.0;
  for(int_t i=0;i<num_pixels_;++i){
    if(!is_active(i)){
      intensities[(y(i)-min_y)*w+(x(i)-min_x)] = 100; // color the inactive areas gray
    }
    else{
      intensities[(y(i)-min_y)*w+(x(i)-min_x)] = use_def_intensities ?
          def_intensities(i) : ref_intensities(i);
    }
  }
  utils::write_image(file_name.c_str(),w,h,intensities.getRawPtr(),true);
}

int_t
Subset::num_active_pixels(){
  int_t num_active = 0;
  for(int_t i=0;i<num_pixels();++i){
    if(is_active(i)&&!is_deactivated_this_step(i))
      num_active++;
  }
  return num_active;
}

scalar_t
Subset::contrast_std_dev(){
  const scalar_t mean_intensity = mean(DEF_INTENSITIES);
  scalar_t std_dev = 0.0;
  int_t num_active = 0;
  for(int_t i = 0;i<num_pixels();++i){
    if(is_active(i)&&!is_deactivated_this_step(i)){
      num_active++;
      std_dev += (def_intensities(i) - mean_intensity)*(def_intensities(i) - mean_intensity);
    }
  }
  std_dev = num_active==0?0.0:std::sqrt(std_dev/num_active);
  return std_dev;
}

template <typename S>
scalar_t
Subset::noise_std_dev(Teuchos::RCP<Image_<S>> image,
  Teuchos::RCP<Local_Shape_Function> shape_function){

  // create the mask
  static scalar_t mask[3][3] = {{1, -2, 1},{-2,4,-2},{1,-2,1}};

  // determine the extents of the subset:
  int_t min_x = x(0);
  int_t max_x = x(0);
  int_t min_y = y(0);
  int_t max_y = y(0);
  for(int_t i=0;i<num_pixels();++i){
    if(x(i) < min_x) min_x = x(i);
    if(x(i) > max_x) max_x = x(i);
    if(y(i) < min_y) min_y = y(i);
    if(y(i) > max_y) max_y = y(i);
  }

  scalar_t u = 0.0;
  scalar_t v = 0.0;
  scalar_t t = 0.0;
  shape_function->map_to_u_v_theta(cx_,cy_,u,v,t);
  min_x += u; max_x += u;
  min_y += v; max_y += v;

  DEBUG_MSG("Subset::noise_std_dev(): Extents of subset " << min_x << " " << max_x << " " << min_y << " " << max_y);
  const int_t h = max_y - min_y + 1;
  const int_t w = max_x - min_x + 1;
  const int_t img_h = image->height();
  const int_t img_w = image->width();
  const int_t ox = image->offset_x();
  const int_t oy = image->offset_y();
  DEBUG_MSG("Subset::noise_std_dev(): Extents of image " << ox << " " << ox + img_w << " " << oy << " " << oy + img_h);

  // ensure that the subset falls inside the image
  if(max_x >= img_w + ox || min_x < ox || max_y >= img_h + oy || min_y < oy){
    return 1.0;
  }

  scalar_t variance = 0.0;
  scalar_t conv_i = 0.0;
  // convolve and sum the intensities with the mask
  for(int_t y=min_y; y<max_y;++y){
    for(int_t x=min_x; x<max_x;++x){
      // don't convolve the edge pixels
      if(x-ox<1||x-ox>=img_w-1||y-oy<1||y-oy>=img_h-1){
        variance += std::sqrt((*image)(x-ox,y-oy)*(*image)(x-ox,y-oy));
      }
      else{
        conv_i = 0.0;
        for(int_t j=0;j<3;++j){
          for(int_t i=0;i<3;++i){
            conv_i += (*image)(x-ox+(i-1),y-oy+(j-1))*mask[i][j];
          }
        }
        variance += std::abs(conv_i);
      }
    }
  }
  variance *= std::sqrt(0.5*DICE_PI) / (6.0*(w-2)*(h-2));
  DEBUG_MSG("Subset::noise_std_dev(): return value " << variance);
  return variance;
}

template DICE_LIB_DLL_EXPORT scalar_t Subset::noise_std_dev(Teuchos::RCP<Image_<scalar_t>>,Teuchos::RCP<Local_Shape_Function>);
#ifndef STORAGE_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT scalar_t Subset::noise_std_dev(Teuchos::RCP<Image_<storage_t>>,Teuchos::RCP<Local_Shape_Function>);
#endif

}// End DICe Namespace
