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
  x_ = x;
  y_ = y;
  ref_intensities_ = Teuchos::ArrayRCP<intensity_t>(num_pixels_,0.0);
  def_intensities_ = Teuchos::ArrayRCP<intensity_t>(num_pixels_,0.0);
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
 is_conformal_(false)
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
  // warn the user if the centroid is outside the subset
  std::pair<int_t,int_t> centroid_pair = std::pair<int_t,int_t>(cy_,cx_);
  if(coords.find(centroid_pair)==coords.end())
    std::cout << "*** Warning: centroid " << cx_ << " " << cy_ << " is outside the subset boundary" << std::endl;
  ref_intensities_ = Teuchos::ArrayRCP<intensity_t>(num_pixels_,0.0);
  def_intensities_ = Teuchos::ArrayRCP<intensity_t>(num_pixels_,0.0);
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

intensity_t&
Subset::ref_intensities(const int_t pixel_index){
  return ref_intensities_[pixel_index];
}

intensity_t&
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
Subset::mean(const Subset_View_Target target){
  scalar_t mean = 0.0;
  if(target==REF_INTENSITIES){
    for(int_t i=0;i<num_pixels_;++i)
      mean += ref_intensities_[i];
  }else{
    for(int_t i=0;i<num_pixels_;++i)
      mean += def_intensities_[i];
  }
  return mean/num_pixels_;
}

scalar_t
Subset::mean(const Subset_View_Target target,
  scalar_t & sum){
  scalar_t mean_ = mean(target);
  sum = 0.0;
  if(target==REF_INTENSITIES){
    for(int_t i=0;i<num_pixels_;++i)
      sum += (ref_intensities_[i]-mean_)*(ref_intensities_[i]-mean_);
  }else{
    for(int_t i=0;i<num_pixels_;++i)
      sum += (def_intensities_[i]-mean_)*(def_intensities_[i]-mean_);
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
  TEUCHOS_TEST_FOR_EXCEPTION(mean_sum_ref==0.0||mean_sum_def==0.0,std::runtime_error," invalid mean sum (cannot be 0.0, ZNSSD is then undefined)" <<
    mean_sum_ref << " " << mean_sum_def);
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

Teuchos::ArrayRCP<scalar_t>
Subset::grad_x_array()const{
  // note: the Kokkos version returns a copy of the array, the serial version returns the array itself (providing access to modify)
  return grad_x_;
}

Teuchos::ArrayRCP<scalar_t>
Subset::grad_y_array()const{
  return grad_y_;
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

  const int_t image_w = image->width();
   const int_t image_h = image->height();
   Teuchos::ArrayRCP<intensity_t> intensities_ = target==REF_INTENSITIES ? ref_intensities_ : def_intensities_;
   // assume if the map is null, use the no_map_tag in the parrel for call of the functor
   if(deformation==Teuchos::null){
     for(int_t i=0;i<num_pixels_;++i)
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
     for(int_t i=0;i<num_pixels_;++i){
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
         grad_x_[px] = image->grad_x(x_[px]-offset_x,y_[px]-offset_y);
         grad_y_[px] = image->grad_y(x_[px]-offset_x,y_[px]-offset_y);
       }
       has_gradients_ = true;
     }
   }
  // turn off the obstructed pixels if the deformation vector is not null
  if(deformation!=Teuchos::null){
    // assumes that the check for obstructions has already been done
    turn_off_obstructed_pixels(deformation);
  }
}

}// End DICe Namespace
