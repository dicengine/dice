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
#include <DICe_ImageIO.h>
#if DICE_KOKKOS
  #include <DICe_Kokkos.h>
#endif

#include <cassert>

namespace DICe {

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
#if DICE_KOKKOS
  is_deactivated_this_step_.modify<host_space>();
  is_deactivated_this_step_.sync<device_space>();
#endif
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

void
Subset::write_subset_on_image(const std::string & file_name,
  Teuchos::RCP<Image> image,
  Teuchos::RCP<const std::vector<scalar_t> > deformation){
  //create a square image that fits the extents of the subet
  const int_t w = image->width();
  const int_t h = image->height();
  intensity_t * intensities = new intensity_t[w*h];
  for(int_t m=0;m<h;++m){
    for(int_t n=0;n<w;++n){
      intensities[m*w+n] = (*image)(n,m);
    }
  }
  if(deformation!=Teuchos::null){
    const scalar_t u = (*deformation)[DISPLACEMENT_X];
    const scalar_t v = (*deformation)[DISPLACEMENT_Y];
    const scalar_t t = (*deformation)[ROTATION_Z];
    const scalar_t ex = (*deformation)[NORMAL_STRAIN_X];
    const scalar_t ey = (*deformation)[NORMAL_STRAIN_Y];
    const scalar_t g = (*deformation)[SHEAR_STRAIN_XY];
    scalar_t dx=0.0,dy=0.0;
    scalar_t Dx=0.0,Dy=0.0;
    scalar_t mapped_x=0.0,mapped_y=0.0;
    int_t px=0,py=0;
    for(int_t i=0;i<num_pixels_;++i){
      // compute the deformed shape:
      // need to cast the x_ and y_ values since the resulting value could be negative
      dx = (scalar_t)(x(i)) - cx_;
      dy = (scalar_t)(y(i)) - cy_;
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
      intensities[py*w+px] = !is_active(i) ? 255
          : is_deactivated_this_step(i) ?  0
          : std::abs((def_intensities(i) - ref_intensities(i))*2);
    }
  }
  else{ // write the original shape of the subset
    for(int_t i=0;i<num_pixels_;++i)
      intensities[y(i)*w+x(i)] = 255;
  }
  utils::write_image(file_name.c_str(),w,h,intensities,true);
  delete[] intensities;
}

void
Subset::write_tiff(const std::string & file_name,
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
  intensity_t * intensities = new intensity_t[w*h];
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
  utils::write_image(file_name.c_str(),w,h,intensities,true);
  delete[] intensities;
}

scalar_t
Subset::noise_std_dev(Teuchos::RCP<Image> image,
  Teuchos::RCP<const std::vector<scalar_t> > deformation){

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

  // move the window to the deformed location using u and v
  const scalar_t u = (*deformation)[DISPLACEMENT_X];
  const scalar_t v = (*deformation)[DISPLACEMENT_Y];
  min_x += u; max_x += u;
  min_y += v; max_y += v;

  DEBUG_MSG("Subset::noise_std_dev(): Extents of subset " << min_x << " " << max_x << " " << min_y << " " << max_y);
  const int_t h = max_y - min_y + 1;
  const int_t w = max_x - min_x + 1;
  const int_t img_h = image->height();
  const int_t img_w = image->width();

  // ensure that the subset falls inside the image
  if(max_x >= image->width() || min_x < 0 || max_y >= image->height() || min_y < 0){
    return 1.0;
  }

  scalar_t variance = 0.0;
  scalar_t conv_i = 0.0;
  // convolve and sum the intensities with the mask
  for(int_t y=min_y; y<max_y;++y){
    for(int_t x=min_x; x<max_x;++x){
      // don't convolve the edge pixels
      if(x<1||x>=img_w-1||y<1||y>=img_h-1){
        variance += std::abs((*image)(x,y));
      }
      else{
        conv_i = 0.0;
        for(int_t j=0;j<3;++j){
          for(int_t i=0;i<3;++i){
            conv_i += (*image)(x+(i-1),y+(j-1))*mask[i][j];
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

}// End DICe Namespace
