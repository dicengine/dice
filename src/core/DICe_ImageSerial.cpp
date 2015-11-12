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
#include <DICe_ImageUtils.h>
#include <DICe_Tiff.h>
#include <DICe_Rawi.h>
#include <DICe_Shape.h>

#include <cassert>

namespace DICe {

Image::Image(const char * file_name,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  offset_x_(0),
  offset_y_(0),
  intensity_rcp_(Teuchos::null),
  has_gradients_(false),
  file_name_(file_name)
{
  const std::string rawi(".rawi");
  bool is_rawi = file_name_.find(rawi)!=std::string::npos;

  if(is_rawi){
    utils::read_rawi_image_dimensions(file_name,width_,height_);
    assert(width_>0);
    assert(height_>0);
    intensities_ = Teuchos::ArrayRCP<intensity_t>(width_*height_,0.0);
    utils::read_rawi_image(file_name,intensities_.getRawPtr());
  }
  // assumes that it is a tiff image as default
  else {
    // get the image dims
    utils::read_tiff_image_dimensions(file_name,width_,height_);
    assert(width_>0);
    assert(height_>0);
    // initialize the pixel containers
    intensities_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,0.0);
    // read in the image
    utils::read_tiff_image(file_name,intensities_.getRawPtr());
  }
  // copy the image to the device (no-op for OpenMP)
  default_constructor_tasks(params);
}

Image::Image(const char * file_name,
  const int_t offset_x,
  const int_t offset_y,
  const int_t width,
  const int_t height,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  width_(width),
  height_(height),
  offset_x_(offset_x),
  offset_y_(offset_y),
  intensity_rcp_(Teuchos::null),
  has_gradients_(false),
  file_name_(file_name)
{
  const std::string rawi(".rawi");
  bool is_rawi = file_name_.find(rawi)!=std::string::npos;
  TEUCHOS_TEST_FOR_EXCEPTION(is_rawi,std::invalid_argument,
    "Error: .rawi files not yet supported for reading only a poriton of the image.");

  // get the image dims
  int_t img_width = 0;
  int_t img_height = 0;
  utils::read_tiff_image_dimensions(file_name,img_width,img_height);
  assert(width_>0&&offset_x_+width_<img_width);
  assert(height_>0&&offset_y_+height_<img_height);
  // initialize the pixel containers
  intensities_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,0.0);
  // read in the image
  utils::read_tiff_image(file_name,
    offset_x,offset_y,
    width_,height_,
    intensities_.getRawPtr());
  default_constructor_tasks(params);
}

Image::Image(const int_t width,
  const int_t height):
  width_(width),
  height_(height),
  offset_x_(0),
  offset_y_(0),
  intensity_rcp_(Teuchos::null),
  has_gradients_(false),
  file_name_("(from array)")
{
  assert(height_>0);
  assert(width_>0);
  intensities_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,0.0);
  default_constructor_tasks(Teuchos::null);
}

Image::Image(Teuchos::RCP<Image> img,
  const int_t offset_x,
  const int_t offset_y,
  const int_t width,
  const int_t height):
  width_(width),
  height_(height),
  offset_x_(offset_x),
  offset_y_(offset_y),
  intensity_rcp_(Teuchos::null),
  has_gradients_(img->has_gradients()),
  file_name_(img->file_name())
{
  TEUCHOS_TEST_FOR_EXCEPTION(offset_x_<0,std::invalid_argument,"Error, offset_x_ cannot be negative.");
  TEUCHOS_TEST_FOR_EXCEPTION(offset_y_<0,std::invalid_argument,"Error, offset_x_ cannot be negative.");
  if(width_==-1)
    width_ = img->width();
  if(height_==-1)
    height_ = img->height();
  assert(width_>0);
  assert(height_>0);
  const int_t src_width = img->width();
  const int_t src_height = img->height();

  // initialize the pixel containers
  intensities_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,0.0);
  intensities_temp_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,0.0);
  grad_x_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  grad_y_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  mask_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  // deep copy values over
  int_t src_y=0, src_x=0;
  for(int_t y=0;y<height_;++y){
    src_y = y + offset_y_;
    for(int_t x=0;x<width_;++x){
      src_x = x + offset_x_;
      if(src_x>=0&&src_x<src_width&&src_y>=0&&src_y<src_height){
        intensities_[y*width_+x] = (*img)(src_x,src_y);
        grad_x_[y*width_+x] = img->grad_x(src_x,src_y);
        grad_y_[y*width_+x] = img->grad_y(src_x,src_y);
        mask_[y*width_+x] = img->mask(src_x,src_y);
      }
      else{
        intensities_[y*width_+x] = 0.0;
        grad_x_[y*width_+x] = 0.0;
        grad_y_[y*width_+x] = 0.0;
        mask_[y*width_+x] = 1.0;
      }
    }
  }
  grad_c1_ = 1.0/12.0;
  grad_c2_ = -8.0/12.0;
  gauss_filter_mask_size_ = img->gauss_filter_mask_size();
  gauss_filter_half_mask_ = gauss_filter_mask_size_/2+1;
}

void
Image::initialize_array_image(intensity_t * intensities){
  assert(width_>0);
  assert(height_>0);
  intensities_ = Teuchos::ArrayRCP<intensity_t>(intensities,0,width_*height_,false);
}

void
Image::default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){

  grad_x_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  grad_y_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
  intensities_temp_ = Teuchos::ArrayRCP<intensity_t>(height_*width_,0.0);
  // image gradient coefficients
  grad_c1_ = 1.0/12.0;
  grad_c2_ = -8.0/12.0;
  const bool compute_image_gradients = params!=Teuchos::null ?
      params->get<bool>(DICe::compute_image_gradients,false) : false;
  const bool image_grad_use_hierarchical_parallelism = params!=Teuchos::null ?
      params->get<bool>(DICe::image_grad_use_hierarchical_parallelism,false) : false;
  const int image_grad_team_size = params!=Teuchos::null ?
      params->get<int>(DICe::image_grad_team_size,256) : 256;
  if(compute_image_gradients)
    compute_gradients(image_grad_use_hierarchical_parallelism,image_grad_team_size);
  const bool gauss_filter_image = params!=Teuchos::null ?
      params->get<bool>(DICe::gauss_filter_images,false) : false;
  const bool gauss_filter_use_hierarchical_parallelism = params!=Teuchos::null ?
      params->get<bool>(DICe::gauss_filter_use_hierarchical_parallelism,false) : false;
  const int gauss_filter_team_size = params!=Teuchos::null ?
      params->get<int>(DICe::gauss_filter_team_size,256) : 256;
  gauss_filter_mask_size_ = params!=Teuchos::null ?
      params->get<int>(DICe::gauss_filter_mask_size,7) : 7;
  gauss_filter_half_mask_ = gauss_filter_mask_size_/2+1;
  if(gauss_filter_image){
    gauss_filter(gauss_filter_use_hierarchical_parallelism,gauss_filter_team_size);
  }
  mask_ = Teuchos::ArrayRCP<scalar_t>(height_*width_,0.0);
}

const intensity_t&
Image::operator()(const int_t x, const int_t y) const {
  return intensities_[y*width_+x];
}

const intensity_t&
Image::operator()(const int_t i) const {
  return intensities_[i];
}

const scalar_t&
Image::grad_x(const int_t x,
  const int_t y) const {
  return grad_x_[y*width_+x];
}

const scalar_t&
Image::grad_y(const int_t x,
  const int_t y) const {
  return grad_y_[y*width_+x];
}

const scalar_t&
Image::mask(const int_t x,
  const int_t y) const {
  return mask_[y*width_+x];
}

Teuchos::ArrayRCP<intensity_t>
Image::intensity_array()const{
  // note: in the kokkos version a copy of the intensities is returned while in the serial version
  // the intensity array itself is returned (therefor giving access to changing it's values...)
  return intensities_;
}

void
Image::compute_gradients(const bool use_hierarchical_parallelism, const int_t team_size){
  for(int_t y=0;y<height_;++y){
    for(int_t x=0;x<width_;++x){
      if(x<2){
        grad_x_[y*width_+x] = intensities_[y*width_+x+1] - intensities_[y*width_+x];
      }
      /// check if this pixel is near the right edge
      else if(x>=width_-2){
        grad_x_[y*width_+x] = intensities_[y*width_+x] - intensities_[y*width_+x-1];
      }
      else{
        grad_x_[y*width_+x] = grad_c1_*intensities_[y*width_+x-2] + grad_c2_*intensities_[y*width_+x-1]
            - grad_c2_*intensities_[y*width_+x+1] - grad_c1_*intensities_[y*width_+x+2];
      }
      /// check if this pixel is near the top edge
      if(y<2){
        grad_y_[y*width_+x] = intensities_[(y+1)*width_+x] - intensities_[y*width_+x];
      }
      /// check if this pixel is near the bottom edge
      else if(y>=height_-2){
        grad_y_[y*width_+x] = intensities_[y*width_+x] - intensities_[(y-1)*width_+x];
      }
      else{
        grad_y_[y*width_+x] = grad_c1_*intensities_[(y-2)*width_+x] + grad_c2_*intensities_[(y-1)*width_+x]
            - grad_c2_*intensities_[(y+1)*width_+x] - grad_c1_*intensities_[(y+2)*width_+x];
      }
    }
  }
  has_gradients_ = true;
}

void
Image::apply_mask(const Conformal_Area_Def & area_def,
  const bool smooth_edges){
  // first create the mask:
  create_mask(area_def,smooth_edges);
  for(int_t i=0;i<num_pixels();++i)
    intensities_[i] = mask_[i]*intensities_[i];
}

void
Image::apply_mask(const bool smooth_edges){
  if(smooth_edges){
    static scalar_t smoothing_coeffs[5][5];
    std::vector<scalar_t> coeffs(5,0.0);
    coeffs[0] = 0.0014;coeffs[1] = 0.1574;coeffs[2] = 0.62825;
    coeffs[3] = 0.1574;coeffs[4] = 0.0014;
    for(int_t j=0;j<5;++j){
      for(int_t i=0;i<5;++i){
        smoothing_coeffs[i][j] = coeffs[i]*coeffs[j];
      }
    }
    Teuchos::ArrayRCP<scalar_t> mask_tmp(mask_.size(),0.0);
    for(int_t i=0;i<mask_tmp.size();++i)
      mask_tmp[i] = mask_[i];
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        if(x>=2&&x<width_-2&&y>=2&&y<height_-2){ // 2 is half the gauss_mask size
          scalar_t value = 0.0;
          for(int_t i=0;i<5;++i){
            for(int_t j=0;j<5;++j){
              // assumes intensity values have already been deep copied into mask_tmp_
              value += smoothing_coeffs[i][j]*mask_tmp[(y+(j-2))*width_+x+(i-2)];
            } //j
          } //i
          mask_[y*width_+x] = value;
        }else{
          mask_[y*width_+x] = mask_tmp[y*width_+x];
        }
      } // x
    } // y
  } // smooth edges
  for(int_t i=0;i<num_pixels();++i)
    intensities_[i] = mask_[i]*intensities_[i];
}

void
Image::create_mask(const Conformal_Area_Def & area_def,
  const bool smooth_edges){
  assert(area_def.has_boundary());
  std::set<std::pair<int_t,int_t> > coords;
  for(size_t i=0;i<area_def.boundary()->size();++i){
    std::set<std::pair<int_t,int_t> > shapeCoords = (*area_def.boundary())[i]->get_owned_pixels();
    coords.insert(shapeCoords.begin(),shapeCoords.end());
  }
  // now remove any excluded regions:
  // now set the inactive bit for the second set of multishapes if they exist.
  if(area_def.has_excluded_area()){
    for(size_t i=0;i<area_def.excluded_area()->size();++i){
      std::set<std::pair<int_t,int_t> > removeCoords = (*area_def.excluded_area())[i]->get_owned_pixels();
      typename std::set<std::pair<int_t,int_t> >::iterator it = removeCoords.begin();
      for(;it!=removeCoords.end();++it){
        if(coords.find(*it)!=coords.end())
          coords.erase(*it);
      } // end removeCoords loop
    } // end excluded_area loop
  } // end has excluded area
  // NOTE: the pairs are (y,x) not (x,y) so that the ordering is correct in the set
  typename std::set<std::pair<int_t,int_t> >::iterator set_it = coords.begin();
  for( ; set_it!=coords.end();++set_it){
    mask_[(set_it->first - offset_y_)*width_+set_it->second - offset_x_] = 1.0;
  }
  if(smooth_edges){
    static scalar_t smoothing_coeffs[5][5];
    std::vector<scalar_t> coeffs(5,0.0);
    coeffs[0] = 0.0014;coeffs[1] = 0.1574;coeffs[2] = 0.62825;
    coeffs[3] = 0.1574;coeffs[4] = 0.0014;
    for(int_t j=0;j<5;++j){
      for(int_t i=0;i<5;++i){
        smoothing_coeffs[i][j] = coeffs[i]*coeffs[j];
      }
    }
    Teuchos::ArrayRCP<scalar_t> mask_tmp(mask_.size(),0.0);
    for(int_t i=0;i<mask_tmp.size();++i)
      mask_tmp[i] = mask_[i];
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        if(x>=2&&x<width_-2&&y>=2&&y<height_-2){ // 2 is half the gauss_mask size
          scalar_t value = 0.0;
          for(int_t i=0;i<5;++i){
            for(int_t j=0;j<5;++j){
              // assumes intensity values have already been deep copied into mask_tmp_
              value += smoothing_coeffs[i][j]*mask_tmp[(y+(j-2))*width_+x+(i-2)];
            } //j
          } //i
          mask_[y*width_+x] = value;
        }else{
          mask_[y*width_+x] = mask_tmp[y*width_+x];
        }
      } // x
    } // y
  }
}

Teuchos::RCP<Image>
Image::apply_transformation(Teuchos::RCP<const std::vector<scalar_t> > deformation,
  const int_t cx,
  const int_t cy,
  const bool apply_in_place){

  if(apply_in_place){
    for(int_t i=0;i<num_pixels();++i)
      intensities_temp_[i] = intensities_[i];
    apply_transform(intensities_temp_,intensities_,cx,cy,width_,height_,deformation);
    return Teuchos::null;
  }
  else{
    Teuchos::ArrayRCP<intensity_t> result_intensities(width_*height_,0.0);
    apply_transform(intensities_,result_intensities,cx,cy,width_,height_,deformation);
    Teuchos::RCP<Image> result = Teuchos::rcp(new Image(width_,height_,result_intensities));
    return result;
  }
}

void
Image::gauss_filter(const bool use_hierarchical_parallelism,
  const int_t team_size){

  std::vector<scalar_t> coeffs(13,0.0);

  // make sure the mask size is appropriate
  if(gauss_filter_mask_size_==5){
    coeffs[0] = 0.0014;coeffs[1] = 0.1574;coeffs[2] = 0.62825;
    coeffs[3] = 0.1574;coeffs[4] = 0.0014;
  }
  else if (gauss_filter_mask_size_==7){
    coeffs[0] = 0.0060;coeffs[1] = 0.0606;coeffs[2] = 0.2418;
    coeffs[3] = 0.3831;coeffs[4] = 0.2418;coeffs[5] = 0.0606;
    coeffs[6] = 0.0060;
  }
  else if (gauss_filter_mask_size_==9){
    coeffs[0] = 0.0007;coeffs[1] = 0.0108;coeffs[2] = 0.0748;
    coeffs[3] = 0.2384;coeffs[4] = 0.3505;coeffs[5] = 0.2384;
    coeffs[6] = 0.0748;coeffs[7] = 0.0108;coeffs[8] = 0.0007;
  }
  else if (gauss_filter_mask_size_==11){
    coeffs[0] = 0.0001;coeffs[1] = 0.0017;coeffs[2] = 0.0168;
    coeffs[3] = 0.0870;coeffs[4] = 0.2328;coeffs[5] = 0.3231;
    coeffs[6] = 0.2328;coeffs[7] = 0.0870;coeffs[8] = 0.0168;
    coeffs[9] = 0.0017;coeffs[10] = 0.0001;
  }
  else if (gauss_filter_mask_size_==13){
    coeffs[0] = 0.0001;coeffs[1] = 0.0012;coeffs[2] = 0.0085;
    coeffs[3] = 0.0380;coeffs[4] = 0.1109;coeffs[5] = 0.2108;
    coeffs[6] = 0.2611;coeffs[7] = 0.2108;coeffs[8] = 0.1109;
    coeffs[9] = 0.0380;coeffs[10] = 0.0085;coeffs[11] = 0.0012;
    coeffs[12] = 0.0001;
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,
      "Error, the Gauss filter mask size is invalid (options include 5,7,9,11,13)");
  }

  for(int_t j=0;j<gauss_filter_mask_size_;++j){
    for(int_t i=0;i<gauss_filter_mask_size_;++i){
      gauss_filter_coeffs_[i][j] = coeffs[i]*coeffs[j];
    }
  }
  for(int_t y=0;y<height_;++y){
    for(int_t x=0;x<width_;++x){
      if(x>=gauss_filter_half_mask_&&x<width_-gauss_filter_half_mask_&&y>=gauss_filter_half_mask_&&y<height_-gauss_filter_half_mask_){
        intensity_t value = 0.0;
        for(int_t i=0;i<gauss_filter_mask_size_;++i){
          for(int_t j=0;j<gauss_filter_mask_size_;++j){
            // assumes intensity values have already been deep copied into intensities_temp_
            value += gauss_filter_coeffs_[i][j]*intensities_temp_[(y+(j-gauss_filter_half_mask_))*width_+x+(i-gauss_filter_half_mask_)];
          } //j
        } //i
        intensities_[y*width_+x] = value;
      }
    }
  }
}

}// End DICe Namespace
