// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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

#include <DICe_Image.h>
#include <DICe_ImageIO.h>
#include <DICe_Shape.h>
#include <DICe_ImageFunctors.h>

#include <cassert>

namespace DICe {

Image::Image(const char * file_name,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  offset_x_(0),
  offset_y_(0),
  intensity_rcp_(Teuchos::null),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_(file_name),
  has_file_name_(true),
  gradient_method_(FINITE_DIFFERENCE)
{
  try{
    utils::read_image_dimensions(file_name,width_,height_);
    TEUCHOS_TEST_FOR_EXCEPTION(width_<=0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(height_<=0,std::runtime_error,"");
    intensities_ = intensity_dual_view_2d("intensities",height_,width_);
    utils::read_image(file_name,
      intensities_.h_view.ptr_on_device(),
      default_is_layout_right());
  }
  catch(std::exception & e){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, image file read failure");
  }
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
  has_gauss_filter_(false),
  file_name_(file_name),
  has_file_name_(true),
  gradient_method_(FINITE_DIFFERENCE)
{
  // get the image dims
  int_t img_width = 0;
  int_t img_height = 0;
  try{
    utils::read_image_dimensions(file_name,img_width,img_height);
    TEUCHOS_TEST_FOR_EXCEPTION(width_<=0||offset_x_+width_>img_width,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(height_<=0||offset_y_+height_>img_height,std::runtime_error,"");
    // initialize the pixel containers
    intensities_ = intensity_dual_view_2d("intensities",height_,width_);
    // read in the image
    utils::read_image(file_name,
      offset_x,offset_y,
      width_,height_,
      intensities_.h_view.ptr_on_device(),
      default_is_layout_right());
  }
  catch(std::exception & e){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, image file read failure");
  }
  default_constructor_tasks(params);
}


Image::Image(const int_t width,
  const int_t height,
  const intensity_t intensity,
  const int_t offset_x,
  const int_t offset_y):
  width_(width),
  height_(height),
  offset_x_(offset_x),
  offset_y_(offset_y),
  intensity_rcp_(Teuchos::null),
  has_gradients_(false),
  has_gauss_filter_(false),
  file_name_("(from array)"),
  has_file_name_(false),
  gradient_method_(FINITE_DIFFERENCE)
{
  assert(height_>0);
  assert(width_>0);
  intensities_ = intensity_dual_view_2d("intensities",height_,width_);
  for(int_t y=0;y<height_;++y){
    for(int_t x=0;x<width_;++x){
      intensities_.h_view(y,x) = intensity;
    }
  }
  default_constructor_tasks(Teuchos::null);
}

Image::Image(Teuchos::RCP<Image> img,
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
  has_gradients_(img->has_gradients()),
  has_gauss_filter_(img->has_gauss_filter()),
  file_name_(img->file_name()),
  has_file_name_(img->has_file_name()),
  gradient_method_(FINITE_DIFFERENCE)
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
  intensities_ = intensity_dual_view_2d("intensities",height_,width_);
  intensities_temp_ = intensity_device_view_2d("intensities_temp",height_,width_);
  grad_x_ = scalar_dual_view_2d("grad_x",height_,width_);
  grad_y_ = scalar_dual_view_2d("grad_y",height_,width_);
  mask_ = scalar_dual_view_2d("mask",height_,width_);
  // deep copy values over
  int_t src_y=0, src_x=0;
  for(int_t y=0;y<height_;++y){
    src_y = y + offset_y_;
    for(int_t x=0;x<width_;++x){
      src_x = x + offset_x_;
      if(src_x>=0&&src_x<src_width&&src_y>=0&&src_y<src_height){
        intensities_.h_view(y,x) = img->intensity_dual_view().h_view(src_y,src_x);
        grad_x_.h_view(y,x) = img->grad_x().h_view(src_y,src_x);
        grad_y_.h_view(y,x) = img->grad_y().h_view(src_y,src_x);
        mask_.h_view(y,x) = img->mask().h_view(src_y,src_x);
      }
      else{
        intensities_.h_view(y,x) = 0.0;
        grad_x_.h_view(y,x) = 0.0;
        grad_y_.h_view(y,x) = 0.0;
        mask_.h_view(y,x) = 1.0;
      }
    }
  }
  intensities_.modify<host_space>();
  intensities_.sync<device_space>();
  grad_x_.modify<host_space>();
  grad_x_.sync<device_space>();
  grad_y_.modify<host_space>();
  grad_y_.sync<device_space>();
  mask_.modify<host_space>();
  mask_.sync<device_space>();
  grad_c1_ = 1.0/12.0;
  grad_c2_ = -8.0/12.0;
  gauss_filter_mask_size_ = img->gauss_filter_mask_size();
  gauss_filter_half_mask_ = gauss_filter_mask_size_/2+1;
  if(params!=Teuchos::null){
    if(params->isParameter(DICe::gauss_filter_mask_size)){
      gauss_filter_mask_size_ = params->get<int>(DICe::gauss_filter_mask_size,7);
      gauss_filter_half_mask_ = gauss_filter_mask_size_/2+1;
    }
    if(params->isParameter(DICe::gauss_filter_images)){
        if(!img->has_gauss_filter()&&params->get<bool>(DICe::gauss_filter_images,false)){
          // if the image was not filtered, but requested here, filter the image
          DEBUG_MSG("Image filter requested, but origin image does not have gauss filter, applying gauss filter here");
          gauss_filter();
        }
    }
    if(params->isParameter(DICe::compute_image_gradients)){
      if(params->isParameter(DICe::gradient_method)){
        gradient_method_ = params->get<Gradient_Method>(DICe::gradient_method);
      }
      if(!img->has_gradients()&&params->get<bool>(DICe::compute_image_gradients,false)){
        // if gradients have not been computed, but ther are requested here, compute them
        DEBUG_MSG("Image gradients requested, but origin image does not have gradients, computing them here");
        compute_gradients();
      }
    }
  }
}

void
Image::initialize_array_image(intensity_t * intensities){
  assert(width_>0);
  assert(height_>0);
  // initialize the pixel containers
  // if the default layout is LayoutRight, then no permutation is necessary
  if(default_is_layout_right()){
    intensity_device_view_2d intensities_dev(intensities,height_,width_);
    intensity_host_view_2d intensities_host  = Kokkos::create_mirror_view(intensities_dev);
    intensities_ = intensity_dual_view_2d(intensities_dev,intensities_host);
  }
  // else the data has to be copied to coalesce with the default layout
  else{
    assert(!default_is_layout_right());
    intensities_ = intensity_dual_view_2d("intensities",height_,width_);
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<width_;++x){
        intensities_.h_view(y,x) = intensities[y*width_+x];
      }
    }
  }
}

void
Image::default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){

  grad_x_ = scalar_dual_view_2d("grad_x",height_,width_);
  grad_y_ = scalar_dual_view_2d("grad_y",height_,width_);
  // copy the image to the device (no-op for OpenMP)
  intensities_.modify<host_space>(); // The template is where the modification took place
  intensities_.sync<device_space>(); // The template is what needs to be synced
  // create a temp container for the pixel intensities
  intensities_temp_ = intensity_device_view_2d("intensities_temp",height_,width_);
  // image gradient coefficients
  grad_c1_ = 1.0/12.0;
  grad_c2_ = -8.0/12.0;
  // create the image mask arrays
  mask_ = scalar_dual_view_2d("mask",height_,width_);
  // initialize the image mask arrays
  Kokkos::parallel_for(Kokkos::RangePolicy<Init_Mask_Tag>(0,num_pixels()),*this);
  mask_.modify<device_space>();
  mask_.sync<host_space>();
  post_allocation_tasks(params);
}

const intensity_t&
Image::operator()(const int_t x, const int_t y) const {
  return intensities_.h_view(y,x);
}

const intensity_t&
Image::operator()(const int_t i) const {
  const int_t y = i / width_;
  const int_t x = i - y*width_;
  return intensities_.h_view(y,x);
}

const scalar_t&
Image::grad_x(const int_t x,
  const int_t y) const {
  return grad_x_.h_view(y,x);
}

const scalar_t&
Image::grad_y(const int_t x,
  const int_t y) const {
  return grad_y_.h_view(y,x);
}

const scalar_t&
Image::mask(const int_t x,
  const int_t y) const {
  return mask_.h_view(y,x);
}

Teuchos::ArrayRCP<scalar_t>
Image::grad_x_array()const{
  Teuchos::ArrayRCP<scalar_t> array(grad_x_.h_view.ptr_on_device(),0,width_*height_,false);
  return array;
}

Teuchos::ArrayRCP<scalar_t>
Image::grad_y_array()const{
  Teuchos::ArrayRCP<scalar_t> array(grad_y_.h_view.ptr_on_device(),0,width_*height_,false);
  return array;
}

Teuchos::ArrayRCP<intensity_t>
Image::intensities()const{
  Teuchos::ArrayRCP<intensity_t> array(intensities_.h_view.ptr_on_device(),0,width_*height_,false);
  return array;
}

intensity_t
Image::interpolate_keys_fourth(const scalar_t & local_x, const scalar_t & local_y){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, method not implemented yet.");
}

scalar_t
Image::interpolate_grad_x_keys_fourth(const scalar_t & local_x, const scalar_t & local_y){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, method not implemented yet.");
}

scalar_t
Image::interpolate_grad_y_keys_fourth(const scalar_t & local_x, const scalar_t & local_y){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, method not implemented yet.");
}

intensity_t
Image::interpolate_bilinear(const scalar_t & local_x, const scalar_t & local_y){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, method not implemented yet.");
}

scalar_t
Image::interpolate_grad_x_bilinear(const scalar_t & local_x, const scalar_t & local_y){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, method not implemented yet.");
}

scalar_t
Image::interpolate_grad_y_bilinear(const scalar_t & local_x, const scalar_t & local_y){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, method not implemented yet.");
}

intensity_t
Image::interpolate_bicubic(const scalar_t & local_x, const scalar_t & local_y){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, method not implemented yet.");
}

scalar_t
Image::interpolate_grad_x_bicubic(const scalar_t & local_x, const scalar_t & local_y){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, method not implemented yet.");
}

scalar_t
Image::interpolate_grad_y_bicubic(const scalar_t & local_x, const scalar_t & local_y){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, method not implemented yet.");
}

void
Image::smooth_gradients_convolution_5_point(){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, this method should not be called");
}

/// compute the image gradients
void
Image::compute_gradients_finite_difference(){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, this method should not be called");
}

void
Image::compute_gradients(const bool use_hierarchical_parallelism, const int_t team_size){
  TEUCHOS_TEST_FOR_EXCEPTION(gradient_method_!=FINITE_DIFFERENCE,std::runtime_error,
    "Error, gradient method must be FINITE_DIFFERENCE (this is the only method implemented for Kokkos");
  // Flat gradients:
  if(use_hierarchical_parallelism)
    Kokkos::parallel_for(Kokkos::TeamPolicy<Grad_Tag>(height_,team_size),*this);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<Grad_Flat_Tag>(0,num_pixels()),*this);
  grad_x_.modify<device_space>();
  grad_x_.sync<host_space>();
  grad_y_.modify<device_space>();
  grad_y_.sync<host_space>();
  has_gradients_ = true;
}

void
Image::apply_mask(const Conformal_Area_Def & area_def,
  const bool smooth_edges){
  // first create the mask:
  create_mask(area_def,smooth_edges);
  // then apply it to the image intensity values
  Mask_Apply_Functor apply_functor(intensities_.d_view,mask_.d_view,width_);
  Kokkos::parallel_for(width_*height_,apply_functor);
  intensities_.modify<device_space>();
  intensities_.sync<host_space>();
}

void
Image::apply_mask(const bool smooth_edges){
  // make sure the mask is synced from host to device
  mask_.modify<host_space>();
  mask_.sync<device_space>();
  if(smooth_edges){
    // smooth the edges of the mask:
    Mask_Smoothing_Functor smoother(mask_,width_,height_);
    Kokkos::parallel_for(width_*height_,smoother);
  }
  // then apply it to the image intensity values
  Mask_Apply_Functor apply_functor(intensities_.d_view,mask_.d_view,width_);
  Kokkos::parallel_for(width_*height_,apply_functor);
  mask_.modify<device_space>();
  mask_.sync<host_space>();
  intensities_.modify<device_space>();
  intensities_.sync<host_space>();
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

  // at this point all the coordinate pairs are in the set
  const int_t num_area_pixels = coords.size();
  // resize the storage arrays now that the num_pixels is known
  pixel_coord_device_view_1d x("x",num_area_pixels);
  pixel_coord_device_view_1d y("y",num_area_pixels);
  int_t index = 0;
  // NOTE: the pairs are (y,x) not (x,y) so that the ordering is correct in the set
  typename std::set<std::pair<int_t,int_t> >::iterator set_it = coords.begin();
  for( ; set_it!=coords.end();++set_it){
    x(index) = set_it->second - offset_x_;
    y(index) = set_it->first - offset_y_;
    index++;
  }
  Mask_Init_Functor init_functor(mask_.d_view,x,y);
  Kokkos::parallel_for(num_area_pixels,init_functor);
  //mask_.modify<device_space>();
  //mask_.sync<host_space>();
  if(smooth_edges){
    // smooth the edges of the mask:
    Mask_Smoothing_Functor smoother(mask_,width_,height_);
    Kokkos::parallel_for(width_*height_,smoother);
  }
  mask_.modify<device_space>();
  mask_.sync<host_space>();
}

Teuchos::RCP<Image>
Image::apply_transformation(Teuchos::RCP<Local_Shape_Function> shape_function,
  const int_t cx,
  const int_t cy,
  const bool apply_in_place){

  assert(shape_function->num_params()==6);
  Teuchos::RCP<std::vector<scalar_t> > deformation = shape_function->rcp();

  if(apply_in_place){
    // deep copy the intesity array to the device temporary container
    Kokkos::deep_copy(intensities_temp_,intensities_.d_view);
    Transform_Functor trans_functor(intensities_temp_,
      intensities_.d_view,
      width_,
      height_,
      cx,
      cy,
      deformation);
    Kokkos::parallel_for(width_*height_,trans_functor);
    // sync up the new image
    intensities_.modify<device_space>();
    intensities_.sync<host_space>();
    return Teuchos::null;
  }
  else{
    Teuchos::RCP<Image> result = Teuchos::rcp(new Image(width_,height_));
    Transform_Functor trans_functor(intensities_.d_view,
      result->intensity_dual_view().d_view,
      width_,
      height_,
      cx,
      cy,
      deformation);
    Kokkos::parallel_for(width_*height_,trans_functor);
    // sync up the new image
    result->intensity_dual_view().modify<device_space>();
    result->intensity_dual_view().sync<host_space>();
    return result;
  }
}

void
Image::gauss_filter(const int_t mask_size,const bool use_hierarchical_parallelism,
  const int_t team_size){

  if(mask_size>0) gauss_filter_mask_size_ = mask_size;

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
  // deep copy the intesity array to the device temporary container
  Kokkos::deep_copy(intensities_temp_,intensities_.d_view);
  // Flat gradients:
  if(use_hierarchical_parallelism)
    Kokkos::parallel_for(Kokkos::TeamPolicy<Gauss_Tag>(height_,team_size),*this);
  else   // Hierarchical filtering:
    Kokkos::parallel_for(Kokkos::RangePolicy<Gauss_Flat_Tag>(0,num_pixels()),*this);
  // copy the intensity array back to the host
  intensities_.modify<device_space>();
  intensities_.sync<host_space>();
  has_gauss_filter_ = true;
}

//
// Kokkos functors
//
Mask_Smoothing_Functor::Mask_Smoothing_Functor(scalar_dual_view_2d mask,
  const int_t width,
  const int_t height):
  mask_(mask),
  width_(width),
  height_(height){
  // create the device array to hold a copy of the mask values
  mask_tmp_ = scalar_device_view_2d("mask_tmp",height_,width_);
  Kokkos::deep_copy(mask_tmp_,mask_.d_view);
  // set up the coefficients:
  std::vector<scalar_t> coeffs(5,0.0);
  coeffs[0] = 0.0014;coeffs[1] = 0.1574;coeffs[2] = 0.62825;
  coeffs[3] = 0.1574;coeffs[4] = 0.0014;
  for(int_t j=0;j<5;++j){
    for(int_t i=0;i<5;++i){
      gauss_filter_coeffs_[i][j] = coeffs[i]*coeffs[j];
    }
  }
};

KOKKOS_INLINE_FUNCTION
void
Transform_Functor::operator()(const int_t pixel_index) const{
  // determine the x and y coordinates of this pixel
  // after taking out the displacements
  const int_t y = pixel_index / width_;
  const int_t x = pixel_index - y*width_;
  // recall that cx_ and cy_ are in the transformed coordinates
  const scalar_t dx = x - cx_;
  const scalar_t dy = y - cy_;
  const scalar_t Dx = (1.0-ex_)*dx - g_*dy;
  const scalar_t Dy = (1.0-ey_)*dy - g_*dx;
  const scalar_t mapped_x = cost_*Dx - sint_*Dy - u_ + cx_;
  const scalar_t mapped_y = sint_*Dx + cost_*Dy - v_ + cy_;
  // determine the current pixel the coordinates fall in:
  int_t px = (int_t)mapped_x;
  if(mapped_x - px >= 0.5) px++;
  int_t py = (int_t)mapped_y;
  if(mapped_y - py >= 0.5) py++;

  // check that the mapped location is inside the image...
  if(mapped_x>2.5&&mapped_x<width_-3.5&&mapped_y>2.5&&mapped_y<height_-3.5){
    intensity_t intensity_value = 0.0;
    // convolve all the pixels within + and - pixels of the point in question
    scalar_t dx=0.0, dy=0.0;
    scalar_t dx2=0.0, dx3=0.0;
    scalar_t dy2=0.0, dy3=0.0;
    scalar_t f0x=0.0, f0y=0.0;
    for(int_t j=py-3;j<=py+3;++j){
      dy = std::abs(mapped_y - j);
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
      for(int_t i=px-3;i<=px+3;++i){
        // compute the f's of x and y
        dx = std::abs(mapped_x - i);
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
        intensity_value += intensities_from_(j,i)*f0x*f0y;
      }
    }
    intensities_to_(y,x) = intensity_value;
  }
  else if(mapped_x>=0&&mapped_x<width_-1.5&&mapped_y>=0&&mapped_y<height_-1.5){
    int_t x1 = (int_t)mapped_x;
    int_t x2 = x1+1;
    int_t y1 = (int_t)mapped_y;
    int_t y2  = y1+1;
    intensities_to_(y,x) =
        (intensities_from_(y1,x1)*(x2-mapped_x)*(y2-mapped_y)
         +intensities_from_(y1,x2)*(mapped_x-x1)*(y2-mapped_y)
         +intensities_from_(y2,x2)*(mapped_x-x1)*(mapped_y-y1)
         +intensities_from_(y2,x1)*(x2-mapped_x)*(mapped_y-y1));
  }
  else{
    intensities_to_(y,x) = 0;
  }
}

KOKKOS_INLINE_FUNCTION
void
Image::operator()(const Grad_Flat_Tag &, const int_t pixel_index)const{
  const int_t y = pixel_index / width_;
  const int_t x = pixel_index - y*width_;
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

KOKKOS_INLINE_FUNCTION
void
Image::operator()(const Init_Mask_Tag &, const int_t pixel_index)const{
  const int_t y = pixel_index / width_;
  const int_t x = pixel_index - y*width_;
  mask_.d_view(y,x) = 0.0;
}

KOKKOS_INLINE_FUNCTION
void
Image::operator()(const Grad_Tag &, const member_type team_member)const{
  const int_t row = team_member.league_rank();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, width_),
    [=] (const int_t col){
    if(col<2){
      grad_x_.d_view(row,col) = intensities_.d_view(row,col+1) - intensities_.d_view(row,col);
    }
    /// check if this pixel is near the right edge
    else if(col>=width_-2){
      grad_x_.d_view(row,col) = intensities_.d_view(row,col) - intensities_.d_view(row,col-1);
    }
    else{
      grad_x_.d_view(row,col) = grad_c1_*intensities_.d_view(row,col-2) + grad_c2_*intensities_.d_view(row,col-1)
          - grad_c2_*intensities_.d_view(row,col+1) - grad_c1_*intensities_.d_view(row,col+2);
    }
    /// check if this pixel is near the top edge
    if(row<2){
      grad_y_.d_view(row,col) = intensities_.d_view(row+1,col) - intensities_.d_view(row,col);
    }
    /// check if this pixel is near the bottom edge
    else if(row>=height_-2){
      grad_y_.d_view(row,col) = intensities_.d_view(row,col) - intensities_.d_view(row-1,col);
    }
    else{
      grad_y_.d_view(row,col) = grad_c1_*intensities_.d_view(row-2,col) + grad_c2_*intensities_.d_view(row-1,col)
          - grad_c2_*intensities_.d_view(row+1,col) - grad_c1_*intensities_.d_view(row+2,col);
    }
  });
}

KOKKOS_INLINE_FUNCTION
void
Image::operator()(const Gauss_Flat_Tag &, const int_t pixel_index)const{
  const int_t y = pixel_index / width_;
  const int_t x = pixel_index - y*width_;
  if(x>=gauss_filter_half_mask_&&x<width_-gauss_filter_half_mask_&&y>=gauss_filter_half_mask_&&y<height_-gauss_filter_half_mask_){
    intensity_t value = 0.0;
    for(int_t i=0;i<gauss_filter_mask_size_;++i){
      for(int_t j=0;j<gauss_filter_mask_size_;++j){
        // assumes intensity values have already been deep copied into intensities_temp_
        value += gauss_filter_coeffs_[i][j]*intensities_temp_(y+(j-gauss_filter_half_mask_+1),x+(i-gauss_filter_half_mask_+1));
      } //j
    } //i
    intensities_.d_view(y,x) = value;
  }
}

KOKKOS_INLINE_FUNCTION
void
Image::operator()(const Gauss_Tag &, const member_type team_member)const{
  const int_t row = team_member.league_rank();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, width_),
    [=] (const int_t col){
    if(col>=gauss_filter_half_mask_&&col<width_-gauss_filter_half_mask_&&row>=gauss_filter_half_mask_&&row<height_-gauss_filter_half_mask_){
      intensity_t value = 0.0;
      for(int_t i=0;i<gauss_filter_mask_size_;++i){
        for(int_t j=0;j<gauss_filter_mask_size_;++j){
          // assumes intensity values have already been deep copied into intensities_temp_
          value += gauss_filter_coeffs_[i][j]*intensities_temp_(row+(j-gauss_filter_half_mask_+1),col+(i-gauss_filter_half_mask_+1));
        } //j
      } //i
      intensities_.d_view(row,col) = value;
    }
  });
}


/// operator
KOKKOS_INLINE_FUNCTION
void
Mask_Smoothing_Functor::operator()(const int_t pixel_index)const{
  const int_t y = pixel_index / width_;
  const int_t x = pixel_index - y*width_;
  if(x>=2&&x<width_-2&&y>=2&&y<height_-2){ // 2 is half the gauss_mask size
    scalar_t value = 0.0;
    for(int_t i=0;i<5;++i){
      for(int_t j=0;j<5;++j){
        // assumes intensity values have already been deep copied into mask_tmp_
        value += gauss_filter_coeffs_[i][j]*mask_tmp_(y+(j-2),x+(i-2));
      } //j
    } //i
    mask_.d_view(y,x) = value;
  }else{
    mask_.d_view(y,x) = mask_tmp_(y,x);
  }
}

}// End DICe Namespace
