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

#ifndef DICE_SUBSETFUNCTORS_H
#define DICE_SUBSETFUNCTORS_H

#include <DICe.h>
#include <DICe_Kokkos.h>
#include <DICe_Subset.h>
#include <DICe_Image.h>

namespace DICe {

/// mean value functor
struct Intensity_Sum_Functor{
  /// pointer to the intensity values on the device
  intensity_device_view_1d intensities_;
  /// active pixel flags (persistent)
  bool_device_view_1d is_active_;
  /// active pixel flags (for current step)
  bool_device_view_1d is_deactivated_this_step_;
  /// constructor
  /// \param intensities the image intensity values
  /// \param is_active flags for pixels that are permanently activated or de-activated
  /// \param is_deactivated_this_step flags for pixels that are actived or de-activated for this step only
  Intensity_Sum_Functor(intensity_device_view_1d intensities,
    bool_device_view_1d is_active,
    bool_device_view_1d is_deactivated_this_step):
    intensities_(intensities),
    is_active_(is_active),
    is_deactivated_this_step_(is_deactivated_this_step){};
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index, scalar_t & mean)const{
    if(is_active_(pixel_index)&!is_deactivated_this_step_(pixel_index)){
      mean += intensities_(pixel_index);
    }
  }
};

/// Functor to compute the sum minus the mean (value used in ZNSSD criteria)
struct Intensity_Sum_Minus_Mean_Functor{
  /// pointer to the intensity values on the device
  intensity_device_view_1d intensities_;
  /// active pixel flags (persistent)
  bool_device_view_1d is_active_;
  /// active pixel flags (for current step)
  bool_device_view_1d is_deactivated_this_step_;
  /// mean value
  scalar_t mean_;
  /// constructor
  /// \param intensities the image intensity values
  /// \param mean the mean value
  /// \param is_active flags for pixels that are permanently activated or de-activated
  /// \param is_deactivated_this_step flags for pixels that are actived or de-activated for this step only
  Intensity_Sum_Minus_Mean_Functor(intensity_device_view_1d intensities,
    bool_device_view_1d is_active,
    bool_device_view_1d is_deactivated_this_step,
    scalar_t & mean):
    intensities_(intensities),
    is_active_(is_active),
    is_deactivated_this_step_(is_deactivated_this_step),
    mean_(mean){};
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index, scalar_t & sum)const{
    if(is_active_(pixel_index)&!is_deactivated_this_step_(pixel_index)){
      sum += (intensities_(pixel_index)-mean_)*(intensities_(pixel_index)-mean_);
    }
  }
};


/// znssd gamma functor
struct ZNSSD_Gamma_Functor{
  /// reference image intensities
  intensity_device_view_1d ref_intensities_;
  /// deformed image intensities
  intensity_device_view_1d def_intensities_;
  /// active pixel flags (persistent)
  bool_device_view_1d is_active_;
  /// active pixel flags (for current step)
  bool_device_view_1d is_deactivated_this_step_;
  /// mean reference intensity value
  scalar_t mean_r_;
  /// mean deformed intensity value
  scalar_t mean_d_;
  /// sum of the reference intensity values minus the mean
  scalar_t mean_sum_r_;
  /// sum of the deformed intensity values minus the mean
  scalar_t mean_sum_d_;
  /// constructor
  /// \param ref_intensities pointer to the reference intensities
  /// \param def_intensities pointer to the deformed intensities
  /// \param is_active array of flags of active pixels (persistent)
  /// \param is_deactivated_this_step array of flags for pixels deactivated this step only
  /// \param mean_r mean of the reference values
  /// \param mean_d mean of the deformed intensity values
  /// \param mean_sum_r sum of values minus the mean for reference intensities
  /// \param mean_sum_d sum of values minus the mean for reference intensities
  ZNSSD_Gamma_Functor(intensity_device_view_1d ref_intensities,
    intensity_device_view_1d def_intensities,
    bool_device_view_1d is_active,
    bool_device_view_1d is_deactivated_this_step,
    const scalar_t & mean_r,
    const scalar_t & mean_d,
    const scalar_t & mean_sum_r,
    const scalar_t & mean_sum_d):
      ref_intensities_(ref_intensities),
      def_intensities_(def_intensities),
      is_active_(is_active),
      is_deactivated_this_step_(is_deactivated_this_step),
      mean_r_(mean_r),
      mean_d_(mean_d),
      mean_sum_r_(mean_sum_r),
      mean_sum_d_(mean_sum_d){}
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index, scalar_t & gamma) const{
    if(is_active_(pixel_index)&!is_deactivated_this_step_(pixel_index)){
      scalar_t value =  (def_intensities_(pixel_index)-mean_d_)/mean_sum_d_ - (ref_intensities_(pixel_index)-mean_r_)/mean_sum_r_;
      //printf("%i value: %e %e %e %e %e %e %e\n",pixel_index,value,def_intensities_(pixel_index),mean_d_,mean_sum_d_,ref_intensities_(pixel_index),mean_r_,mean_sum_r_);
      gamma += value*value;
    }
  }
};

/// subset intensity value initialization functor
struct Subset_Init_Functor{
  /// deformation parameters
  /// displacement x
  scalar_t u_;
  /// displacement y
  scalar_t v_;
  /// rotation
  scalar_t t_;
  /// normal strain xx
  scalar_t ex_;
  /// normal strain yy
  scalar_t ey_;
  /// shear strain xy
  scalar_t g_;
  /// cos theta
  scalar_t cos_t_;
  /// sin theta
  scalar_t sin_t_;
  /// x centroid of the deformation
  int_t cx_;
  /// y centroid of the deformation
  int_t cy_;
  /// width of the image
  int_t image_w_;
  /// height of the image
  int_t image_h_;
  /// offset taken from image rcp
  int_t offset_x_;
  /// offset taken from image rcp
  int_t offset_y_;
  /// tolerance
  scalar_t tol_;
  // note all of the views below are only the device views
  // the host views are not necessary
  /// 2d view of image intensities
  intensity_device_view_2d image_intensities_;
  /// 1d view of the subset intensities
  intensity_device_view_1d subset_intensities_;
  /// view of x coordinates
  pixel_coord_device_view_1d x_;
  /// view of y coordinates
  pixel_coord_device_view_1d y_;
  /// constructor
  Subset_Init_Functor(Subset * subset,
    Teuchos::RCP<Image> image,
    Teuchos::RCP<const std::vector<scalar_t> > deformation=Teuchos::null,
    const Subset_View_Target target=REF_INTENSITIES):
      u_(0.0),
      v_(0.0),
      t_(0.0),
      ex_(0.0),
      ey_(0.0),
      g_(0.0),
      offset_x_(image->offset_x()),
      offset_y_(image->offset_y()),
      tol_(0.00001){
    if(deformation!=Teuchos::null){
      u_ = (*deformation)[DISPLACEMENT_X];
      v_ = (*deformation)[DISPLACEMENT_Y];
      t_ = (*deformation)[ROTATION_Z];
      ex_ = (*deformation)[NORMAL_STRAIN_X];
      ey_ = (*deformation)[NORMAL_STRAIN_Y];
      g_ = (*deformation)[SHEAR_STRAIN_XY];
    }
    cos_t_ = std::cos(t_);
    sin_t_ = std::sin(t_);
    // get the image intensities
    image_intensities_ = image->intensity_dual_view().d_view;
    // get a view of the subset intensities
    if(target==REF_INTENSITIES){
      subset_intensities_ = subset->ref_intensities().d_view;
    }
    else{
      subset_intensities_ = subset->def_intensities().d_view;
    }
    // get view of the coordinates:
    x_ = subset->x().d_view;
    y_ = subset->y().d_view;
    cx_ = subset->centroid_x();
    cy_ = subset->centroid_y();
    image_h_ = image->height();
    image_w_ = image->width();
  }
  /// tag
  struct Map_Bilinear_Tag {};
  /// functor to perform a mapping on the initial coordinates to get
  /// the deformed pixel intensity using bilinear interpolation
  KOKKOS_INLINE_FUNCTION
  void operator()(const Map_Bilinear_Tag &, const int_t pixel_index)const;
  /// tag
  struct Map_Keys_Tag {};
  /// functor to perform a mapping on the initial coordinates to get
  /// the deformed pixel intensity using Keys 4th-order interpolation
  KOKKOS_INLINE_FUNCTION
  void operator()(const Map_Keys_Tag &, const int_t pixel_index)const;
  /// tag
  struct No_Map_Tag {};
  /// functor to perform direct access of image pixel values without mapping
  KOKKOS_INLINE_FUNCTION
  void operator()(const No_Map_Tag &, const int_t pixel_index)const;
};

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
