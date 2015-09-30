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

#ifndef DICE_SUBSET_H
#define DICE_SUBSET_H

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Kokkos.h>

#include <Teuchos_ArrayRCP.hpp>

/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe {

/// \class DICe::Subset
/// Subsets are used to store temporary collections of pixels for comparison between the
/// reference and deformed images. The data that is stored by a subset is a list of x and y
/// corrdinates of each pixel (this allows for arbitrary shape) and containers for pixel
/// intensity values.

class DICE_LIB_DLL_EXPORT
Subset {
public:

  /// constructor that takes arrays of the pixel locations as input
  /// \param cx centroid x pixel location
  /// \param cy centroid y pixel location
  /// \param x array of x coordinates
  /// \param y array of y coordinates
  Subset(int_t cx,
    int_t cy,
    Teuchos::ArrayRCP<int_t> x,
    Teuchos::ArrayRCP<int_t> y);

  /// constructor that takes a centroid and dims as arguments
  /// \param cx centroid x pixel location
  /// \param cy centroid y pixel location
  /// \param width width of the centroid (should be an odd number, otherwise the next larger odd number is used)
  /// \param height height of the centroid (should be an odd number, otherwise the next larger odd numer is used)
  Subset(const int_t cx,
    const int_t cy,
    const int_t width,
    const int_t height);

  /// virtual destructor
  virtual ~Subset(){};

  /// returns the number of pixels in the subset
  int_t num_pixels()const{
    return num_pixels_;
  }

  /// returns the centroid x pixel location
  int_t centroid_x()const{
    return cx_;
  }

  /// returns the centroid y pixel location
  int_t centroid_y()const{
    return cy_;
  }

  /// x coordinate view accessor
  pixel_coord_dual_view_1d x()const{
    return x_;
  }

  /// y coordinate view accessor
   pixel_coord_dual_view_1d y()const{
    return y_;
  }

  /// x coordinate accessor
  /// \param pixel_index the pixel id
  /// note there is no bounds checking on the index
  const int_t& x(const int_t pixel_index)const{
    return x_.h_view(pixel_index);
  }

  /// y coordinate accessor
  /// \param pixel_index the pixel id
  /// note there is no bounds checking on the index
  const int_t& y(const int_t pixel_index)const{
    return y_.h_view(pixel_index);
  }

  /// ref intensities accessor
  /// \param pixel_index the pixel id
  const intensity_t& ref_intensities(const int_t pixel_index)const{
    return ref_intensities_.h_view(pixel_index);
  }

  /// ref intensities accessor
  /// \param pixel_index the pixel id
  const intensity_t& def_intensities(const int_t pixel_index)const{
    return def_intensities_.h_view(pixel_index);
  }

  /// ref intensities device view accessor
  intensity_dual_view_1d ref_intensities()const{
    return ref_intensities_;
  }

  /// ref intensities device view accessor
  intensity_dual_view_1d def_intensities()const{
    return def_intensities_;
  }

  /// initialization method:
  /// \param image the image to get the intensity values from
  /// \param map the deformation map (optional)
  /// \param target the initialization mode (put the values in the ref or def intensities)
  void initialize(Teuchos::RCP<Image> image,
    const Subset_View_Target target=REF_INTENSITIES,
    Teuchos::RCP<Def_Map> map=Teuchos::null,
    const Interpolation_Method interp=KEYS_FOURTH_ORDER);

  /// write the subset intensity values to a tif file
  /// \param file_name the name of the tif file to write
  /// \param use_def_intensities use the deformed intensities rather than the reference
  void write_tiff(const std::string & file_name,
    const bool use_def_intensities=false);

  /// draw the subset over an image
  /// \param file_name the name of the tif file to write
  /// \param image pointer to the image to use as the background
  /// \param deform_subset deform the subset on the image (rather than its original shape)
  void write_subset_on_image(const std::string & file_name,
    Teuchos::RCP<Image> image,
    Teuchos::RCP<Def_Map> map=Teuchos::null);

  /// returns the mean intensity value
  /// \param target either the reference or deformed intensity values
  scalar_t mean(const Subset_View_Target target);

  /// returns the mean intensity value
  /// \param target either the reference or deformed intensity values
  /// \param sum [output] returns the reduction value of the intensities minus the mean
  scalar_t mean(const Subset_View_Target target,
    scalar_t & sum);

  /// returns the ZNSSD gamma correlation value between the reference and deformed subsets
  scalar_t gamma();

private:
  /// number of pixels in the subset
  int_t num_pixels_;
  /// pixel container
  intensity_dual_view_1d ref_intensities_;
  /// pixel container
  intensity_dual_view_1d def_intensities_;
  /// centroid location x
  int_t cx_; // assumed to be the middle of the pixel
  /// centroid location y
  int_t cy_; // assumed to be the middle of the pixel
  /// initial x position of the pixels in the reference image
  pixel_coord_dual_view_1d x_;
  /// initial x position of the pixels in the reference image
  pixel_coord_dual_view_1d y_;

};

/// mean value functor
struct Intensity_Sum_Functor{
  /// pointer to the intensity values on the device
  intensity_device_view_1d intensities_;
  /// constructor
  /// \param intensities the image intensity values
  Intensity_Sum_Functor(intensity_device_view_1d intensities):
    intensities_(intensities){};
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index, scalar_t & mean)const{
    mean += intensities_(pixel_index);
  }
};

struct Intensity_Sum_Minus_Mean_Functor{
  /// pointer to the intensity values on the device
  intensity_device_view_1d intensities_;
  /// mean value
  scalar_t mean_;
  /// constructor
  /// \param intensities the image intensity values
  Intensity_Sum_Minus_Mean_Functor(intensity_device_view_1d intensities,
    scalar_t & mean):
    intensities_(intensities),
    mean_(mean){};
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index, scalar_t & sum)const{
    sum += (intensities_(pixel_index)-mean_)*(intensities_(pixel_index)-mean_);
  }
};


/// znssd gamma functor
struct ZNSSD_Gamma_Functor{
  /// reference image intensities
  intensity_device_view_1d ref_intensities_;
  /// deformed image intensities
  intensity_device_view_1d def_intensities_;
  /// mean reference intensity value
  scalar_t mean_r_;
  /// mean deformed intensity value
  scalar_t mean_d_;
  /// sum of the reference intensity values minus the mean
  scalar_t mean_sum_r_;
  /// sum of the deformed intensity values minus the mean
  scalar_t mean_sum_d_;
  /// constructor
  /// \param
  ZNSSD_Gamma_Functor(intensity_device_view_1d ref_intensities,
    intensity_device_view_1d def_intensities,
    const scalar_t & mean_r,
    const scalar_t & mean_d,
    const scalar_t & mean_sum_r,
    const scalar_t & mean_sum_d):
      ref_intensities_(ref_intensities),
      def_intensities_(def_intensities),
      mean_r_(mean_r),
      mean_d_(mean_d),
      mean_sum_r_(mean_sum_r),
      mean_sum_d_(mean_sum_d){}
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index, scalar_t & gamma) const{
    scalar_t value =  (def_intensities_(pixel_index)-mean_d_)/mean_sum_d_ - (ref_intensities_(pixel_index)-mean_r_)/mean_sum_r_;
    gamma += value*value;
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
    Teuchos::RCP<Def_Map> map=Teuchos::null,
    const Subset_View_Target target=REF_INTENSITIES):
      u_(0.0),
      v_(0.0),
      t_(0.0),
      ex_(0.0),
      ey_(0.0),
      g_(0.0),
      tol_(0.00001){
    if(map!=Teuchos::null){
      u_ = map->u_;
      v_ = map->v_;
      t_ = map->t_;
      ex_ = map->ex_;
      ey_ = map->ey_;
      g_ = map->g_;
    }
    cos_t_ = std::cos(t_);
    sin_t_ = std::sin(t_);
    // get the image intensities
    image_intensities_ = image->intensities().d_view;
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
