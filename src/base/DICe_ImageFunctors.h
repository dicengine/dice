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
#ifndef DICE_IMAGEFUNCTORS_H
#define DICE_IMAGEFUNCTORS_H

#include <DICe.h>
#include <DICe_Kokkos.h>
#include <Teuchos_ParameterList.hpp>

namespace DICe{

/// \brief image mask initialization functor
/// note, the number of pixels is the size of the x and y arrays, not the image
struct Mask_Init_Functor{
  /// pointer to the mask dual view
  scalar_device_view_2d mask_;
  /// pointer to the array of x values to enable (mask = 1.0)
  pixel_coord_device_view_1d x_;
  /// pointer to the array of y values to enable (mask = 1.0)
  pixel_coord_device_view_1d y_;
  /// all other pixels will set to mask = 0.0
  /// constructor
  /// \param mask pointer to the mask array on the device
  /// \param x pointer to the array of x coordinates on the device
  /// \param y pointer to the array of y coordinates on the device
  Mask_Init_Functor(scalar_device_view_2d mask,
    pixel_coord_device_view_1d x,
    pixel_coord_device_view_1d y):
    mask_(mask),
    x_(x),
    y_(y){};
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index)const{
    mask_(y_(pixel_index),x_(pixel_index)) = 1.0;
  }
};

/// \brief image mask initialization functor
/// note, the number of pixels is the size the image
struct Mask_Smoothing_Functor{
  /// pointer to the mask dual view
  scalar_dual_view_2d mask_;
  /// pointer to a temporary copy of the mask field prior to smoothing
  scalar_device_view_2d mask_tmp_;
  /// width of the image
  int_t width_;
  /// height of the image
  int_t height_;
  /// gauss filter coefficients
  scalar_t gauss_filter_coeffs_[5][5];
  /// constructor
  /// \param mask pointer to the mask array on the device
  /// \param width the width of the image
  /// \param height the height of the image
  Mask_Smoothing_Functor(scalar_dual_view_2d mask,
    const int_t width,
    const int_t height);
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index)const;
};

/// \brief image mask apply functor
struct Mask_Apply_Functor{
  /// pointer to the intensity values
  intensity_device_view_2d intensities_;
  /// pointer to the mask dual view
  scalar_device_view_2d mask_;
  /// image width
  int_t width_;
  /// constructor
  /// \param intensities pointer to the intensity array on the device
  /// \param mask pointer to the mask array on the device
  /// \param width the image width
  Mask_Apply_Functor(intensity_device_view_2d intensities,
    scalar_device_view_2d mask,
    int_t width):
    intensities_(intensities),
    mask_(mask),
    width_(width){};
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index)const{
    const int_t y = pixel_index / width_;
    const int_t x = pixel_index - y*width_;
    intensities_(y,x) = mask_(y,x)*intensities_(y,x);
  }
};

/// \brief image transformation functor
/// given parameters theta, u, and v, transform the given image
/// uses the keys interpolant (TODO add other interpolants)
struct Transform_Functor{
  /// pointer to the intensity dual view
  intensity_device_view_2d intensities_from_;
  /// pointer to the intensity dual view
  intensity_device_view_2d intensities_to_;
  /// centroid in x (note this is the transformed centroid)
  scalar_t cx_;
  /// centroid in y (node this is the transformed centroid)
  scalar_t cy_;
  /// width of the image
  int_t width_;
  /// height of the image
  int_t height_;
  /// displacement x;
  scalar_t u_;
  /// displacement y;
  scalar_t v_;
  /// rotation angle
  scalar_t t_;
  /// normal strain x
  scalar_t ex_;
  /// normal strain y
  scalar_t ey_;
  /// shear strain xy
  scalar_t g_;
  /// cosine of theta
  scalar_t cost_;
  /// sin of theta
  scalar_t sint_;
  /// tolerance
  scalar_t tol_;
  /// constructor
  /// \param intensities_from pointer to the intensity array to be transformed
  /// \param intensities_to pointer to the result intensity array
  /// \param width the width of the image
  /// \param height the height of the image
  /// \param cx centroid in x
  /// \param cy centroid in y
  /// \param def the deformation map parameters
  Transform_Functor(intensity_device_view_2d intensities_from,
    intensity_device_view_2d intensities_to,
    const int_t width,
    const int_t height,
    const int_t cx,
    const int_t cy,
    Teuchos::RCP<const std::vector<scalar_t> > def):
    intensities_from_(intensities_from),
    intensities_to_(intensities_to),
    cx_(cx + (*def)[DISPLACEMENT_X]),
    cy_(cy + (*def)[DISPLACEMENT_Y]),
    width_(width),
    height_(height),
    u_((*def)[DISPLACEMENT_X]),
    v_((*def)[DISPLACEMENT_Y]),
    t_((*def)[ROTATION_Z]),
    ex_((*def)[NORMAL_STRAIN_X]),
    ey_((*def)[NORMAL_STRAIN_Y]),
    g_((*def)[SHEAR_STRAIN_XY]),
    tol_(0.00001){
    cost_ = std::cos(t_);
    sint_ = std::sin(t_);
  };
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index)const;
};

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
