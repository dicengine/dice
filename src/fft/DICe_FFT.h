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

#ifndef DICE_FFT_H
#define DICE_FFT_H

#include <DICe_Image.h>
#include <kiss_fft.h>

#include <Teuchos_ArrayRCP.hpp>

#include <cassert>

namespace DICe {

/// complex number divide
/// \param lhs left hand side
/// \param rhs right hand side
/// \param size number of elements in the array
DICE_LIB_DLL_EXPORT
void
complex_divide(kiss_fft_cpx * lhs,
  kiss_fft_cpx * rhs,
  const int_t size);

/// FFT of an image
/// \param image the image to take the FFT of
/// \param real [out] the real part of the FFT
/// \param complex [out] the imaginary part of the FFT
/// \param inverse 1 if this should be an inverse FFT (back to time domain)
/// \param hamming_filter true if a hamming filter should be applied to the image
template <typename S>
DICE_LIB_DLL_EXPORT
void
image_fft(Teuchos::RCP<Image_<S>> image,
  Teuchos::ArrayRCP<scalar_t> & real,
  Teuchos::ArrayRCP<scalar_t> & complex,
  const int_t inverse = 0,
  const bool hamming_filter=true);

/// compute the image fft and return an image with
/// intensity values as the magnitude of the FFT values
/// note: the fft values are scaled and the log is taken,
/// so taking the inverse fft of an fft image is not the original image
/// \param image the image to take the FFT of
/// \param hamming_filter true if a hamming filter should be applied to the image
/// \param apply_log true if the log of the values should be taken
/// \param scale_factor used if the log is applied i = scale_factor*log(i+1)
/// \param shift true if the quandrants of the fft image should be swapped
/// \param high_pass_filter true if the values should be filtered

template <typename S>
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Image_<S>>
image_fft(Teuchos::RCP<Image_<S>> image,
  const bool hamming_filter=true,
  const bool apply_log=true,
  const scalar_t scale_factor=100.0,
  bool shift=true,
  const bool high_pass_filter=false);

/// Phase correlate two images,
/// If the images have been polar transformed, then set the
/// convert_to_r_theta flag to true, in that case theta is positive counter-clockwise
/// the return value is the peak magnitude value from the fft cross-correlation
/// \param image_a the left image to correlate
/// \param image_b the right image to correlate
/// \param u_x [out] displacement x
/// \param u_y [out] displacement y
/// \param convert_to_r_theta true if the images are polar transforms and
/// the correlation is for radius and angle of rotation
DICE_LIB_DLL_EXPORT
scalar_t
phase_correlate_x_y(Teuchos::RCP<Image> image_a,
  Teuchos::RCP<Image> image_b,
  scalar_t & u_x,
  scalar_t & u_y,
  const bool convert_to_r_theta=false);

/// Phase correlate a single row from two images
/// \param image_a the first image
/// \param image_b the second image
/// \param row_id the id of the row to correlate
/// \param u the output displacement
/// \param convert_to_theta conver the displacement to degrees
DICE_LIB_DLL_EXPORT
void
phase_correlate_row(Teuchos::RCP<Image> image_a,
  Teuchos::RCP<Image> image_b,
  const int_t row_id,
  scalar_t & u,
  const bool convert_to_theta=false);


/// 2D polar transformation of an image
/// returns a new image that is the polar transform of the input image
/// \param image the input image
/// \param high_pass_filter true if the radius of the polar transform
/// should be limited to w/4 since the rest of the image would be zero from filtering
/// so the output image will be twice as tall as the input
template <typename S>
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Image_<S>>
polar_transform(Teuchos::RCP<Image_<S>> image,
  const bool high_pass_filter = false);

/// 2 dimensional FFT of an array
/// \param w width
/// \param h height
/// \param real the real array
/// \param complex the imaginary array
/// \param inverse 1 if the FFT should be to the time domain
DICE_LIB_DLL_EXPORT
void
array_2d_fft_in_place(const int_t w,
  const int_t h,
  Teuchos::ArrayRCP<scalar_t> & real,
  Teuchos::ArrayRCP<scalar_t> & complex,
  const int_t inverse = 0);

/// multiply two complex numbers
/// \param result_r [out] the real result
/// \param result_i [out] the imaginary result
/// \param a_r the left hand side real part
/// \param a_i the left hand side imaginary part
/// \param b_r the right hand side real part
/// \param b_i the right hand side imaginary part
DICE_LIB_DLL_EXPORT
void
complex_multiply(scalar_t & result_r,
  scalar_t & result_i,
  const scalar_t & a_r,
  const scalar_t & a_i,
  const scalar_t & b_r,
  const scalar_t & b_i);

/// absolute value of a complex number
/// \param result [out] the result
/// \param a_r the real part of the complex number
/// \param a_i the imaginary part of the complex number
DICE_LIB_DLL_EXPORT
void
complex_abs(scalar_t & result,
  const scalar_t & a_r,
  const scalar_t & a_i);

/// divide two complex numbers
/// \param result_r [out] the real result
/// \param result_i [out] the imaginary result
/// \param a_r the left hand side real part
/// \param a_i the left hand side imaginary part
/// \param b_r the right hand side real part
/// \param b_i the right hand side imaginary part
DICE_LIB_DLL_EXPORT
void
complex_divide(scalar_t & result_r,
  scalar_t & result_i, const
  scalar_t & a_r,
  const scalar_t & a_i,
  const scalar_t & b_r,
  const scalar_t & b_i);

}// End DICe Namespace

#endif
