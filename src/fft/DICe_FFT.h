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
DICE_LIB_DLL_EXPORT
void
image_fft(Teuchos::RCP<Image> image,
  Teuchos::ArrayRCP<scalar_t> & real,
  Teuchos::ArrayRCP<scalar_t> & complex,
  int_t inverse = 0);

/// Phase correlate two images
/// \param image_a the left image to correlate
/// \param image_b the right image to correlate
/// \u_x [out] displacement x
/// \u_y [out] displacement y
DICE_LIB_DLL_EXPORT
void
phase_correlate_x_y(Teuchos::RCP<Image> image_a,
  Teuchos::RCP<Image> image_b,
  scalar_t & u_x,
  scalar_t & u_y);

/// 2 dimensional FFT of an array
/// \param w width
/// \param h height
/// \real the real array
/// \complex the imaginary array
/// \inverse 1 if the FFT should be to the time domain
DICE_LIB_DLL_EXPORT
void
array_2d_fft_in_place(int_t w,
  int_t h,
  Teuchos::ArrayRCP<scalar_t> & real,
  Teuchos::ArrayRCP<scalar_t> & complex,
  int_t inverse = 0);

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
