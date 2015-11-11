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

#ifndef DICE_RAWI_H
#define DICE_RAWI_H

#include <DICe.h>

#include <string>

namespace DICe{
namespace utils{

/// Raw Intensity Format (.rawi), allows saving decimal numbers
/// as well as negative numbers, neither of which are enabled for
/// standard image file formats

/// read the image dimensions
/// \param file_name the .rawi file name
/// \param width [out] returned as the width of the image
/// \param height [out] returned as the height of the image
DICE_LIB_DLL_EXPORT
void read_rawi_image_dimensions(const char * file_name,
  int_t & width,
  int_t & height);

/// Read an image into the host memory
/// \param file_name the name of the .rawi file
/// \param intensities [out] populated with the pixel intensity values
/// \param is_layout_right [optional] memory layout is LayoutRight (row-major)
DICE_LIB_DLL_EXPORT
void read_rawi_image(const char * file_name,
  intensity_t * intensities,
  const bool is_layout_right = true);

// TODO read portions of a .rawi file

// TODO write a function that reads into the device memory directly

// TODO save the image gradients as well as other info?

/// write an image to disk
/// \param file_name the name of the .rawi file
/// \param width the width of the image to write
/// \param height the height of the image
/// \param intensities assumed to be an array of size width x height
/// \param is_layout_right [optional] memory layout is LayoutRight (row-major)
DICE_LIB_DLL_EXPORT
void write_rawi_image(const char * file_name,
  const int_t width,
  const int_t height,
  intensity_t * intensities,
  const bool is_layout_right = true);

} // end namespace utils
} // end namespace DICe

#endif
