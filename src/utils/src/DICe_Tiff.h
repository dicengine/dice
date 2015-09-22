// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICE)
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
// Questions? Contact:  Dan Turner (dzturne@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef DICE_TIFF_H
#define DICE_TIFF_H

#include <DICe.h>

#include <string>

namespace DICe{

/// read the image dimensions
/// \param file_name the tiff file name
/// \param width [out] returned as the width of the image
/// \param height [out] returned as the height of the image
DICE_LIB_DLL_EXPORT
void read_tiff_image_dimensions(const std::string & file_name,
  size_t & width,
  size_t & height);

/// Read an image into the host memory
/// \param file_name the name of the tiff file
/// \param intensities [out] populated with the pixel intensity values
DICE_LIB_DLL_EXPORT
void read_tiff_image(const std::string & file_name,
  intensity_t * intensities,
  const bool is_layout_right = true);

/// Read an image into the host memory
/// \param file_name the name of the tiff file
/// \param offset_x the upper left corner x-coordinate in global image coordinates
/// \param offset_y the upper left corner y-coordinate in global image coordinates
/// \param width width of the portion of the image to read (must be smaller than the global image width)
/// \param height height of the portion of the image to read (must be smaller than the global image height)
/// \param intensities [out] populated with the image intensities
DICE_LIB_DLL_EXPORT
void read_tiff_image(const std::string & file_name,
  size_t offset_x,
  size_t offset_y,
  size_t width,
  size_t height,
  intensity_t * intensities,
  const bool is_layout_right = true);

// TODO write a function that reads into the device memory directly

/// write an image to disk
/// \param file_name the name of the tiff file
/// \param width the width of the image to write
/// \param height the height of the image
/// \param intensities assumed to be an array of size width x height
DICE_LIB_DLL_EXPORT
void write_tiff_image(const std::string & file_name,
  const size_t width,
  const size_t height,
  intensity_t * intensities,
  const bool is_layout_right = true);

}

#endif
