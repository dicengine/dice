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

#ifndef DICE_IMAGE_H
#define DICE_IMAGE_H

#include <DICe.h>

#include <Teuchos_ParameterList.hpp>

/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe {

/// Note: the coordinates are from the top left corner (positive right for x and positive down for y)

class DICECORE_LIB_DLL_EXPORT
Image {
public:
  // TODO constructor by array
  // TODO constructor from cine
  // TODO constructor from other image
  // TODO constructor from ASCII file

  /// Constructor that takes a tiff file as an argument
  Image(const std::string & fileName,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// intensity access is always in GLOBAL image coordinates


  /// Virtual destructor
  virtual ~Image(){};
private:
  /// host pixel container
  intensity_host_view_t intensities_host_;
  /// device pixel container
  intensity_device_view_t intensities_dev_;
  /// offsets are used to convert to global image coordinates
  /// (the pixel container may be a subset of a larger image)
  size_t offset_x_;
  /// offsets are used to convert to global image coordinates
  /// (the pixel container may be a subset of a larger image)
  size_t offset_y_;
  /// pixel container width_
  size_t width_;
  /// pixel container height_
  size_t height_;
};

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
