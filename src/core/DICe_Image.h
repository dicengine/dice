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
#include <DICe_KokkosTypes.h>

#include <Teuchos_ParameterList.hpp>

/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe {

/// \class DICe::Image
/// A container class to hold the pixel intensity information and provide some basic methods
/// Note: the coordinates are from the top left corner (positive right for x and positive down for y)
/// intensity access is always in local coordinates, for example if only a portion of an image is read
/// into the intensity values, accessing the first value in the array is via the indicies (0,0) even if
/// the first pixel is not in the upper left corner of the global image from which the poriton was taken

class DICECORE_LIB_DLL_EXPORT
Image {
public:
  // TODO constructor by array
  // TODO constructor from cine
  // TODO constructor from other image
  // TODO constructor from ASCII file

  //
  // tiff image constructors
  //

  /// constructor that reads in a whole tiff file
  /// \param file_name the name of the tiff file
  /// \param params image parameters
  Image(const std::string & file_name,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// constructor that stores only a portion of a tiff file given by the offset and dims
  /// \param file_name the name of the tiff file
  /// \param offset_x upper left corner x-coordinate
  /// \param offset_y upper left corner y-coorindate
  /// \param width x-dim of the image (offset_x + width must be < the global image width)
  /// \param height y-dim of the image (offset_y + height must be < the global image height)
  /// \param params image parameters
  Image(const std::string & file_name,
    const size_t offset_x,
    const size_t offset_y,
    const size_t width,
    const size_t height,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  //
  // pre allocated array images
  //

  /// constrtuctor that takes an array as input
  /// \param intensities pre-allocated array of intensity values
  /// \param width the width of the image
  /// \param height the height of the image
  /// \param params image parameters
  Image(scalar_t * intensities,
    const size_t width,
    const size_t height,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);


  /// write the image to tiff file
  /// \param file_name the name of the file to write to
  void write(const std::string & file_name);

  /// returns the width of the image
  size_t width()const{return width_;}

  /// return the height of the image
  size_t height()const{return height_;}

  /// returns the offset x coordinate
  size_t offset_x()const{return offset_x_;}

  /// returns the offset y coordinate
  size_t offset_y()const{return offset_y_;}

  /// intensity accessors:
  /// note the internal arrays are stored as (row,column) so the indices have to be switched from coordinates x,y to y,x
  /// y is row, x is column
  /// \param x image coordinate x
  /// \param y image coordinate y
  const intensity_t& operator()(const size_t x, const size_t y) const {return intensities_host_(y,x);}

  /// gradient accessors:
  /// note the internal arrays are stored as (row,column) so the indices have to be switched from coordinates x,y to y,x
  /// y is row, x is column
  /// \param x image coordinate x
  /// \param y image coordinate y
  const scalar_t& grad_x(const size_t x, const size_t y) const {return grad_x_host_(y,x);}

  /// gradient accessor for y
  /// \param x image coordinate x
  /// \param y image coordinate y
  const scalar_t& grad_y(const size_t x, const size_t y) const {return grad_y_host_(y,x);}

  /// compute the image gradients
  void compute_gradients();

  /// returns true if the gradients have been computed
  bool has_gradients()const{return has_gradients_;}

  /// returns the number of pixels in the image
  size_t num_pixels()const{return width_*height_;}

  /// virtual destructor
  virtual ~Image(){};
private:
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
  /// host pixel container
  intensity_host_view_t intensities_host_;
  /// device pixel container
  intensity_device_view_t intensities_dev_;
  /// host image gradient x container
  scalar_host_view_2d_t grad_x_host_;
  /// device image gradient x container
  scalar_device_view_2d_t grad_x_dev_;
  /// host image gradient y container
  scalar_host_view_2d_t grad_y_host_;
  /// device image gradient y container
  scalar_device_view_2d_t grad_y_dev_;
  /// flag that the gradients have been computed
  bool has_gradients_;
};

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
