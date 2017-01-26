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

#ifndef DICE_IMAGEIO_H
#define DICE_IMAGEIO_H

#include <DICe.h>
#include <DICe_Cine.h>

#include <Teuchos_RCP.hpp>

#include <string>
#include <map>

namespace DICe{
/*!
 *  \namespace DICe::utils
 *  @{
 */
/// utilities that wont build with NVCC so they are separated out into another namespace
namespace utils{

/// returns the name of a cine file given a decorated cine string
/// \param cine_string the decorated string that contains the name
DICE_LIB_DLL_EXPORT
std::string cine_file_name(const char * decorated_cine_file);

/// returns the start index decyphered from the cine file descriptor passed in
/// \param decorated_cine_file the descriptor that has the cine name and index concatendated
/// \param start_index [out] the start index requested
/// \param end_index [out] returned as -1 unless the descriptor has an "avg" tag appended
/// \param is_avg true if more than one frame gets averaged together
/// The convention is as follows cine_file_0.cine would lead to a file name of cine_file.cine and an index of 0
/// Another example would be cine_file_avg45to51.cine that would lead to a file name of cine_file.cine and
/// averaging of frames 45 through 51 inclusive.
/// negative numbers are okay e.g. cine_file_avg-12to-10.cine or cine_file_-500.cine
DICE_LIB_DLL_EXPORT
void cine_index(const char * decorated_cine_file,
  int_t & start_index,
  int_t & end_index,
  bool & is_avg);

/// returns the type of file based on the name
/// \param file_name the name of the file
DICE_LIB_DLL_EXPORT
Image_File_Type image_file_type(const char * file_name);

/// read the image dimensions
/// \param file_name the name of the file
/// \param width [out] returned as the width of the image
/// \param height [out] returned as the height of the image
DICE_LIB_DLL_EXPORT
void read_image_dimensions(const char * file_name,
  int_t & width,
  int_t & height);

/// Read an image into the host memory
/// \param file_name the name of the file
/// \param intensities [out] populated with the pixel intensity values
/// \param is_layout_right true if the arrays are row-major
DICE_LIB_DLL_EXPORT
void read_image(const char * file_name,
  intensity_t * intensities,
  const bool is_layout_right = true);

/// Read an image into the host memory
/// \param file_name the name of the file
/// \param offset_x the upper left corner x-coordinate in global image coordinates
/// \param offset_y the upper left corner y-coordinate in global image coordinates
/// \param width width of the portion of the image to read (must be smaller than the global image width)
/// \param height height of the portion of the image to read (must be smaller than the global image height)
/// \param intensities [out] populated with the image intensities
/// \param is_layout_right [optional] memory layout is LayoutRight (row-major)
DICE_LIB_DLL_EXPORT
void read_image(const char * file_name,
  int_t offset_x,
  int_t offset_y,
  int_t width,
  int_t height,
  intensity_t * intensities,
  const bool is_layout_right = true);

// TODO write a function that reads into the device memory directly

/// write an image to disk (always output as an 8-bit grayscale image)
/// for more precise output, for example to read the intensity values in
/// later with the same precision, use the .rawi format (see DICe::rawi)
/// \param file_name the name of the file
/// \param width the width of the image to write
/// \param height the height of the image
/// \param intensities assumed to be an array of size width x height
/// \param is_layout_right [optional] memory layout is LayoutRight (row-major)
/// \param scale_to_8_bit scale the values to 8 bit range
DICE_LIB_DLL_EXPORT
void write_image(const char * file_name,
  const int_t width,
  const int_t height,
  intensity_t * intensities,
  const bool is_layout_right = true,
  const bool scale_to_8_bit = true);


/// write an image to disk with two base images overlayed with transparency
/// \param file_name the name of the file
/// \param width the width of the image to write
/// \param height the height of the image
/// \param bottom_intensities assumed to be an array of size width x height
/// \param top_intensities assumed to be an array of size width x height
DICE_LIB_DLL_EXPORT
void write_color_overlap_image(const char * file_name,
  const int_t width,
  const int_t height,
  intensity_t * bottom_intensities,
  intensity_t * top_intensities);


// singleton class to keep track of image readers from high speed video or netcdf files:
/// \class Image_Reader_Cache
/// used for file reads and getting image dimensions without having to reload the header every time
class Image_Reader_Cache{
public:
  /// return an instance of the singleton
  static Image_Reader_Cache &instance(){
    static Image_Reader_Cache instance_;
    return instance_;
  }

  /// filters failed pixels from the images
  void set_filter_failed_pixels(const bool flag){
    filter_failed_pixels_ = flag;
  }

  /// filters failed pixels from the images
  bool filter_failed_pixels()const{
    return filter_failed_pixels_;
  }

  /// add a cine reader to the map
  /// \param id the string name of the reader in case multiple headers are loaded (for example in stereo)
  /// if the reader doesn't exist, it gets created
  Teuchos::RCP<DICe::cine::Cine_Reader> cine_reader(const std::string & id);
private:
  /// constructor
  Image_Reader_Cache():filter_failed_pixels_(false){};
  /// copy constructor
  Image_Reader_Cache(Image_Reader_Cache const&);
  /// asignment operator
  void operator=(Image_Reader_Cache const &);
  /// map of cine readers
  std::map<std::string,Teuchos::RCP<DICe::cine::Cine_Reader> > cine_reader_map_;
  /// filter failed pixels from images as they are loaded
  bool filter_failed_pixels_;
};


} // end namespace utils
} // end namespace DICe

#endif
