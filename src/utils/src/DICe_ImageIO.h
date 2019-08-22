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

#ifndef DICE_IMAGEIO_H
#define DICE_IMAGEIO_H

#include <DICe.h>
#include <DICe_Cine.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>
#include <map>

namespace DICe{
/*!
 *  \namespace DICe::utils
 *  @{
 */
/// utilities that wont build with NVCC so they are separated out into another namespace
namespace utils{

/// returns the name of a netcdf file given a decorated cine string
/// \param decorated_netcdf_file_string the decorated string that contains the name
DICE_LIB_DLL_EXPORT
std::string netcdf_file_name(const char * decorated_netcdf_file);

/// returns the frame index decyphered from the netcdf file descriptor passed in
/// \param decorated_netcdf_file the descriptor that has the netcdf name and index concatendated
DICE_LIB_DLL_EXPORT
int_t netcdf_index(const char * decorated_netcdf_file);

/// returns the name of a file given a decorated file name string
/// \param decorated_cine_file the decorated string that contains the name
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
/// \param intensities [out] populated with the image intensities
/// \param params apply special filters or select sub portions of the image
DICE_LIB_DLL_EXPORT
void read_image(const char * file_name,
  intensity_t * intensities,
  const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);


/// Spread the image intensity histogram if it's grouped in a cluster
/// \param width
/// \param height
/// \param intensities
DICE_LIB_DLL_EXPORT
void spread_histogram(const int_t width,
  const int_t height,
  intensity_t * intensities);

/// Round the image intensity values to the nearest integer value
/// \param width
/// \param height
/// \param intensities
DICE_LIB_DLL_EXPORT
void round_intensities(const int_t width,
  const int_t height,
  intensity_t * intensities);

/// Set the intensity value for outlier pixels (the ones with the highest intensity value) to the second highest value
/// This is helpful in removing failed cine pixels
/// \param width
/// \param height
/// \param intensities
DICE_LIB_DLL_EXPORT
void remove_outliers(const int_t width,
  const int_t height,
  intensity_t * intensities);

/// Round the image intensity values to the nearest integer value
/// \param width
/// \param height
/// \param intensities
DICE_LIB_DLL_EXPORT
void floor_intensities(const int_t width,
  const int_t height,
  intensity_t * intensities);

/// write an image to disk (always output as an 8-bit grayscale image)
/// for more precise output, for example to read the intensity values in
/// later with the same precision, use the .rawi format (see DICe::rawi)
/// \param file_name the name of the file
/// \param width the width of the image to write
/// \param height the height of the image
/// \param intensities assumed to be an array of size width x height
/// \param is_layout_right [optional] memory layout is LayoutRight (row-major)
DICE_LIB_DLL_EXPORT
void write_image(const char * file_name,
  const int_t width,
  const int_t height,
  intensity_t * intensities,
  const bool is_layout_right = true);


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

/// Read an image into the host memory returning an opencv Mat object
/// \param file_name the name of the file
DICE_LIB_DLL_EXPORT
cv::Mat read_image(const char * file_name);

// singleton class to keep track of image readers from high speed video or netcdf files:
/// \class Image_Reader_Cache
/// used for file reads and getting image dimensions without having to reload the header every time
DICE_LIB_DLL_EXPORT
class Image_Reader_Cache{
public:
  /// return an instance of the singleton
  static Image_Reader_Cache &instance(){
    static Image_Reader_Cache instance_;
    return instance_;
  }

  /// add a cine reader to the map
  /// \param id the string name of the reader in case multiple headers are loaded (for example in stereo)
  /// if the reader doesn't exist, it gets created
  Teuchos::RCP<DICe::cine::Cine_Reader> cine_reader(const std::string & id);
  /// clear the map
  void clear(){cine_reader_map_.clear();}
private:
  /// constructor
  Image_Reader_Cache(){};
  /// copy constructor
  Image_Reader_Cache(Image_Reader_Cache const&);
  /// asignment operator
  void operator=(Image_Reader_Cache const &);
  /// map of cine readers
  std::map<std::string,Teuchos::RCP<DICe::cine::Cine_Reader> > cine_reader_map_;
};


} // end namespace utils
} // end namespace DICe

#endif
