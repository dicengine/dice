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

#ifndef DICE_CINE_H
#define DICE_CINE_H

#include <DICe_Image.h>
#include <DICe_Parser.h>

#include <cassert>
#include <iostream>

#if defined(WIN32)
  #include <cstdint>
#endif

#include <Teuchos_RCP.hpp>

namespace DICe {
/*!
 *  \namespace DICe::cine
 *  @{
 */
/// cine file format utilities
namespace cine{

/// \brief function to swap the endianness of a 16bit data type
/// \param x the variable to endian swap
inline void endian_swap(uint16_t& x){
  x = (x>>8) | (x<<8);
}

/// Fractions
typedef uint32_t FRACTIONS;
/// Pointer to fractions
typedef uint32_t *PFRACTIONS;
/// Structure to hold the trigger time
typedef struct tagTIME64
{
  /// fractions
  FRACTIONS fractions;
  /// seconds
  uint32_t seconds;
} TIME64;

/// Bit depth enumeration
enum Bit_Depth{
  BIT_DEPTH_8=0,
  BIT_DEPTH_16,
  BIT_DEPTH_10_PACKED,
  NO_SUCH_BIT_DEPTH
};

/// Structure to hold the cine file header information
struct cine_file_header{
  /// Type
  uint16_t Type;
  /// Header size
  uint16_t Headersize;
  /// Compression
  uint16_t Compression;
  /// Version
  uint16_t Version;
  /// First movie image
  int32_t FirstMovieImage;
  /// Total image count
  uint32_t TotalImageCount;
  /// First image index
  int32_t FirstImageNo;
  /// Image count
  uint32_t ImageCount;
  /// Offset to header
  uint32_t OffImageHeader;
  /// Offset to setup
  uint32_t OffSetup;
  /// Offset to image offsets
  uint32_t OffImageOffsets;
  /// trigger time
  TIME64 TriggerTime;
};

/// Structure to hold the image information
struct bitmap_info_header{
  /// bitmap size
  uint32_t biSize;
  /// bitmap width
  int32_t biWidth;
  /// bitmap height
  int32_t biHeight;
  /// bitmap planes
  uint16_t biPlanes;
  /// bitmap bit count
  uint16_t biBitCount;
  /// bitmap compression
  uint32_t biCompression;
  /// bitmap image size
  uint32_t biSizeImage;
  /// bitmap x pixels per meter
  int32_t biXPelsPerMeter;
  /// bitmap y pixels per meter
  int32_t biYPelsPerMeter;
  /// bitmap color used
  uint32_t biClrUsed;
  /// bitmap color important
  uint32_t biClrImportant;
};

/// \class DICe::cine::Cine_Header
/// \brief A class to hold header information for cine files.
/// Always assumes that the cine file has been stored in LayoutRight or row-aligned
/// in memory
class DICE_LIB_DLL_EXPORT
Cine_Header
{
public:
  /// \brief default constructor
  /// \param file_name The name of the cine file
  /// \param header The header information for the cine file
  /// \param bitmap_header The header containing all the information about the images
  Cine_Header(const std::string & file_name,
    const cine_file_header & header,
    const bitmap_info_header & bitmap_header):
  header_(header),
  bitmap_header_(bitmap_header),
  file_name_(file_name),
  bit_depth_(NO_SUCH_BIT_DEPTH){
    image_offsets_ = new int64_t[header_.ImageCount];
    // set the bit depth of the images
    int_t bit_depth = (bitmap_header_.biSizeImage * 8) / (bitmap_header_.biWidth * bitmap_header_.biHeight);
    if(bit_depth==8) bit_depth_=BIT_DEPTH_8;
    else if (bit_depth==16) bit_depth_=BIT_DEPTH_16;
    else if (bit_depth==10) bit_depth_=BIT_DEPTH_10_PACKED;
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid bit depth .cine");
    }
  };
  /// default destructor
  virtual ~Cine_Header(){
    delete[] image_offsets_;
  };
  /// cine file header
  cine_file_header header_;
  /// bitmap information
  bitmap_info_header bitmap_header_;
  /// pointer to the array with the offsets to each image
  int64_t * image_offsets_;
  /// the name of the cine file
  std::string file_name_;
  /// bit depth of the file
  Bit_Depth bit_depth_;
};
/// function that reads the header information from the cine file
Teuchos::RCP<Cine_Header>
read_cine_headers(const char *file,
  std::ostream * out_stream = NULL);

/// \class DICe::cine::Cine_Reader
/// \brief A helper class the reads in cine files from disk and provides some methods
/// such as getting a particular frame from the cine file
class DICE_LIB_DLL_EXPORT
Cine_Reader
{
public:
  /// \brief default constructor
  /// \param file_name the name of the cine file
  /// \param out_stream (optional) output stream
  /// \param filter_failed_pixels true if failed pixels should be filtered out by taking the neighbor value
  Cine_Reader(const std::string & file_name,
    std::ostream * out_stream = NULL,
    const bool filter_failed_pixels=false);
  /// default destructor
  virtual ~Cine_Reader(){};

  /// \brief create and populate an image from a cine frame
  /// allows for reading only portions of the image
  /// \param frame_index the frame number (starting with 0 as first frame, disregard triggger indexing)
  /// \param start_x upper left corner x image coordinate (inclusive)
  /// \param end_x lower right corner x image coordinate (inclusive)
  /// \param start_y upper left corner y image coordinate (inclusive)
  /// \param end_y lower right corner y image coordinate (inclusive)
  /// \param convert_to_8_bit true if the values should be converted to the range 0-255
  /// \param filter_failed_pixels true if the pixel values should be binned and filtered for outliers
  /// like dead pixels that have the max value
  /// \param params image parameters (filter image, compute gradients, etc)
  Teuchos::RCP<Image> get_frame(const int_t frame_index,
    const int_t start_x,
    const int_t start_y,
    const int_t end_x,
    const int_t end_y,
    const bool convert_to_8_bit=true,
    const bool filter_failed_pixels=false,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// \brief create and populate an image from a cine frame
  /// allows for reading only portions of the image
  /// \param frame_index the frame number (starting with 0 as first frame, disregard triggger indexing)
  /// \param convert_to_8_bit true if the values should be converted to the range 0-255
  /// \param filter_failed_pixels true if the pixel values should be binned and filtered for outliers
  /// like dead pixels that have the max value
  /// \param params image parameters (filter image, compute gradients, etc)
  Teuchos::RCP<Image> get_frame(const int_t frame_index,
    const bool convert_to_8_bit=true,
    const bool filter_failed_pixels=false,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null){
    return get_frame(frame_index,0,0,
      cine_header_->bitmap_header_.biWidth-1,cine_header_->bitmap_header_.biHeight-1,
      convert_to_8_bit,filter_failed_pixels,params);
  }

  /// \brief 8 bit frame fetch
  /// \param image the image to populate
  /// \param frame_index the frame to gather
  /// \param filter_failed_pixels get rid of outlier pixels or failed pixels
  void get_frame_8_bit(const Teuchos::RCP<Image> & image,
    const int_t frame_index,
  const bool filter_failed_pixels);

  /// \brief 10 bit frame fetch
  /// \param image the image to populate
  /// \param frame_index the frame to gather
  void get_frame_10_bit(Teuchos::RCP<Image> image,
    const int_t frame_index);

  /// \brief 10 bit frame fetch with filtering
  /// separate function to avoid if statement for each pixel (optimization)
  /// \param image the image to populate
  /// \param frame_index the frame to gather
  void get_frame_10_bit_filtered(Teuchos::RCP<Image> image,
    const int_t frame_index);

  /// \brief 16 bit frame fetch
  /// \param image the image to populate
  /// \param frame_index the frame to gather
  /// \param filter_failed_pixels get rid of outlier pixels or failed pixels
  void get_frame_16_bit(const Teuchos::RCP<Image> & image,
    const int_t frame_index,
    const bool filter_failed_pixels);

  /// \brief bins the pixels and determines a filter value
  /// The pixels are separated into 10 bins (0-9). The top bin
  /// is discarded as too bright. The average of bin 8 is used as the
  /// filter value. Every pixel above the filter value is replaced with
  /// it's neighbor value. The filtered values are then converted to 8 bit
  void initialize_cine_filter(const int_t frame_index);

  /// returns the number of images in the cine file
  int_t num_frames()const{
    return cine_header_->header_.ImageCount;
  }
  /// returns the image width
  int_t width()const{
    return cine_header_->bitmap_header_.biWidth;
  }
  /// return the image height
  int_t height()const{
    return cine_header_->bitmap_header_.biHeight;
  }
  /// return the offset to the first indexed image (can be negative)
  int_t first_image_number()const{
    return cine_header_->header_.FirstImageNo;
  }
private:
  /// pointer to the cine file header information
  Teuchos::RCP<Cine_Header> cine_header_;
  /// pointer to the output stream
  std::ostream * out_stream_;
  /// flag to prevent warnings from appearing multiple times for each frame
  bool bit_12_warning_;
  /// flag to filter out failed pixels
  bool filter_failed_pixels_;
  /// file offset
  long long int header_offset_;
  /// maximum value of intensity above which the values are filtered by taking nearest neighbor
  intensity_t filter_value_;
  /// conversion factor for converting to 8 bit depth
  intensity_t conversion_factor_;
};

}// end cine namespace
}// end DICe namespace

#endif
