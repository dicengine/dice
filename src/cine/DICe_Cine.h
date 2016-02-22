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
  file_name_(file_name){
    image_offsets_ = new int64_t[header_.ImageCount];
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
  Cine_Reader(const std::string & file_name,
    std::ostream * out_stream = NULL);
  /// default destructor
  virtual ~Cine_Reader(){};
  /// \brief get an image from the cine file
  /// \param frame_index the index of the frame to get
  /// \param params (optional) image parameter, such as compute_gradients, etc.
  Teuchos::RCP<Image> get_frame(const int_t frame_index,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null){
    std::vector<Teuchos::RCP<Image> > frame_vec = get_frames(frame_index,frame_index,params);
    return frame_vec[0];
  }
  /// \brief get a set of images from a cine file using buffers and threading
  /// \param frame_index_start the start index of the frames to get
  /// \param frame_index_end the end index of the frames to get
  /// \param params (optional) image parameter, such as compute_gradients, etc.
  std::vector<Teuchos::RCP<Image> > get_frames(const int_t frame_index_start, const int_t frame_index_end,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);
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
};

}// end cine namespace
}// end DICe namespace

#endif
