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

#include <cassert>
#include <iostream>
#include <fstream>
#if defined(WIN32)
  #include <cstdint>
#endif

#include <DICe_Rawi.h>

namespace DICe{
namespace utils{

DICE_LIB_DLL_EXPORT
void read_rawi_image_dimensions(const char * file_name,
  int_t & width,
  int_t & height){

  std::ifstream rawi_file (file_name, std::ifstream::in | std::ifstream::binary);
  if (rawi_file.fail()){
    std::cerr << "ERROR: Can't open the file: " + (std::string)file_name << std::endl;
    exit(1);
  }
  // read the file details
  uint32_t w = 0;
  uint32_t h = 0;
  rawi_file.read(reinterpret_cast<char*>(&w), sizeof(w));
  rawi_file.read(reinterpret_cast<char*>(&h), sizeof(h));
  width = w;
  height = h;
  rawi_file.close();
}

DICE_LIB_DLL_EXPORT
void read_rawi_image(const char * file_name,
  intensity_t * intensities,
  const bool is_layout_right){

  std::ifstream rawi_file (file_name, std::ifstream::in | std::ifstream::binary);
  if (!rawi_file.is_open()){
    std::cerr << "ERROR: Can't open the file: " + (std::string)file_name << std::endl;
    exit(1);
  }
  // read the file details
  uint32_t w = 0;
  uint32_t h = 0;
  uint32_t num_bytes = 0;
  rawi_file.read(reinterpret_cast<char*>(&w), sizeof(uint32_t));
  rawi_file.read(reinterpret_cast<char*>(&h), sizeof(uint32_t));
  rawi_file.read(reinterpret_cast<char*>(&num_bytes), sizeof(uint32_t));
  // check that the byte size of the intensity values in the file is compatible with the current
  // size used to store intensity_t values
  if(num_bytes!=sizeof(intensity_t)){
    std::cerr << "Can't open file because it was saved using a different basic type for intensity_t: " + (std::string)file_name << std::endl;
    exit(1);
  }
  // read the image data:
  for (uint32_t y=0; y<h; ++y) {
    if(is_layout_right)
      for (uint32_t x=0; x<w;++x){
        rawi_file.read(reinterpret_cast<char*>(&intensities[y*w+x]),sizeof(intensity_t));
      }
    else // otherwise assume layout left
      for (uint32_t x=0; x<w;++x){
        rawi_file.read(reinterpret_cast<char*>(&intensities[x*h+y]),sizeof(intensity_t));
      }
  }
  rawi_file.close();
}

DICE_LIB_DLL_EXPORT
void write_rawi_image(const char * file_name,
  const int_t width,
  const int_t height,
  intensity_t * intensities,
  const bool is_layout_right){
  assert(width > 0);
  assert(height > 0);

  // TODO make sure this cast is okay
  uint32_t w = (uint32_t)width;
  uint32_t h = (uint32_t)height;
  uint32_t num_bytes = sizeof(intensity_t);
  //create a new file:
  std::ofstream rawi_file (file_name, std::ofstream::out | std::ofstream::binary);
  if (!rawi_file.is_open()){
    std::cerr << "ERROR: Can't open the file: " + (std::string)file_name << std::endl;
    exit(1);
  }
  // write the file details
  rawi_file.write(reinterpret_cast<char*>(&w), sizeof(uint32_t));
  rawi_file.write(reinterpret_cast<char*>(&h), sizeof(uint32_t));
  rawi_file.write(reinterpret_cast<char*>(&num_bytes), sizeof(uint32_t));

  // write the image data:
  for (int_t y=0; y<height; ++y) {
    if(is_layout_right){
      for (int_t x=0; x<width;++x){
        rawi_file.write(reinterpret_cast<char*>(&intensities[y*width + x]),sizeof(intensity_t));
      }
    }
    else // otherwise assume layout left
      for (int_t x=0; x<width;++x){
        rawi_file.write(reinterpret_cast<char*>(&intensities[x*height+y]),sizeof(intensity_t));
      }
  }
  rawi_file.close();
}

} // end namespace utils
} // end namespace DICe
