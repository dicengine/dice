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

#include <cassert>
#include <iostream>
#include <fstream>

#include <DICe_Rawi.h>

#include <Teuchos_TestForException.hpp>

namespace DICe{

DICE_LIB_DLL_EXPORT
void read_rawi_image_dimensions(const char * file_name,
  size_t & width,
  size_t & height){

  std::ifstream rawi_file (file_name, std::ifstream::in | std::ifstream::binary);
  if (rawi_file.fail()){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"ERROR: Can't open the file: " + (std::string)file_name);
  }
  // read the file details
  uint32_t w = 0;
  uint32_t h = 0;
  uint32_t num_bytes = 0;
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
  assert(file_name!="");

  std::ifstream rawi_file (file_name, std::ifstream::in | std::ifstream::binary);
  if (!rawi_file.is_open()){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"ERROR: Can't open the file: " + (std::string)file_name);
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
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Can't open file because it was saved using a different basic type for intensity_t: " + (std::string)file_name);
  }
  // read the image data:
  for (size_t y=0; y<h; ++y) {
    if(is_layout_right)
      for (size_t x=0; x<w;++x){
        rawi_file.read(reinterpret_cast<char*>(&intensities[y*w+x]),sizeof(intensity_t));
      }
    else // otherwise assume layout left
      for (size_t x=0; x<w;++x){
        rawi_file.read(reinterpret_cast<char*>(&intensities[x*h+y]),sizeof(intensity_t));
      }
  }
  rawi_file.close();
}

DICE_LIB_DLL_EXPORT
void write_rawi_image(const char * file_name,
  const size_t width,
  const size_t height,
  intensity_t * intensities,
  const bool is_layout_right){
  assert(file_name!="");
  assert(width > 0);
  assert(height > 0);

  // TODO make sure this cast is okay
  uint32_t w = (uint32_t)width;
  uint32_t h = (uint32_t)height;
  uint32_t num_bytes = sizeof(intensity_t);
  //create a new file:
  std::ofstream rawi_file (file_name, std::ofstream::out | std::ofstream::binary);
  if (!rawi_file.is_open()){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"ERROR: Can't open the file: " + (std::string)file_name);
  }
  // write the file details
  rawi_file.write(reinterpret_cast<char*>(&w), sizeof(uint32_t));
  rawi_file.write(reinterpret_cast<char*>(&h), sizeof(uint32_t));
  rawi_file.write(reinterpret_cast<char*>(&num_bytes), sizeof(uint32_t));

  // write the image data:
  for (size_t y=0; y<height; ++y) {
    if(is_layout_right){
      for (size_t x=0; x<width;++x){
        rawi_file.write(reinterpret_cast<char*>(&intensities[y*width + x]),sizeof(intensity_t));
      }
    }
    else // otherwise assume layout left
      for (size_t x=0; x<width;++x){
        rawi_file.write(reinterpret_cast<char*>(&intensities[x*height+y]),sizeof(intensity_t));
      }
  }
  rawi_file.close();
}

} // end namespace DICe
