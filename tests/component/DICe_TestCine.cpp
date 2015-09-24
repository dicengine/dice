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

#include <DICe.h>
#include <DICe_Kokkos.h>
#include <DICe_Cine.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  // initialize kokkos
  Kokkos::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  size_t iprint     = argc - 1;
  size_t errorFlag  = 0;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  *outStream << "reading a packed cine file " << std::endl;
  DICe::cine::Cine_Reader cine_reader("./images/CinePacked.cine",outStream.getRawPtr());
  if(cine_reader.num_images()!=6){ // there are six images in this file
    *outStream << "Error, the number of images is not correct, should be 6 is " << cine_reader.num_images() << std::endl;
    errorFlag++;
  }
  if(cine_reader.image_height()!=128){
    *outStream << "Error, the image height is not correct. Should be 128 and is" << cine_reader.image_height() << std::endl;
    errorFlag++;
  }
  if(cine_reader.image_width()!=256){
    *outStream << "Error, the image width is not correct. Should be 256 and is " << cine_reader.image_width() << std::endl;
    errorFlag++;
  }

  *outStream << "checked basic image dimensions" << std::endl;

  *outStream << "reading images from .rawi files to compare to the cine" << std::endl;

  for(size_t i=0;i<cine_reader.num_images();++i){
    *outStream << "testing image " << i << std::endl;
    Teuchos::RCP<Image> cine_img = cine_reader.get_image(i);
    std::cout << " first 20 from cine " << std::endl;
    for(size_t j=0;j<20;++j)
      std::cout << j << " " << cine_img->intensities().h_view.ptr_on_device()[j] << std::endl;
    std::stringstream name;
    std::stringstream outname;
    outname << "Cine" << i << ".rawi";
    //cine_img->write_rawi(outname.str());
    name << "Cine" << i << ".rawi";
    Image cine_img_exact(name.str().c_str());
    std::cout << " first 20 from rawi" << std::endl;
    for(size_t j=0;j<20;++j)
      std::cout << j << " " << cine_img_exact.intensities().h_view.ptr_on_device()[j] << std::endl;

    bool intensity_value_error = false;
    for(size_t j=0;j<cine_reader.image_height()*cine_reader.image_width();++j){
      if(cine_img_exact.intensities().h_view.ptr_on_device()[j]!=cine_img->intensities().h_view.ptr_on_device()[j]){
        intensity_value_error = true;
        std::cout << " j " << j << " expected " << cine_img_exact.intensities().h_view.ptr_on_device()[j] << " actual " << cine_img->intensities().h_view.ptr_on_device()[j] << std::endl;
      }
    }
//    for(size_t y=0;y<cine_reader.image_height();++y){
//      for(size_t x=0;x<cine_reader.image_width();++x){
//        if((*cine_img)(x,y)!=cine_img_exact(x,y)){
//          std::cout << x << " " << y << " actual " << (*cine_img)(x,y) << " exptected " << cine_img_exact(x,y) << std::endl;
//          intensity_value_error=true;
//        }
//      }
//    }
    if(intensity_value_error){
      *outStream << "Error, image " << i << ", the intensity values for the packed cine file are not correct" << std::endl;
      errorFlag++;
    }
  }

//
//  *outStream << "reading a raw packed cine file" << std::endl;
//  DICe::cine::Cine_Reader raw_cine_reader("./images/CineRawPacked.cine",outStream.getRawPtr());
//  if(raw_cine_reader.num_images()!=6){ // there are six images in this file
//    *outStream << "Error, the number of images is not correct, should be 6 is " << raw_cine_reader.num_images() << std::endl;
//    errorFlag++;
//  }
//  if(raw_cine_reader.image_height()!=128){
//    *outStream << "Error, the image height is not correct. Should be 128 and is" << raw_cine_reader.image_height() << std::endl;
//    errorFlag++;
//  }
//  if(raw_cine_reader.image_width()!=256){
//    *outStream << "Error, the image width is not correct. Should be 256 and is " << raw_cine_reader.image_width() << std::endl;
//    errorFlag++;
//  }

  *outStream << "--- End test ---" << std::endl;

  // finalize kokkos
  Kokkos::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

