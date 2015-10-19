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
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  std::vector<std::string> cine_files;
  cine_files.push_back("packed_12bpp");
  cine_files.push_back("packed_raw_12bpp");
  cine_files.push_back("phantom_v12_raw_16bpp");
  cine_files.push_back("phantom_v1610");
  cine_files.push_back("phantom_v1610_16bpp");
  cine_files.push_back("phantom_v1610_raw");
  cine_files.push_back("phantom_v1610_raw_16bpp");
  cine_files.push_back("phantom_v2511_raw_16bpp");
  cine_files.push_back("phantom_v611_raw_16bpp");
  cine_files.push_back("phantom_v7_raw_16bpp");
  cine_files.push_back("phantom_v9_raw_16bpp");

  // all of these cine files should be dimensions 128 x 256 and have 6 frames each

  for(int_t i=0;i<cine_files.size();++i){
    std::stringstream full_name;
    full_name << "./images/" << cine_files[i] << ".cine";
    *outStream << "testing cine file: " << full_name.str() << std::endl;

    DICe::cine::Cine_Reader cine_reader(full_name.str(),outStream.getRawPtr());
    if(cine_reader.num_frames()!=6){
      *outStream << "Error, the number of frames is not correct, should be 6 is " << cine_reader.num_frames() << std::endl;
      errorFlag++;
    }
    if(cine_reader.height()!=128){
      *outStream << "Error, the image height is not correct. Should be 128 and is" << cine_reader.height() << std::endl;
      errorFlag++;
    }
    if(cine_reader.width()!=256){
      *outStream << "Error, the image width is not correct. Should be 256 and is " << cine_reader.width() << std::endl;
      errorFlag++;
    }
    *outStream << "image dimensions have been checked" << std::endl;

    for(int_t frame=0;frame<cine_reader.num_frames();++frame){
      *outStream << "testing frame " << frame << std::endl;
      Teuchos::RCP<Image> cine_img = cine_reader.get_frame(frame);
      std::stringstream name;
      //std::stringstream outname;
      //std::stringstream tiffname;
      //outname << cine_files[i] << "_d_" << frame << ".rawi";
      //tiffname << cine_files[i] << "_" << frame << ".tiff";
      //cine_img->write_rawi(outname.str());
      //cine_img->write_tiff(tiffname.str());
#ifdef DICE_USE_DOUBLE
      name << "./images/" << cine_files[i]<< "_d_" << frame << ".rawi";
#else
      name << "./images/" << cine_files[i]<< "_" << frame << ".rawi";
#endif
      Image cine_img_exact(name.str().c_str());
      bool intensity_value_error = false;
      for(int_t y=0;y<cine_reader.height();++y){
        for(int_t x=0;x<cine_reader.width();++x){
          if((*cine_img)(x,y)!=cine_img_exact(x,y)){
            //std::cout << x << " " << y << " actual " << (*cine_img)(x,y) << " exptected " << cine_img_exact(x,y) << std::endl;
            intensity_value_error=true;
          }
        }
      }
      if(intensity_value_error){
        *outStream << "Error, image " << i << ", the intensity values are not correct" << std::endl;
        errorFlag++;
      }
      *outStream << "intensity values have been checked" << std::endl;
    }
  }

  *outStream << "testing invalid cine file for an exception" << std::endl;

  bool exception_thrown = false;
  try{
    DICe::cine::Cine_Reader cine_reader("./images/invalid_color.cine",outStream.getRawPtr());
  }
  catch(const std::exception &e){
    exception_thrown = true;
    *outStream << "exception thrown as expected." << std::endl;
  }
  if(!exception_thrown){
    *outStream << "Error, an exception should have been thrown for the compressed color cine file: invalid_color.cine" << std::endl;
    errorFlag++;
  }

  *outStream << "testing that invalid frame index throws an exception" << std::endl;
  exception_thrown = false;
  try{
    DICe::cine::Cine_Reader cine_reader("./images/packed_12bpp.cine",outStream.getRawPtr());
    Teuchos::RCP<Image> cine_img = cine_reader.get_frame(1000);
  }
  catch(const std::exception &e){
    exception_thrown=true;
    *outStream << "exception thrown as expected." << std::endl;
  }
  if(!exception_thrown){
    *outStream << "Error, an exception should have been thrown for an invalid frame index" << std::endl;
    errorFlag++;
  }

  *outStream << "--- End test ---" << std::endl;

  // finalize kokkos
  Kokkos::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

