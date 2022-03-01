// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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
#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_ImageIO.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include "opencv2/opencv.hpp"

#include <iostream>


using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // TODO add .avi example (uncomment the code below) when we sort out how to read avi on mac

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

  // create a basline image to use as a

  const int_t w = 640;
  const int_t h = 480;
  const double fps = 26.777;
  const int_t num_frames = 31;

  Teuchos::RCP<cv::VideoCapture> cup = DICe::utils::Video_Singleton::instance().video_capture("./images/cup.mp4");
  if(cup->get(cv::CAP_PROP_FRAME_WIDTH)!=w){
    errorFlag++;
    *outStream << "Error, cup mp4 width " << cup->get(cv::CAP_PROP_FRAME_WIDTH) << " is not correct " << w << std::endl;
  }
  if(cup->get(cv::CAP_PROP_FRAME_HEIGHT)!=h){
    errorFlag++;
    *outStream << "Error, cup mp4 height " << cup->get(cv::CAP_PROP_FRAME_HEIGHT) << " is not correct " << h << std::endl;
  }
  if(std::abs(cup->get(cv::CAP_PROP_FPS)-fps) > 0.1){
    errorFlag++;
    *outStream << "Error fps " << cup->get(cv::CAP_PROP_FPS) << " is not correct for mp4" << std::endl;
  }
  if(cup->get(cv::CAP_PROP_FRAME_COUNT)!=num_frames){
    errorFlag++;
    *outStream << "Error num_frames " << cup->get(cv::CAP_PROP_FRAME_COUNT) << " is not correct for mp4" << std::endl;
  }

  // use this code to output the test images
//  for(int_t i=0;i<5;++i){
//    std::stringstream filename, fileout;
//    filename << "./images/cup_" << i << ".mp4";
//    fileout << "./images/cup_" << i << ".png";
//    DICe::Image img(filename.str().c_str());
//    img.write(fileout.str());
//  }

  for(int_t i=0;i<5;++i){
    std::stringstream filename, filegold, fileout;
    filename << "./images/cup_" << i << ".mp4";
    filegold << "./images/cup_" << i << ".png";
    fileout << "cup_" << i << ".png";
    Teuchos::RCP<DICe::Image> img = Teuchos::rcp(new DICe::Image(filename.str().c_str()));
    img->write(fileout.str());
    Teuchos::RCP<DICe::Image> img_gold = Teuchos::rcp(new DICe::Image(filegold.str().c_str()));
    const scalar_t diff = img->diff(img_gold);
    *outStream << "Testing frame " << i << " diff " << diff << std::endl;
    if(diff > w*h){
      errorFlag++;
      *outStream << "Error image diff too high" << std::endl;
    }
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

