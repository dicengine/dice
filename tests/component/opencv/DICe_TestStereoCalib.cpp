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
#include <DICe_StereoCalib.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>

using namespace DICe;



int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

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

  *outStream << "testing checkerboard grid images " << std::endl;

  std::vector<std::string> image_list;
  for(int_t i=1;i<10;++i){
    std::stringstream left_name;
    std::stringstream right_name;
    left_name << "../images/left0" << i << ".jpg";
    right_name << "../images/right0" << i << ".jpg";
    image_list.push_back(left_name.str());
    image_list.push_back(right_name.str());
  }
  for(int_t i=10;i<15;++i){
    std::stringstream left_name;
    std::stringstream right_name;
    left_name << "../images/left" << i << ".jpg";
    right_name << "../images/right" << i << ".jpg";
    image_list.push_back(left_name.str());
    image_list.push_back(right_name.str());
  }
  *outStream << "image list: " << std::endl;
  for(size_t i=0;i<image_list.size();++i){
    *outStream << image_list[i] << std::endl;
  }

  cv::Size board_size;
  board_size.width = 9;
  board_size.height = 6;

  const float square_size = 1.0;
  int mode = 0; // checkerboard
  const float rms = StereoCalib(mode, image_list, board_size, square_size, true, false, "checkerboard_cal.txt");

  if(rms <0.0 || rms > 0.75){
    *outStream << "Error, rms error too high or negative: " << rms << std::endl;
    errorFlag++;
  }

  *outStream << "testing circle grid images" << std::endl;

  std::vector<std::string> image_list_circ;
  for(int_t i=1;i<10;++i){
    std::stringstream left_name;
    std::stringstream right_name;
    left_name << "../images/CalB-sys2-000" << i << "_0.jpeg";
    right_name << "../images/CalB-sys2-000" << i << "_1.jpeg";
    image_list_circ.push_back(left_name.str());
    image_list_circ.push_back(right_name.str());
  }
  for(int_t i=10;i<15;++i){
    std::stringstream left_name;
    std::stringstream right_name;
    left_name << "../images/CalB-sys2-00" << i << "_0.jpeg";
    right_name << "../images/CalB-sys2-00" << i << "_1.jpeg";
    image_list_circ.push_back(left_name.str());
    image_list_circ.push_back(right_name.str());
  }
  *outStream << "circle grid image list: " << std::endl;
  for(size_t i=0;i<image_list_circ.size();++i){
    *outStream << image_list_circ[i] << std::endl;
  }

  cv::Size board_size_circ;
  board_size_circ.width = 6;
  board_size_circ.height = 4;

  const float circ_size = 10.0;
  mode = 1; // vic3d circle grid with marker dots and possible other random dots in the pattern
  const float rms_circ = StereoCalib(mode,image_list_circ, board_size_circ, circ_size, true, false, "circle_cal.txt");

  if(rms_circ <0.0 || rms_circ > 0.75){
    *outStream << "Error, rms error too high or negative for circle grid: " << rms_circ << std::endl;
    errorFlag++;
  }


  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

