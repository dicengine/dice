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
#include <DICe_Feature.h>

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
  scalar_t errorTol = 1.0;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  Teuchos::RCP<Image> left_img = Teuchos::rcp(new Image("../images/refSpeckled.tif"));
  Teuchos::RCP<Image> right_img = Teuchos::rcp(new Image("../images/shiftSpeckled_160x140y.tif"));

  std::vector<scalar_t> left_x;
  std::vector<scalar_t> left_y;
  std::vector<scalar_t> right_x;
  std::vector<scalar_t> right_y;

  const int_t num_expected_matches = 1800;
  // set up a range of expected matches since the opencv feature matching changes frequently
  // and leads to diffs
  const int_t num_expected_min = num_expected_matches * 0.75;
  const int_t num_expected_max = num_expected_matches * 1.25;

  const float tol = 0.001f;
  match_features(left_img,right_img,left_x,left_y,right_x,right_y,tol,"res.png");
  const int_t num_matches = left_x.size();
  *outStream << "number of features matched: " << num_matches << std::endl;
  if(num_matches<num_expected_min||num_matches>num_expected_max){//!=1962){
    errorFlag++;
    *outStream << "Error wrong number of matching features detected. Should be between 1350 and 2250 and is " << num_matches << std::endl;
  }

  for(int_t i=0;i<num_matches;++i){
    const scalar_t dist_x = right_x[i] - left_x[i];
    const scalar_t dist_y = right_y[i] - left_y[i];
    if(std::abs(dist_x - 160) > errorTol){
      errorFlag++;
      *outStream << "Matching point error in x left (" << left_x[i] << "," << left_y[i] << ") right (" << right_x[i] << "," << right_y[i] << ") dist " << dist_x <<
          " should be 160" << std::endl;
    }
    if(std::abs(dist_y - 140) > errorTol){
      errorFlag++;
      *outStream << "Matching point error in y left (" << left_x[i] << "," << left_y[i] << ") right (" << right_x[i] << "," << right_y[i] << ") dist " << dist_y <<
          " should be 140" << std::endl;
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

