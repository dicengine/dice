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

/*! \file  DICe_TestTensor.cpp
    \brief Test 2D Tensor
*/

#include <DICe_Utilities.h>
#include <DICe.h>
#include <DICe_Kokkos.h>

#include <Teuchos_oblackholestream.hpp>

#include <iostream>
#include <fstream>
#include <cstdio>

using namespace DICe;

int main(int argc, char *argv[]) {

  // initialize kokkos
  Kokkos::initialize(argc, argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int_t iprint     = argc - 1;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  int_t errorFlag  = 0;
  scalar_t errtol  = 1.0E-4;

  *outStream << "--- Begin test ---" << std::endl;

  // create a 2D tensor and test its values
  DICe::Tensor_2D<scalar_t> tensor(1.5,2.5,3.5,4.5);

  if(std::abs(tensor.xx - 1.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensor.XX is wrong\n" ;
    errorFlag++;
  }
  if(std::abs(tensor.xy - 2.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensor.XY is wrong\n" ;
    errorFlag++;
  }
  if(std::abs(tensor.yx - 3.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensor.YX is wrong\n" ;
    errorFlag++;
  }
  if(std::abs(tensor.yy - 4.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensor.YY is wrong\n" ;
    errorFlag++;
  }

  DICe::Tensor_2D<scalar_t> tensorT = tensor.transpose();
  if(std::abs(tensorT.xx - 1.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensorT.XX is wrong\n" ;
    errorFlag++;
  }
  if(std::abs(tensorT.xy - 3.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensorT.XY is wrong\n" ;
    errorFlag++;
  }
  if(std::abs(tensorT.yx - 2.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensorT.YX is wrong\n" ;
    errorFlag++;
  }
  if(std::abs(tensorT.yy - 4.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensorT.YY is wrong\n" ;
    errorFlag++;
  }

  DICe::Tensor_2D<scalar_t> tensor0p5 = tensorT*0.5;
  if(std::abs(tensor0p5.xx - 1.5*0.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensor0p5.XX is wrong\n" ;
    errorFlag++;
  }
  if(std::abs(tensor0p5.xy - 3.5*0.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensor0p5.XY is wrong\n" ;
    errorFlag++;
  }
  if(std::abs(tensor0p5.yx - 2.5*0.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensor0p5.YX is wrong\n" ;
    errorFlag++;
  }
  if(std::abs(tensor0p5.yy - 4.5*0.5) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensor0p5.YY is wrong\n" ;
    errorFlag++;
  }

  DICe::Tensor_2D<scalar_t> eye(1.0);
  DICe::Tensor_2D<scalar_t> tensorComplex = tensorT - eye + tensor*tensorT;
  if(std::abs(tensorComplex.xx - (0.5 + 2.25+6.25)) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensorComplex.XX is wrong, value: " << tensorComplex.xx << "\n" ;
    errorFlag++;
  }
  if(std::abs(tensorComplex.xy - (3.5 + 5.25+11.25)) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensorComplex.XY is wrong, value: " << tensorComplex.xy << "\n" ;
    errorFlag++;
  }
  if(std::abs(tensorComplex.yx - (2.5 + 5.25+11.25)) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensorComplex.YX is wrong, value: " << tensorComplex.yx << "\n" ;
    errorFlag++;
  }
  if(std::abs(tensorComplex.yy - (3.5 + 12.25+20.25)) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! tensorComplex.YY is wrong, value: " << tensorComplex.yy << "\n" ;
    errorFlag++;
  }

  // test the inverse
  DICe::Tensor_2D<scalar_t> test(2.6,1.4,6.9,11);
  DICe::Tensor_2D<scalar_t> testInv = test.inverse();
  //*outStream << " testInv xx: " << testInv.xx << " testInv xy: " << testInv.xy << " testInv yx: " << testInv.yx << " testInv yy: " << testInv.yy << std::endl;
  if(std::abs(testInv.xx - 0.580781) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! testInv.XX is wrong\n";
    errorFlag++;
  }
  if(std::abs(testInv.xy + 0.0739176) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! testInv.XY is wrong\n";
    errorFlag++;
  }
  if(std::abs(testInv.yx + 0.364308) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! testInv.YX is wrong\n";
    errorFlag++;
  }
  if(std::abs(testInv.yy - 0.137276) > errtol){
    *outStream << "---> POSSIBLE ERROR ABOVE! testInv.YY is wrong\n";
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

