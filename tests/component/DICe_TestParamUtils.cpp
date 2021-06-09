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

/*! \file  DICe_TestParamUtils.cpp
    \brief Test of various parameters that can be set for a correlation.
    We try to hit all combinations below for a square subset
*/

#include <DICe_ParameterUtilities.h>
#include <DICe.h>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_RCP.hpp>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  int_t iprint     = argc - 1;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);
  int_t errorFlag  = 0;

  *outStream << "--- Begin test ---" << std::endl;

  for(int_t meth=0;meth<DICe::MAX_PROJECTION_METHOD;++meth){
    std::string methStr = DICe::to_string(static_cast<DICe::Projection_Method>(meth));
    assert(methStr == DICe::projectionMethodStrings[meth]);
    // convert string to lower case to test toUpper:
    DICe::stringToLower(methStr);
    *outStream << "projection method: " << methStr << std::endl;
    if(meth!=DICe::string_to_projection_method(methStr)){
      *outStream << "Error, projection method is wrong" << std::endl;
      errorFlag++;
    }
  }
  *outStream << "projection method success " << std::endl;

  for(int_t meth=0;meth<DICe::MAX_INITIALIZATION_METHOD;++meth){
    std::string methStr = DICe::to_string(static_cast<DICe::Initialization_Method>(meth));
    assert(methStr == DICe::initializationMethodStrings[meth]);
    // convert string to lower case to test toUpper:
    DICe::stringToLower(methStr);
    *outStream << "initialization method: " << methStr << std::endl;
    if(meth!=DICe::string_to_initialization_method(methStr)){
      *outStream << "Error, initialization method is wrong" << std::endl;
      errorFlag++;
    }
  }
  *outStream << "initialization method success " << std::endl;

  for(int_t meth=0;meth<DICe::MAX_OPTIMIZATION_METHOD;++meth){
    std::string methStr = DICe::to_string(static_cast<DICe::Optimization_Method>(meth));
    assert(methStr == DICe::optimizationMethodStrings[meth]);
    // convert string to lower case to test toUpper:
    DICe::stringToLower(methStr);
    *outStream << "optimization method: " << methStr << std::endl;
    if(meth!=DICe::string_to_optimization_method(methStr)){
      *outStream << "Error, optimization method is wrong" << std::endl;
      errorFlag++;
    }
  }
  *outStream << "optimization method success " << std::endl;

  for(int_t meth=0;meth<DICe::MAX_INTERPOLATION_METHOD;++meth){
    std::string methStr = DICe::to_string(static_cast<DICe::Interpolation_Method>(meth));
    assert(methStr == DICe::interpolationMethodStrings[meth]);
    // convert string to lower case to test toUpper:
    DICe::stringToLower(methStr);
    *outStream << "interpolation method: " << methStr << std::endl;
    if(meth!=DICe::string_to_interpolation_method(methStr)){
      *outStream << "Error, interpolation method is wrong" << std::endl;
      errorFlag++;
    }
  }
  *outStream << "interpolation method success " << std::endl;

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

