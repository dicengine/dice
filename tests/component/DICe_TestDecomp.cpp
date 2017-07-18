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

/*! \file  DICe_TestDecomp.cpp
    \brief Test 2D Tensor
*/

#include <DICe.h>
#include <DICe_Decomp.h>
#include <DICe_Parser.h>

#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int_t iprint     = argc - 1;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  int_t errorFlag  = 0;

  *outStream << "--- Begin test ---" << std::endl;

  // create the images folder if it doesn't exist
#if defined(WIN32)
  std::string input_file = ".\\decomp\\input.xml";
  std::string cine_input_file = ".\\decomp\\cine_input.xml";
  std::string params_file = ".\\decomp\\params.xml";
  std::string cine_params_file = ".\\decomp\\cine_params.xml";
#else
  std::string input_file = "./decomp/input.xml";
  std::string cine_input_file = "./decomp/cine_input.xml";
  std::string params_file = "./decomp/params.xml";
  std::string cine_params_file = "./decomp/cine_params.xml";
#endif

  Teuchos::RCP<Teuchos::ParameterList> inputParams = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::Ptr<Teuchos::ParameterList> inputParamsPtr(inputParams.get());
  Teuchos::updateParametersFromXmlFile(input_file, inputParamsPtr);
  TEUCHOS_TEST_FOR_EXCEPTION(inputParams==Teuchos::null,std::runtime_error,"");

  Teuchos::RCP<Teuchos::ParameterList> correlationParams = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::Ptr<Teuchos::ParameterList> correlationParamsPtr(correlationParams.get());
  Teuchos::updateParametersFromXmlFile(params_file, correlationParamsPtr);
  TEUCHOS_TEST_FOR_EXCEPTION(correlationParams==Teuchos::null,std::runtime_error,"");

  Teuchos::RCP<Decomp> decomp = Teuchos::rcp(new Decomp(inputParams,correlationParams));

  if(decomp->num_global_subsets()!=123){
    errorFlag++;
    *outStream << "Error, wrong number of global subsets" << std::endl;
  }

  Teuchos::RCP<Teuchos::ParameterList> cineInputParams = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::Ptr<Teuchos::ParameterList> cineInputParamsPtr(cineInputParams.get());
  Teuchos::updateParametersFromXmlFile(cine_input_file, cineInputParamsPtr);
  TEUCHOS_TEST_FOR_EXCEPTION(cineInputParams==Teuchos::null,std::runtime_error,"");

  Teuchos::RCP<Teuchos::ParameterList> cineCorrelationParams = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::Ptr<Teuchos::ParameterList> cineCorrelationParamsPtr(cineCorrelationParams.get());
  Teuchos::updateParametersFromXmlFile(cine_params_file, cineCorrelationParamsPtr);
  TEUCHOS_TEST_FOR_EXCEPTION(cineCorrelationParams==Teuchos::null,std::runtime_error,"");

  Teuchos::RCP<Decomp> cine_decomp = Teuchos::rcp(new Decomp(cineInputParams,cineCorrelationParams));

  if(cine_decomp->num_global_subsets()!=168){
    errorFlag++;
    *outStream << "Error, wrong number of global subsets" << std::endl;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

