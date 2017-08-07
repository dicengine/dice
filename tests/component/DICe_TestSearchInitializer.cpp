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

/*! \file  DICe_TestSearchInitializer.cpp
    \brief Testing of Search_Path_Initializer class
*/

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Subset.h>
#include <DICe_Initializer.h>
#include <DICe_Schema.h>
#include <DICe_ParameterUtilities.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <iostream>
#include <fstream>
#include <cassert>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  const scalar_t errorTol = 1.0E-4;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  const int_t subset_size = 27;
  const int_t cx = 151,cy=277;
  const scalar_t u_exact = 138.0;
  const scalar_t v_exact = -138.0;
  Teuchos::RCP<DICe::Schema> schema = Teuchos::rcp(new DICe::Schema());
  schema->set_ref_image("./images/InitRef.tif");
  schema->set_def_image("./images/InitDef.tif");
  Teuchos::RCP<DICe::Subset> subset = Teuchos::rcp(new Subset(cx,cy,subset_size,subset_size));
  subset->initialize(schema->ref_img(),REF_INTENSITIES);
  const scalar_t step_size_xy = 1.0;
  const scalar_t step_size_theta = 0.5;
  const scalar_t search_dim_xy = 140;
  const scalar_t search_dim_theta = 1;
  Search_Initializer searcher(schema.getRawPtr(),subset,step_size_xy,search_dim_xy,step_size_theta,search_dim_theta);
  Teuchos::RCP<std::vector<scalar_t> > def = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));
  searcher.initial_guess(-1,def);
  if(std::abs((*def)[DOF_U] - u_exact) > errorTol || std::abs((*def)[DOF_V] - v_exact) > errorTol){
    *outStream << "Error, the initialized value is not correct" << std::endl;
    *outStream << "       should be 138,-138 and is " << (*def)[DOF_U] << "," << (*def)[DOF_V] << std::endl;
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

