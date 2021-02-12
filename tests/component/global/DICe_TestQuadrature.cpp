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

#include <DICe.h>
#include <DICe_Mesh.h>

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

  *outStream << "Testing point locations for inexact rule (pixel aligned)" << std::endl;

  const int_t image_integration_order = 1000;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<precision_t> > gp_locs;
  Teuchos::ArrayRCP<precision_t> gp_weights;
  int_t num_image_integration_points = -1;
  tri2d_nonexact_integration_points(image_integration_order,
    gp_locs,gp_weights,num_image_integration_points);

  precision_t sum_w = 0.0;
  *outStream << "gp_locs: " << std::endl;
  for(int_t i=0;i<num_image_integration_points;++i){
    //*outStream << "xi: " << gp_locs[i][0] << " eta: " << gp_locs[i][1] <<
    //    " w: " << gp_weights[i] << std::endl;
    sum_w+=gp_weights[i];
  }
  if(std::abs(sum_w-0.5)>1.0E-5){
    *outStream << "Error, weights are not correct, sum: " << sum_w << " (should be 1.0)" << std::endl;
    errorFlag++;
  }

  // check integration of a linear function over a single element
  precision_t func_total = 0.0;
  for(int_t i=0;i<num_image_integration_points;++i){
    func_total += gp_locs[i][1]*gp_weights[i];
  }
  *outStream << "func_total " << func_total << std::endl;
  if(std::abs(func_total-1.0/6.0)>1.0E-5){
    *outStream << "Error, function integration error too high: " << std::abs(func_total-1.0/3.0) << std::endl;
    errorFlag++;
  }

  for(int_t order=1;order<7;++order){
    *outStream << "testing Gauss order " << order << std::endl;
    const int_t gauss_order = order;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<precision_t> > gauss_locs;
    Teuchos::ArrayRCP<precision_t> gauss_weights;
    int_t num_gauss_integration_points = -1;
    tri2d_natural_integration_points(gauss_order,
      gauss_locs,gauss_weights,num_gauss_integration_points);
    precision_t sum_gw = 0.0;
    for(int_t i=0;i<num_gauss_integration_points;++i){
      sum_gw+=gauss_weights[i];
    }
    if(std::abs(sum_gw-0.5)>1.0E-5){
      *outStream << "Error, Gauss weights are not correct, sum: " << sum_gw << " (should be 1.0)" << std::endl;
      errorFlag++;
    }
    precision_t func_total = 0.0;
    for(int_t i=0;i<num_gauss_integration_points;++i){
      func_total += gauss_locs[i][1]*gauss_weights[i];
    }
    *outStream << "func_total " << func_total << std::endl;
    if(std::abs(func_total-1.0/6.0)>1.0E-5){
      *outStream << "Error, function integration error too high: " << std::abs(func_total-1.0/6.0) << std::endl;
      errorFlag++;
    }
    *outStream << "order: " << order << " sum_gw " << sum_gw << " func_total " << func_total << std::endl;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

