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
#include <DICe_Global.h>
#include <DICe_Parser.h>
#include <DICe_ParameterUtilities.h>

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

  *outStream << "creating the global parameter list" << std::endl;

  Teuchos::RCP<Teuchos::ParameterList> global_params = Teuchos::rcp(new Teuchos::ParameterList());
  global_params->set(DICe::global_solver,GMRES_SOLVER);
  global_params->set(DICe::output_folder,"");
  Teuchos::ParameterList mms_sublist;
  mms_sublist.set(DICe::problem_name,"div_curl_modulator");
  mms_sublist.set(DICe::phi_coeff,10.0);
  mms_sublist.set(DICe::b_coeff,2.0);
  global_params->set(DICe::mms_spec,mms_sublist);
  global_params->print(*outStream);

  std::vector<std::string> out_file_name;
  std::vector<Global_Formulation> formulation;
  std::vector<scalar_t> alpha;

  // HORN SCHNUNCK
  formulation.push_back(HORN_SCHUNCK);
  out_file_name.push_back("test_global_alg_hs");
  alpha.push_back(1.0);
  // MIXED HORN SCHUNCK
  formulation.push_back(MIXED_HORN_SCHUNCK);
  out_file_name.push_back("test_global_alg_hs_mixed");
  alpha.push_back(1.0);
  // LEHOUCQ TURNER
  formulation.push_back(LEHOUCQ_TURNER);
  out_file_name.push_back("test_global_alg_lt_mixed");
  alpha.push_back(1.0);
  // LEVENBERG_MARQUARDT
  formulation.push_back(LEVENBERG_MARQUARDT);
  out_file_name.push_back("test_global_alg_lm");
  alpha.push_back(1.0);
  // UNREGULARIZED
  formulation.push_back(UNREGULARIZED);
  out_file_name.push_back("test_global_alg_unreg");
  alpha.push_back(1.0);

  const scalar_t error_max = 0.1;
  TEUCHOS_TEST_FOR_EXCEPTION(formulation.size()!=out_file_name.size()||formulation.size()!=alpha.size(),
    std::runtime_error,"Error missing a parameter");
  std::vector<scalar_t> error_x(formulation.size(),-1.0);
  std::vector<scalar_t> error_y(formulation.size(),-1.0);
  std::vector<scalar_t> error_l(formulation.size(),-1.0);
  std::vector<scalar_t> max_error_x(formulation.size(),-1.0);

  for(size_t i=0;i<formulation.size();++i){
    scalar_t error_bx = 0.0;
    scalar_t error_by = 0.0;
    scalar_t error_lambda = 0.0;
    scalar_t max_error_bx = 0.0;
    scalar_t max_error_by = 0.0;
    scalar_t max_error_lambda = 0.0;
    *outStream << " TESTING " << to_string(formulation[i]) << " FORMULATION " << std::endl;
    global_params->set(DICe::global_regularization_alpha,alpha[i]);
    global_params->set(DICe::global_stabilization_tau,0.0);
    global_params->set(DICe::global_formulation,formulation[i]);
    global_params->set(DICe::output_prefix,out_file_name[i]);
    global_params->set(DICe::parser_use_regular_grid,true);
    global_params->set(DICe::parser_enforce_lagrange_bc,true);
    global_params->set(DICe::num_image_integration_points,75);
    //if(formulation[i]==UNREGULARIZED)
    //  global_params->set(DICe::mesh_size,10000.0);
    //else
    global_params->set(DICe::mesh_size,25.0);
    *outStream << "creating a global algorithm" << std::endl;
    Teuchos::RCP<DICe::global::Global_Algorithm> global_alg = Teuchos::rcp(new DICe::global::Global_Algorithm(global_params));
    //*outStream << "pre-execution tasks" << std::endl;
    //global_alg->pre_execution_tasks();
    *outStream << "executing" << std::endl;
    global_alg->execute();
    *outStream << "post execution tasks" << std::endl;
    global_alg->post_execution_tasks(1.0);
    *outStream << "evaluating the error" << std::endl;
    global_alg->evaluate_mms_error(error_bx,error_by,error_lambda,max_error_bx,max_error_by,max_error_lambda);
    error_x[i] = error_bx;
    error_y[i] = error_by;
    error_l[i] = error_lambda;
    max_error_x[i] = max_error_bx;
    if(formulation[i]!=UNREGULARIZED&&(error_bx > error_max || error_by > error_max)){
      *outStream << "error, the solution error is too large for " << to_string(formulation[i]) << std::endl;
      errorFlag++;
    }
  } // end formulation loop

  *outStream << "-----------------------------------------------------------------------------------------------------------" << std::endl;
  *outStream << "Results Summary:" << std::endl;
  *outStream << "-----------------------------------------------------------------------------------------------------------" << std::endl;
  for(size_t i=0;i<formulation.size();++i){
    *outStream << std::setw(25) << to_string(formulation[i]) << " error x: " << std::setw(15) << error_x[i] << " error y: "
        << std::setw(15) << error_y [i] << " error l: " << std::setw(15) << error_l[i] << " max error x: " << max_error_x[i] << std::endl;
  }
  *outStream << "-----------------------------------------------------------------------------------------------------------" << std::endl;

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

