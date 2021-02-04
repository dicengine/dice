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

/*! \file  DICe_TestCorrelateSquare.cpp
    \brief Test of various components of a correlation (Schema, interpolation, gradients, gamma...)
    all of these are implicitly tested through using the schema correlate method
*/

#include <DICe.h>
#include <DICe_Schema.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>
#include <cstdio>

#include <cassert>

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

  *outStream << "testing multiple square subsets generated by the schema" << std::endl;
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::initialization_method,DICe::USE_NEIGHBOR_VALUES);
  params->set(DICe::interpolation_method,DICe::KEYS_FOURTH);
  params->set(DICe::robust_solver_tolerance,1.0E-4);


  // get the dimensions of the images and set up an array of points
  Image img("./images/refSpeckled.tif");
  const int_t roi_w = img.width();
  const int_t roi_h = img.height();
  const int_t multiple_step_size_x = 101;
  const int_t multiple_step_size_y = 51;
  const int_t multiple_subset_size = 31;
  Teuchos::RCP<DICe::Schema> schemaMultiple = Teuchos::rcp(new DICe::Schema(roi_w,roi_h,multiple_step_size_x,multiple_step_size_y,multiple_subset_size,params));
  schemaMultiple->set_ref_image("./images/refSpeckled.tif");
  schemaMultiple->set_def_image("./images/defSpeckled.tif");
  schemaMultiple->execute_correlation();
  schemaMultiple->write_control_points_image("schemaMultipleControlPoints.tif");

  *outStream << "multiple square subsets results: " << std::endl;

  *outStream << "params: (See DICe.h for definitions)" << std::endl;
  *outStream << "    Correlation Routine:    " << schemaMultiple->correlation_routine() << std::endl;
  if(schemaMultiple->correlation_routine()!=DICe::GENERIC_ROUTINE){
    *outStream << "Error, wrong correlation routine" << std::endl;
    errorFlag++;
  }
  *outStream << "    Optimization Method:    " << schemaMultiple->optimization_method() << std::endl;
  if(schemaMultiple->optimization_method()!=DICe::GRADIENT_BASED_THEN_SIMPLEX){
    *outStream << "Error, wrong optimization method" << std::endl;
    errorFlag++;
  }
  *outStream << "    Initialization Method:  " << schemaMultiple->initialization_method() << std::endl;
  if(schemaMultiple->initialization_method()!=DICe::USE_NEIGHBOR_VALUES){
    *outStream << "Error, wrong initialization method" << std::endl;
    errorFlag++;
  }
  *outStream << "    Interpolation Method:   " << schemaMultiple->interpolation_method() << std::endl;
  if(schemaMultiple->interpolation_method()!=DICe::KEYS_FOURTH){
    *outStream << "Error, wrong interpolation routine" << std::endl;
    errorFlag++;
  }
  *outStream << "    Translation Enabled:    " << schemaMultiple->translation_enabled() << std::endl;
  if(!schemaMultiple->translation_enabled()){
    *outStream << "Error, translation should be enabled" << std::endl;
    errorFlag++;
  }
  *outStream << "    Rotation Enabled:       " << schemaMultiple->rotation_enabled() << std::endl;
  if(schemaMultiple->rotation_enabled()){
    *outStream << "Error, rotation should not be enabled" << std::endl;
    errorFlag++;
  }
  *outStream << "    Normal Strain Enabled:  " << schemaMultiple->normal_strain_enabled() << std::endl;
  if(schemaMultiple->normal_strain_enabled()){
    *outStream << "Error, normal strain should not be enabled" << std::endl;
    errorFlag++;
  }
  *outStream << "    Shear Strain Enabled:   " << schemaMultiple->shear_strain_enabled() << std::endl;
  if(schemaMultiple->shear_strain_enabled()){
    *outStream << "Error, shear strain should not be enabled" << std::endl;
    errorFlag++;
  }
  schemaMultiple->print_fields();

  // sum the solutions to ensure its roughly 0.5 ux and 0.5 uy
  bool error_x_occurred = false;
  bool error_y_occurred = false;
  bool error_flag = false;
  for(int_t i=0;i<schemaMultiple->local_num_subsets();++i){
    const work_t disp_x = schemaMultiple->local_field_value(i,DICe::field_enums::SUBSET_DISPLACEMENT_X_FS);
    const work_t disp_y = schemaMultiple->local_field_value(i,DICe::field_enums::SUBSET_DISPLACEMENT_Y_FS);
    const work_t flag = schemaMultiple->local_field_value(i,DICe::field_enums::STATUS_FLAG_FS);
    if(disp_x<0.1||disp_x>0.8) error_x_occurred = true;
    if(disp_y<0.1||disp_y>0.8) error_y_occurred = true;
    if(flag!=DICe::INITIALIZE_USING_PREVIOUS_FRAME_SUCCESSFUL&&flag!=DICe::INITIALIZE_USING_NEIGHBOR_VALUE_SUCCESSFUL) error_flag = true;
  }
  if(error_x_occurred){
    *outStream << "---> POSSIBLE ERROR ABOVE! Error in UX is too high." << std::endl;
    errorFlag++;
  }
  if(error_y_occurred){
    *outStream << "---> POSSIBLE ERROR ABOVE! Error in UY is too high." << std::endl;
    errorFlag++;
  }
  if(error_flag){
    *outStream << "---> POSSIBLE ERROR ABOVE! Error in output flags." << std::endl;
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

