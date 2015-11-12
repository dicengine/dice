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

#include <DICe_ParameterUtilities.h>
#include <DICe.h>
#include <cassert>
#include <iostream>


namespace DICe {

DICE_LIB_DLL_EXPORT
void stringToUpper(std::string &s){
  std::transform(s.begin(), s.end(),s.begin(), ::toupper);
}

DICE_LIB_DLL_EXPORT
void stringToLower(std::string &s){
  std::transform(s.begin(), s.end(),s.begin(), ::tolower);
}

DICE_LIB_DLL_EXPORT
const std::string to_string(Field_Name in){
  assert(in < MAX_FIELD_NAME);
  static const char * fieldNameStrings[] = {
    "DISPLACEMENT_X",
    "DISPLACEMENT_Y",
    "DISPLACEMENT_Z",
    "ROTATION_X",
    "ROTATION_Y",
    "ROTATION_Z",
    "NORMAL_STRAIN_X",
    "NORMAL_STRAIN_Y",
    "NORMAL_STRAIN_Z",
    "SHEAR_STRAIN_XY",
    "SHEAR_STRAIN_YZ",
    "SHEAR_STRAIN_XZ",
    "COORDINATE_X",
    "COORDINATE_Y",
    "COORDINATE_Z",
    "VAR_X",
    "VAR_Y",
    "VAR_Z",
    "SIGMA",
    "GAMMA",
    "MATCH",
    "ITERATIONS",
    "STATUS_FLAG",
    "NEIGHBOR_ID",
    "CONDITION_NUMBER"
  };
  return fieldNameStrings[in];
}
DICE_LIB_DLL_EXPORT
const std::string to_string(Status_Flag in){
  assert(in < MAX_STATUS_FLAG);
  const static char * statusFlagStrings[] = {
    "CORRELATION_SUCCESSFUL",
    "INITIALIZE_USING_PREVIOUS_STEP_SUCCESSFUL",
    "INITIALIZE_USING_CONNECTED_SUBSET_VALUE_SUCCESSFUL",
    "INITIALIZE_USING_NEIGHBOR_VALUE_SUCCESSFUL",
    "INITIALIZE_SUCCESSFUL",
    "INITIALIZE_FAILED",
    "SEARCH_SUCCESSFUL",
    "SEARCH_FAILED",
    "CORRELATION_FAILED",
    "SUBSET_CONSTRUCTION_FAILED",
    "LINEAR_SOLVE_FAILED",
    "MAX_ITERATIONS_REACHED",
    "INITIALIZE_FAILED_BY_EXCEPTION",
    "SEARCH_FAILED_BY_EXCEPTION",
    "CORRELATION_FAILED_BY_EXCEPTION",
    "CORRELATION"
    "_BY_AVERAGING_CONNECTED_VALUES",
    "JUMP_TOLERANCE_EXCEEDED",
    "ZERO_HESSIAN_DETERMINANT",
    "SEARCH_USING_PREVIOUS_STEP_SUCCESSFUL",
    "LINEARIZED_GAMMA_OUT_OF_BOUNDS",
    "NAN_IN_HESSIAN_OR_RESIDUAL",
    "HESSIAN_SINGULAR",
    "SKIPPED_FRAME_DUE_TO_HIGH_GAMMA",
    "RESET_REF_SUBSET_DUE_TO_HIGH_GAMMA",
    "MAX_GLOBAL_ITERATIONS_REACHED_IN_EVOLUTION_LOOP",
    "FAILURE_DUE_TO_TOO_MANY_RESTARTS",
    "FAILURE_DUE_TO_DEVIATION_FROM_PATH",
    "FRAME_SKIPPED",
    "FRAME_SKIPPED_DUE_TO_NO_MOTION"
  };
  return statusFlagStrings[in];
}
DICE_LIB_DLL_EXPORT
const std::string to_string(Correlation_Routine in){
  assert(in < MAX_CORRELATION_ROUTINE);
  return correlationRoutineStrings[in];
}
DICE_LIB_DLL_EXPORT
const std::string to_string(Projection_Method in){
  assert(in < MAX_PROJECTION_METHOD);
  const static char * projectionMethodStrings[] = {
    "DISPLACEMENT_BASED",
    "VELOCITY_BASED",
    "MULTISTEP"
  };
  return projectionMethodStrings[in];
}
DICE_LIB_DLL_EXPORT
const std::string to_string(Initialization_Method in){
  assert(in < MAX_INITIALIZATION_METHOD);
  return initializationMethodStrings[in];
}
DICE_LIB_DLL_EXPORT
const std::string to_string(Optimization_Method in){
  assert(in < MAX_OPTIMIZATION_METHOD);
  return optimizationMethodStrings[in];
}
DICE_LIB_DLL_EXPORT
const std::string to_string(Interpolation_Method in){
  assert(in < MAX_INTERPOLATION_METHOD);
  return interpolationMethodStrings[in];
}
DICE_LIB_DLL_EXPORT
Field_Name string_to_field_name(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_FIELD_NAME;++i){
    if(to_string(static_cast<Field_Name>(i))==in) return static_cast<Field_Name>(i);
  }
  std::cout << "Error: Field_Name " << in << " does not exist." << std::endl;
  return NO_SUCH_FIELD_NAME; // prevent no return errors
}
DICE_LIB_DLL_EXPORT
Correlation_Routine string_to_correlation_routine(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_CORRELATION_ROUTINE;++i){
    if(correlationRoutineStrings[i]==in) return static_cast<Correlation_Routine>(i);
  }
  std::cout << "Error: Correlation_Routine " << in << " does not exist." << std::endl;
  return NO_SUCH_CORRELATION_ROUTINE; // prevent no return errors
}
DICE_LIB_DLL_EXPORT
Projection_Method string_to_projection_method(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_PROJECTION_METHOD;++i){
    if(projectionMethodStrings[i]==in) return static_cast<Projection_Method>(i);
  }
  std::cout << "Error: Projection_Method " << in << " does not exist." << std::endl;
  return NO_SUCH_PROJECTION_METHOD; // prevent no return errors
}
DICE_LIB_DLL_EXPORT
Initialization_Method string_to_initialization_method(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_INITIALIZATION_METHOD;++i){
    if(initializationMethodStrings[i]==in) return static_cast<Initialization_Method>(i);
  }
  std::cout << "Error: Initialization_Method " << in << " does not exist." << std::endl;
  return NO_SUCH_INITIALIZATION_METHOD; // prevent no return errors
}
DICE_LIB_DLL_EXPORT
Optimization_Method string_to_optimization_method(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_OPTIMIZATION_METHOD;++i){
    if(optimizationMethodStrings[i]==in) return static_cast<Optimization_Method>(i);
  }
  std::cout << "Error: Optimization_Method " << in << " does not exist." << std::endl;
  return NO_SUCH_OPTIMIZATION_METHOD; // prevent no return errors
}
DICE_LIB_DLL_EXPORT
Interpolation_Method string_to_interpolation_method(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_INTERPOLATION_METHOD;++i){
    if(interpolationMethodStrings[i]==in) return static_cast<Interpolation_Method>(i);
  }
  std::cout << "Error: Interpolation_Method " << in << " does not exist." << std::endl;
  return NO_SUCH_INTERPOLATION_METHOD; // prevent no return errors
}

/// Determine if the parameter is a string parameter
DICE_LIB_DLL_EXPORT
bool is_string_param(const std::string & in){
  // change the string to lower case (string param names are lower)
  std::string lower = in;
  stringToLower(lower);
  for(int_t i=0;i<num_valid_correlation_params;++i){
    if(lower==valid_correlation_params[i].name_ && valid_correlation_params[i].type_==STRING_PARAM)
      return true;
  }
  return false;
}


DICE_LIB_DLL_EXPORT void sl_default_params(Teuchos::ParameterList *  defaultParams){
  defaultParams->set(DICe::correlation_routine,DICe::SL_ROUTINE);
  defaultParams->set(DICe::max_evolution_iterations,10);
  defaultParams->set(DICe::max_solver_iterations_fast,250);
  defaultParams->set(DICe::max_solver_iterations_robust,1000);
  defaultParams->set(DICe::robust_solver_tolerance,1.0E-6);
  defaultParams->set(DICe::fast_solver_tolerance,1.0E-4);
  defaultParams->set(DICe::robust_delta_disp,1.0);  // simplex method initial shape is based on these
  defaultParams->set(DICe::robust_delta_theta,0.1); //
  defaultParams->set(DICe::interpolation_method,DICe::BILINEAR);
  defaultParams->set(DICe::optimization_method,DICe::GRADIENT_BASED_THEN_SIMPLEX);
  defaultParams->set(DICe::initialization_method,DICe::USE_FIELD_VALUES);
  defaultParams->set(DICe::projection_method,DICe::DISPLACEMENT_BASED);
  defaultParams->set(DICe::disp_jump_tol,5.0);
  defaultParams->set(DICe::theta_jump_tol,0.1);
  defaultParams->set(DICe::enable_translation,true);
  defaultParams->set(DICe::enable_rotation,true);
  defaultParams->set(DICe::enable_normal_strain,false);
  defaultParams->set(DICe::enable_shear_strain,false);
  defaultParams->set(DICe::output_deformed_subset_images,false);
  defaultParams->set(DICe::output_deformed_subset_intensity_images,false);
  defaultParams->set(DICe::output_evolved_subset_images,false);
  defaultParams->set(DICe::use_subset_evolution,false);
  defaultParams->set(DICe::output_delimiter," ");
  defaultParams->set(DICe::omit_output_row_id,false);
  defaultParams->set(DICe::obstruction_buffer_size,3);
  defaultParams->set(DICe::obstruction_skin_factor,1.0);
  defaultParams->set(DICe::search_window_size_factor,1.0);
  defaultParams->set(DICe::search_window_skip,0);
  defaultParams->set(DICe::update_obstructed_pixels_each_iteration,false);
  defaultParams->set(DICe::normalize_gamma_with_active_pixels,false);
  defaultParams->set(DICe::use_objective_regularization,false);
  defaultParams->set(DICe::objective_regularization_factor,1000.0);
  defaultParams->set(DICe::pixel_integration_order,1);
  defaultParams->set(DICe::skip_solve_gamma_threshold,1.0E-10);
  defaultParams->set(DICe::rotate_ref_image_90,false);
  defaultParams->set(DICe::rotate_def_image_90,false);
  defaultParams->set(DICe::rotate_ref_image_180,false);
  defaultParams->set(DICe::rotate_def_image_180,false);
  defaultParams->set(DICe::rotate_ref_image_270,false);
  defaultParams->set(DICe::rotate_def_image_270,false);
}

DICE_LIB_DLL_EXPORT void dice_default_params(Teuchos::ParameterList *  defaultParams){
  defaultParams->set(DICe::correlation_routine,DICe::GENERIC_ROUTINE);
  defaultParams->set(DICe::max_evolution_iterations,10);
  defaultParams->set(DICe::max_solver_iterations_fast,250);
  defaultParams->set(DICe::max_solver_iterations_robust,1000);
  defaultParams->set(DICe::robust_solver_tolerance,1.0E-6);
  defaultParams->set(DICe::fast_solver_tolerance,1.0E-4);
  defaultParams->set(DICe::robust_delta_disp,1.0);  // simplex method initial shape is based on these
  defaultParams->set(DICe::robust_delta_theta,0.1); //
  defaultParams->set(DICe::interpolation_method,DICe::KEYS_FOURTH);
  defaultParams->set(DICe::optimization_method,DICe::GRADIENT_BASED_THEN_SIMPLEX);
  defaultParams->set(DICe::initialization_method,DICe::USE_FIELD_VALUES);
  defaultParams->set(DICe::projection_method,DICe::DISPLACEMENT_BASED);
  defaultParams->set(DICe::disp_jump_tol,5.0);
  defaultParams->set(DICe::theta_jump_tol,0.1);
  defaultParams->set(DICe::enable_translation,true);
  defaultParams->set(DICe::enable_rotation,false);
  defaultParams->set(DICe::enable_normal_strain,false);
  defaultParams->set(DICe::enable_shear_strain,false);
  defaultParams->set(DICe::search_window_size_factor,1.0);
  defaultParams->set(DICe::search_window_skip,0);
  defaultParams->set(DICe::output_deformed_subset_images,false);
  defaultParams->set(DICe::output_deformed_subset_intensity_images,false);
  defaultParams->set(DICe::output_evolved_subset_images,false);
  defaultParams->set(DICe::use_subset_evolution,false);
  defaultParams->set(DICe::output_delimiter," ");
  defaultParams->set(DICe::omit_output_row_id,false);
  defaultParams->set(DICe::obstruction_buffer_size,3);
  defaultParams->set(DICe::obstruction_skin_factor,1.0);
  defaultParams->set(DICe::update_obstructed_pixels_each_iteration,false);
  defaultParams->set(DICe::normalize_gamma_with_active_pixels,false);
  defaultParams->set(DICe::use_objective_regularization,false);
  defaultParams->set(DICe::objective_regularization_factor,1000.0);
  defaultParams->set(DICe::pixel_integration_order,1);
  defaultParams->set(DICe::skip_solve_gamma_threshold,1.0E-10);
  defaultParams->set(DICe::rotate_ref_image_90,false);
  defaultParams->set(DICe::rotate_def_image_90,false);
  defaultParams->set(DICe::rotate_ref_image_180,false);
  defaultParams->set(DICe::rotate_def_image_180,false);
  defaultParams->set(DICe::rotate_ref_image_270,false);
  defaultParams->set(DICe::rotate_def_image_270,false);
}

}// End DICe Namespace
