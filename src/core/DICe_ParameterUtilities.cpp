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
// Questions? Contact lead developer:
//              Dan Turner   (danielzturner@gmail.com)
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
  return fieldNameStrings[in];
}
DICE_LIB_DLL_EXPORT
const std::string to_string(Correlation_Routine in){
  assert(in < MAX_CORRELATION_ROUTINE);
  return correlationRoutineStrings[in];
}
DICE_LIB_DLL_EXPORT
const std::string to_string(Projection_Method in){
  assert(in < MAX_PROJECTION_METHOD);
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
const Field_Name string_to_field_name(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_FIELD_NAME;++i){
    if(fieldNameStrings[i]==in) return static_cast<Field_Name>(i);
  }
  std::cout << "Error: Field_Name " << in << " does not exist." << std::endl;
  return NO_SUCH_FIELD_NAME; // prevent no return errors
}
DICE_LIB_DLL_EXPORT
const Correlation_Routine string_to_correlation_routine(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_CORRELATION_ROUTINE;++i){
    if(correlationRoutineStrings[i]==in) return static_cast<Correlation_Routine>(i);
  }
  std::cout << "Error: Correlation_Routine " << in << " does not exist." << std::endl;
  return NO_SUCH_CORRELATION_ROUTINE; // prevent no return errors
}
DICE_LIB_DLL_EXPORT
const Projection_Method string_to_projection_method(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_PROJECTION_METHOD;++i){
    if(projectionMethodStrings[i]==in) return static_cast<Projection_Method>(i);
  }
  std::cout << "Error: Projection_Method " << in << " does not exist." << std::endl;
  return NO_SUCH_PROJECTION_METHOD; // prevent no return errors
}
DICE_LIB_DLL_EXPORT
const Initialization_Method string_to_initialization_method(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_INITIALIZATION_METHOD;++i){
    if(initializationMethodStrings[i]==in) return static_cast<Initialization_Method>(i);
  }
  std::cout << "Error: Initialization_Method " << in << " does not exist." << std::endl;
  return NO_SUCH_INITIALIZATION_METHOD; // prevent no return errors
}
DICE_LIB_DLL_EXPORT
const Optimization_Method string_to_optimization_method(std::string & in){
  // convert the string to uppercase
  stringToUpper(in);
  for(int_t i=0;i<MAX_OPTIMIZATION_METHOD;++i){
    if(optimizationMethodStrings[i]==in) return static_cast<Optimization_Method>(i);
  }
  std::cout << "Error: Optimization_Method " << in << " does not exist." << std::endl;
  return NO_SUCH_OPTIMIZATION_METHOD; // prevent no return errors
}
DICE_LIB_DLL_EXPORT
const Interpolation_Method string_to_interpolation_method(std::string & in){
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
const bool is_string_param(const std::string & in){
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
  defaultParams->set(DICe::robust_solver_tolerance,(scalar_t)1.0E-6);
  defaultParams->set(DICe::fast_solver_tolerance,(scalar_t)1.0E-4);
  defaultParams->set(DICe::robust_delta_disp,(scalar_t)1.0);  // simplex method initial shape is based on these
  defaultParams->set(DICe::robust_delta_theta,(scalar_t)0.1); //
  defaultParams->set(DICe::interpolation_method,DICe::BILINEAR);
  defaultParams->set(DICe::optimization_method,DICe::GRADIENT_BASED_THEN_SIMPLEX);
  defaultParams->set(DICe::initialization_method,DICe::USE_FIELD_VALUES);
  defaultParams->set(DICe::projection_method,DICe::DISPLACEMENT_BASED);
  defaultParams->set(DICe::disp_jump_tol,(scalar_t)5.0);
  defaultParams->set(DICe::theta_jump_tol,(scalar_t)0.1);
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
  defaultParams->set(DICe::obstruction_skin_factor,(scalar_t)1.0);
  defaultParams->set(DICe::search_window_size_factor,(scalar_t)1.0);
  defaultParams->set(DICe::search_window_skip,0);
  defaultParams->set(DICe::update_obstructed_pixels_each_iteration,false);
  defaultParams->set(DICe::normalize_gamma_with_active_pixels,false);
  defaultParams->set(DICe::use_objective_regularization,false);
  defaultParams->set(DICe::pixel_integration_order,1);
  defaultParams->set(DICe::skip_solve_gamma_threshold,(scalar_t)1.0E-10);
}

DICE_LIB_DLL_EXPORT void dice_default_params(Teuchos::ParameterList *  defaultParams){
  defaultParams->set(DICe::correlation_routine,DICe::GENERIC_ROUTINE);
  defaultParams->set(DICe::max_evolution_iterations,10);
  defaultParams->set(DICe::max_solver_iterations_fast,250);
  defaultParams->set(DICe::max_solver_iterations_robust,1000);
  defaultParams->set(DICe::robust_solver_tolerance,(scalar_t)1.0E-6);
  defaultParams->set(DICe::fast_solver_tolerance,(scalar_t)1.0E-4);
  defaultParams->set(DICe::robust_delta_disp,(scalar_t)1.0);  // simplex method initial shape is based on these
  defaultParams->set(DICe::robust_delta_theta,(scalar_t)0.1); //
  defaultParams->set(DICe::interpolation_method,DICe::KEYS_FOURTH);
  defaultParams->set(DICe::optimization_method,DICe::GRADIENT_BASED_THEN_SIMPLEX);
  defaultParams->set(DICe::initialization_method,DICe::USE_FIELD_VALUES);
  defaultParams->set(DICe::projection_method,DICe::DISPLACEMENT_BASED);
  defaultParams->set(DICe::disp_jump_tol,(scalar_t)5.0);
  defaultParams->set(DICe::theta_jump_tol,(scalar_t)0.1);
  defaultParams->set(DICe::enable_translation,true);
  defaultParams->set(DICe::enable_rotation,false);
  defaultParams->set(DICe::enable_normal_strain,false);
  defaultParams->set(DICe::enable_shear_strain,false);
  defaultParams->set(DICe::search_window_size_factor,(scalar_t)1.0);
  defaultParams->set(DICe::search_window_skip,0);
  defaultParams->set(DICe::output_deformed_subset_images,false);
  defaultParams->set(DICe::output_deformed_subset_intensity_images,false);
  defaultParams->set(DICe::output_evolved_subset_images,false);
  defaultParams->set(DICe::use_subset_evolution,false);
  defaultParams->set(DICe::output_delimiter," ");
  defaultParams->set(DICe::omit_output_row_id,false);
  defaultParams->set(DICe::obstruction_buffer_size,3);
  defaultParams->set(DICe::obstruction_skin_factor,(scalar_t)1.0);
  defaultParams->set(DICe::update_obstructed_pixels_each_iteration,false);
  defaultParams->set(DICe::normalize_gamma_with_active_pixels,false);
  defaultParams->set(DICe::use_objective_regularization,false);
  defaultParams->set(DICe::pixel_integration_order,1);
  defaultParams->set(DICe::skip_solve_gamma_threshold,(scalar_t)1.0E-10);
}

}// End DICe Namespace
