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

#ifndef DICE_PARAMETER_UTILITIES_H
#define DICE_PARAMETER_UTILITIES_H

#include <DICe.h>

#include <Teuchos_ParameterList.hpp>

#include <algorithm>
#include <string>

namespace DICe{

/// Convert a string to all upper case
DICE_LIB_DLL_EXPORT
void stringToUpper(std::string &s);

/// Convert a string to all lower case
DICE_LIB_DLL_EXPORT
void stringToLower(std::string &s);

/// Convert a DICe::Field_Name to string
DICE_LIB_DLL_EXPORT
const std::string to_string(Field_Name in);

/// Convert a DICe::Correlation_Routine to string
DICE_LIB_DLL_EXPORT
const std::string to_string(Correlation_Routine in);

/// Convert a DICe::Projection_Method to string
DICE_LIB_DLL_EXPORT
const std::string to_string(Projection_Method in);

/// Convert a DICe::Initialization_Method to string
DICE_LIB_DLL_EXPORT
const std::string to_string(Initialization_Method in);

/// Convert a DICe::Optimization_Method to string
DICE_LIB_DLL_EXPORT
const std::string to_string(Optimization_Method in);

/// Convert a DICe::Interpolation_Method to string
DICE_LIB_DLL_EXPORT
const std::string to_string(Interpolation_Method in);

/// Convert a string to a DICe::Field_Name
DICE_LIB_DLL_EXPORT
const Field_Name string_to_field_name(std::string & in);

/// Convert a string to a DICe::Correlation_Routine
DICE_LIB_DLL_EXPORT
const Correlation_Routine string_to_correlation_routine(std::string & in);

/// Convert a string to a DICe::Projection_Method
DICE_LIB_DLL_EXPORT
const Projection_Method string_to_projection_method(std::string & in);

/// Convert a string to a DICe::Initialization_Method
DICE_LIB_DLL_EXPORT
const Initialization_Method string_to_initialization_method(std::string & in);

/// Convert a string to a DICe::Interpolation_Method
DICE_LIB_DLL_EXPORT
const Interpolation_Method string_to_interpolation_method(std::string & in);

/// Convert a string to a DICe::Optimization_Method
DICE_LIB_DLL_EXPORT
const Optimization_Method string_to_optimization_method(std::string & in);

/// Determine if the parameter is a string parameter
DICE_LIB_DLL_EXPORT
const bool is_string_param(const std::string & in);

/// \brief Returns a pointer to a ParameterList with default params specified
/// \param defaultParams Pointer to the ParameterList to be returned
///
/// A free function like this is used to prevent having to re-code the defaults everywhere they are used.
DICE_LIB_DLL_EXPORT
void sl_default_params(Teuchos::ParameterList * defaultParams);

/// \brief Sets the default correlation parameters
/// \param defaultParams returned ParameterList with the default params set
DICE_LIB_DLL_EXPORT
void dice_default_params(Teuchos::ParameterList *  defaultParams);

}// DICe namespace

#endif
