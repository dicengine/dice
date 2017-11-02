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

#ifndef DICE_MESHENUMS_H_
#define DICE_MESHENUMS_H_

#include <DICe.h>

namespace DICe{
namespace mesh{

/// Status of the current step
enum
DICE_LIB_DLL_EXPORT
Step_Status
{
  STEP_CONVERGED = 0,
  STEP_ITERATING,
  STEP_FAILED
};

/// vector components
enum
DICE_LIB_DLL_EXPORT
Component
{
  NO_COMP=0,
  X_COMP,
  Y_COMP,
  Z_COMP
};

/// converts a component to a string
/// \param comp the component
DICE_LIB_DLL_EXPORT
std::string tostring(const Component & comp);
/// converts a string to a component
/// \param input_string the input string
DICE_LIB_DLL_EXPORT
Component string_to_component(const std::string & input_string);
/// converts an index to a component string
/// \param index the index
DICE_LIB_DLL_EXPORT
std::string index_to_component_string(const int_t index);
/// converts and index to a component
/// \param index the index
DICE_LIB_DLL_EXPORT
std::string index_to_component(const int_t index);
/// The base element type for a block
enum
DICE_LIB_DLL_EXPORT
Base_Element_Type
{
  NO_SUCH_ELEMENT_TYPE=0,
  HEX8,
  QUAD4,
  TETRA4,
  TETRA,
  TRI3,
  TRI6,
  CVFEM_TRI,
  CVFEM_TETRA,
  SPHERE,
  PYRAMID5,
  MESHLESS
};

/// converts a base element type to a string
/// \param base_element_type the element type
DICE_LIB_DLL_EXPORT
std::string tostring(const Base_Element_Type & base_element_type);
/// converts a string to a base element type
/// \param input_string the input string
DICE_LIB_DLL_EXPORT
Base_Element_Type string_to_base_element_type(const std::string & input_string);
/// converts a component to an index
/// \param comp the component
DICE_LIB_DLL_EXPORT
int_t toindex(const Component comp);

/// Terms that are included in the physics
enum
DICE_LIB_DLL_EXPORT
Physics_Term
{
  NO_SUCH_PHYSICS_TERM=0,
  MASS,
  DIFF,
  ADV,
  SRC
};

/// converts a physics term to a string
/// \param physics_term the term to convert
DICE_LIB_DLL_EXPORT
std::string tostring(const Physics_Term & physics_term);
/// converts a string to a physics term
/// \param input_string the input string
DICE_LIB_DLL_EXPORT
Physics_Term string_to_physics_term(const std::string & input_string);

} //namespace mesh
} //namespace DICe

#endif /* DICE_MESHENUMS_H_ */
