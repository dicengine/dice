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

#include <DICe_MeshEnums.h>

#include <Teuchos_RCP.hpp>

#include <map>

namespace DICe{
namespace mesh{

static bool mesh_string_maps_created = false;
static std::map<Component,std::string> component_string;
static std::map<std::string,Component> string_component;
static std::map<Base_Element_Type,std::string> base_element_type_string;
static std::map<std::string,Base_Element_Type> string_base_element_type;
static std::map<Physics_Term,std::string> physics_term_string;
static std::map<std::string,Physics_Term> string_physics_term;

DICE_LIB_DLL_EXPORT
void create_mesh_string_maps();
DICE_LIB_DLL_EXPORT
void create_mesh_string_maps()
{
  if (mesh_string_maps_created)
  {
    return;
  }
  mesh_string_maps_created = true;

  component_string[NO_COMP]                                                          = "NO_COMP";
  component_string[X_COMP]                                                           = "X";
  component_string[Y_COMP]                                                           = "Y";
  component_string[Z_COMP]                                                           = "Z";

  base_element_type_string[NO_SUCH_ELEMENT_TYPE]                                     = "NULL";
  base_element_type_string[HEX8]                                                     = "HEX8";
  base_element_type_string[TETRA4]                                                   = "TETRA4";
  base_element_type_string[TETRA]                                                    = "TETRA";
  base_element_type_string[TRI3]                                                     = "TRI3";
  base_element_type_string[TRI6]                                                     = "TRI6";
  base_element_type_string[CVFEM_TRI]                                                = "CVFEM_TRI";
  base_element_type_string[CVFEM_TETRA]                                              = "CVFEM_TETRA";
  base_element_type_string[QUAD4]                                                    = "QUAD4";
  base_element_type_string[SPHERE]                                                   = "SPHERE";
  base_element_type_string[PYRAMID5]                                                 = "PYRAMID5";
  base_element_type_string[MESHLESS]                                                 = "MESHLESS";

  physics_term_string[NO_SUCH_PHYSICS_TERM]                                          = "NO_SUCH_PHYSICS_TERM";
  physics_term_string[MASS]                                                          = "MASS";
  physics_term_string[DIFF]                                                          = "DIFF";
  physics_term_string[ADV]                                                           = "ADV";
  physics_term_string[SRC]                                                           = "SRC";

  for (std::map<Component,std::string>::iterator pos = component_string.begin(); pos != component_string.end(); ++pos){
    string_component[pos->second] = pos->first;
  }
  for (std::map<Base_Element_Type,std::string>::iterator pos = base_element_type_string.begin(); pos != base_element_type_string.end(); ++pos){
    string_base_element_type[pos->second] = pos->first;
  }
  for (std::map<Physics_Term,std::string>::iterator pos = physics_term_string.begin(); pos != physics_term_string.end(); ++pos){
    string_physics_term[pos->second] = pos->first;
  }
}

DICE_LIB_DLL_EXPORT
std::string tostring(const Physics_Term & physics_term){
  create_mesh_string_maps();
  std::map<Physics_Term,std::string>::iterator pos=physics_term_string.find(physics_term);
    if (physics_term_string.count(physics_term)==0){
      std::stringstream oss;
      oss << "Unknown physics term: " << physics_term << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
    }
    return pos->second;
}

DICE_LIB_DLL_EXPORT
std::string tostring(const Component & comp){
  create_mesh_string_maps();
  std::map<Component,std::string>::iterator pos=component_string.find(comp);
    if (component_string.count(comp)==0){
      std::stringstream oss;
      oss << "Unknown component: " << comp << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
    }
    return pos->second;
}
DICE_LIB_DLL_EXPORT
std::string tostring(const Base_Element_Type & base_element_type){
  create_mesh_string_maps();
  std::map<Base_Element_Type,std::string>::iterator pos=base_element_type_string.find(base_element_type);
    if (base_element_type_string.count(base_element_type)==0){
      std::stringstream oss;
      oss << "Unknown base element type: " << base_element_type << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
    }
    return pos->second;
}
DICE_LIB_DLL_EXPORT
int_t toindex(const Component comp){
  if(comp == NO_COMP || comp == X_COMP)
    return 0;
  else if(comp == Y_COMP)
    return 1;
  else if(comp == Z_COMP)
    return 2;
  else{
    std::stringstream oss;
    oss << " toindex(): Unknown component " << tostring(comp) << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  return -1;
}
DICE_LIB_DLL_EXPORT
Physics_Term string_to_physics_term(const std::string & input_string){
  create_mesh_string_maps();
  std::map<std::string,Physics_Term,std::string>::iterator pos=string_physics_term.find(input_string);
  if (string_physics_term.count(input_string)==0){
    std::stringstream oss;
    oss << "Unknown physics term: " << input_string << std::endl;
    oss << "Valid options are: " << std::endl;
    for (std::map<std::string,Physics_Term,std::string >::iterator it = string_physics_term.begin(); it != string_physics_term.end(); ++it)
    {
      oss << "  " << it->first << std::endl;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  return pos->second;
}

DICE_LIB_DLL_EXPORT
Component string_to_component(const std::string & input_string){
  create_mesh_string_maps();
  std::map<std::string,Component,std::string>::iterator pos=string_component.find(input_string);
  if (string_component.count(input_string)==0){
    std::stringstream oss;
    oss << "Unknown component: " << input_string << std::endl;
    oss << "Valid options are: " << std::endl;
    for (std::map<std::string,Component,std::string>::iterator it = string_component.begin(); it != string_component.end(); ++it)
    {
      oss << "  " << it->first << std::endl;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  return pos->second;
}

DICE_LIB_DLL_EXPORT
std::string index_to_component(const int_t index){
  if (index==0)
    return "X";
  else if (index==1)
    return "Y";
  else if (index==2)
    return "Z";
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unknown index");
}

DICE_LIB_DLL_EXPORT
std::string index_to_component_string(const int_t index){
  if (index==0)
    return "_X";
  else if (index==1)
    return "_Y";
  else if (index==2)
    return "_Z";
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unknown index");
}

DICE_LIB_DLL_EXPORT
Base_Element_Type string_to_base_element_type(const std::string & input_string){
  create_mesh_string_maps();
  std::map<std::string,Base_Element_Type,std::string>::iterator pos=string_base_element_type.find(input_string);
  if (string_base_element_type.count(input_string)==0){
    std::stringstream oss;
    oss << "Unknown base element type: " << input_string << std::endl;
    oss << "Valid options are: " << std::endl;
    for (std::map<std::string,Base_Element_Type,std::string >::iterator it = string_base_element_type.begin(); it != string_base_element_type.end(); ++it)
    {
      oss << "  " << it->first << std::endl;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  return pos->second;
}

} //namespace mesh
} // namespace DICe

