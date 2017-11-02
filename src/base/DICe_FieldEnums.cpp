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

#include <DICe_FieldEnums.h>

#include <Teuchos_RCP.hpp>

#include <map>

namespace DICe{

static bool string_maps_created = false;
static std::map<field_enums::Entity_Rank,std::string> entity_rank_string;
static std::map<std::string,field_enums::Entity_Rank> string_entity_rank;
static std::map<field_enums::Field_Type,std::string> field_type_string;
static std::map<std::string,field_enums::Field_Type> string_field_type;
static std::map<field_enums::Field_Name,std::string> field_name_string;
static std::map<std::string,field_enums::Field_Name> string_field_name;

DICE_LIB_DLL_EXPORT
void create_string_maps();
DICE_LIB_DLL_EXPORT
void create_string_maps()
{
  if (string_maps_created)
  {
    return;
  }
  string_maps_created = true;

  entity_rank_string[field_enums::NO_SUCH_ENTITY_RANK]                               = "NO_SUCH_ENTITY_RANK";
  entity_rank_string[field_enums::NODE_RANK]                                         = "NODE_RANK";
  entity_rank_string[field_enums::ELEMENT_RANK]                                      = "ELEMENT_RANK";
  entity_rank_string[field_enums::SUBELEMENT_RANK]                                   = "SUBELEMENT_RANK";
  entity_rank_string[field_enums::FACE_RANK]                                         = "FACE_RANK";
  entity_rank_string[field_enums::EDGE_RANK]                                         = "EDGE_RANK";
  entity_rank_string[field_enums::INTERNAL_FACE_EDGE_RANK]                           = "INTERNAL_FACE_EDGE_RANK";
  entity_rank_string[field_enums::EXTERNAL_FACE_EDGE_RANK]                           = "EXTERNAL_FACE_EDGE_RANK";
  entity_rank_string[field_enums::INTERNAL_CELL_RANK]                                = "INTERNAL_CELL_RANK";
  entity_rank_string[field_enums::BOND_RANK]                                         = "BOND_RANK";

  field_type_string[field_enums::NO_SUCH_FIELD_TYPE]                                 = "NO_SUCH_FIELD_TYPE";
  field_type_string[field_enums::SCALAR_FIELD_TYPE]                                  = "SCALAR_FIELD_TYPE";
  field_type_string[field_enums::VECTOR_FIELD_TYPE]                                  = "VECTOR_FIELD_TYPE";
  field_type_string[field_enums::MIXED_VECTOR_FIELD_TYPE]                            = "MIXED_VECTOR_FIELD_TYPE";

  field_name_string[field_enums::NO_SUCH_FIELD_NAME]                                 = "NO_SUCH_FIELD_NAME";
  field_name_string[field_enums::BLOCK_ID]                                           = "BLOCK_ID";
  field_name_string[field_enums::PROCESSOR_ID]                                       = "PROCESSOR_ID";
  field_name_string[field_enums::MASTER_NODE_ID]                                     = "MASTER_NODE_ID";
  field_name_string[field_enums::INITIAL_COORDINATES]                                = "INITIAL_COORDINATES";
  field_name_string[field_enums::CROSS_CORR_Q]                                       = "CROSS_CORR_Q";
  field_name_string[field_enums::CROSS_CORR_R]                                       = "CROSS_CORR_R";
  field_name_string[field_enums::SUBSET_COORDINATES_X]                               = "COORDINATE_X";  // the string is different from the enum because of legacy field names
  field_name_string[field_enums::SUBSET_COORDINATES_Y]                               = "COORDINATE_Y";  // the string is different from the enum because of legacy field names
  field_name_string[field_enums::STEREO_COORDINATES_X]                               = "STEREO_COORDINATES_X";
  field_name_string[field_enums::STEREO_COORDINATES_Y]                               = "STEREO_COORDINATES_Y";
  field_name_string[field_enums::MODEL_COORDINATES_X]                                = "MODEL_COORDINATES_X";
  field_name_string[field_enums::MODEL_COORDINATES_Y]                                = "MODEL_COORDINATES_Y";
  field_name_string[field_enums::MODEL_COORDINATES_Z]                                = "MODEL_COORDINATES_Z";
  field_name_string[field_enums::INITIAL_CELL_COORDINATES]                           = "INITIAL_CELL_COORDINATES";
  field_name_string[field_enums::INITIAL_CELL_SIZE]                                  = "INITIAL_CELL_SIZE";
  field_name_string[field_enums::INITIAL_SUBELEMENT_SIZE]                            = "INITIAL_SUBELEMENT_SIZE";
  field_name_string[field_enums::INITIAL_WEIGHTED_CELL_SIZE]                         = "INITIAL_WEIGHTED_CELL_SIZE";
  field_name_string[field_enums::INITIAL_CELL_RADIUS]                                = "INITIAL_CELL_RADIUS";
  field_name_string[field_enums::DISPLACEMENT]                                       = "DISPLACEMENT";
  field_name_string[field_enums::PROJECTION_AUG_X]                                   = "PROJECTION_AUG_X";
  field_name_string[field_enums::PROJECTION_AUG_Y]                                   = "PROJECTION_AUG_Y";
  field_name_string[field_enums::SUBSET_DISPLACEMENT_X]                              = "DISPLACEMENT_X"; // the string is different from the enum because of legacy field names
  field_name_string[field_enums::SUBSET_DISPLACEMENT_Y]                              = "DISPLACEMENT_Y"; // the string is different from the enum because of legacy field names
  field_name_string[field_enums::STEREO_DISPLACEMENT_X]                              = "STEREO_DISPLACEMENT_X";
  field_name_string[field_enums::STEREO_DISPLACEMENT_Y]                              = "STEREO_DISPLACEMENT_Y";
  field_name_string[field_enums::MODEL_DISPLACEMENT_X]                               = "MODEL_DISPLACEMENT_X";
  field_name_string[field_enums::MODEL_DISPLACEMENT_Y]                               = "MODEL_DISPLACEMENT_Y";
  field_name_string[field_enums::MODEL_DISPLACEMENT_Z]                               = "MODEL_DISPLACEMENT_Z";
  field_name_string[field_enums::ROTATION_Z]                                         = "ROTATION_Z";
  field_name_string[field_enums::SIGMA]                                              = "SIGMA";
  field_name_string[field_enums::GAMMA]                                              = "GAMMA";
  field_name_string[field_enums::BETA]                                               = "BETA";
  field_name_string[field_enums::OMEGA]                                              = "OMEGA";
  field_name_string[field_enums::NOISE_LEVEL]                                        = "NOISE_LEVEL";
  field_name_string[field_enums::CONTRAST_LEVEL]                                     = "CONTRAST_LEVEL";
  field_name_string[field_enums::ACTIVE_PIXELS]                                      = "ACTIVE_PIXELS";
  field_name_string[field_enums::MATCH]                                              = "MATCH";
  field_name_string[field_enums::ITERATIONS]                                         = "ITERATIONS";
  field_name_string[field_enums::STATUS_FLAG]                                        = "STATUS_FLAG";
  field_name_string[field_enums::NEIGHBOR_ID]                                        = "NEIGHBOR_ID";
  field_name_string[field_enums::CONDITION_NUMBER]                                   = "CONDITION_NUMBER";
  field_name_string[field_enums::SHEAR_STRETCH_XY]                                   = "SHEAR_STRAIN_XY"; // the string is different from the enum because of legacy field names
  field_name_string[field_enums::NORMAL_STRETCH_XX]                                  = "NORMAL_STRAIN_X"; // the string is different from the enum because of legacy field names
  field_name_string[field_enums::NORMAL_STRETCH_YY]                                  = "NORMAL_STRAIN_Y"; // the string is different from the enum because of legacy field names
  field_name_string[field_enums::LAGRANGE_MULTIPLIER]                                = "LAGRANGE_MULTIPLIER";
  field_name_string[field_enums::RESIDUAL]                                           = "RESIDUAL";
  field_name_string[field_enums::LHS]                                                = "LHS";
  field_name_string[field_enums::MIXED_RESIDUAL]                                     = "MIXED_RESIDUAL";
  field_name_string[field_enums::MIXED_LHS]                                          = "MIXED_LHS";
  field_name_string[field_enums::EXACT_SOL_VECTOR]                                   = "EXACT_SOL_VECTOR";
  field_name_string[field_enums::EXACT_LAGRANGE_MULTIPLIER]                          = "EXACT_LAGRANGE_MULTIPLIER";
  field_name_string[field_enums::IMAGE_PHI]                                          = "IMAGE_PHI";
  field_name_string[field_enums::IMAGE_GRAD_PHI]                                     = "IMAGE_GRAD_PHI";
  field_name_string[field_enums::INTERNAL_FACE_EDGE_NORMAL]                          = "INTERNAL_FACE_EDGE_NORMAL";
  field_name_string[field_enums::INTERNAL_FACE_EDGE_COORDINATES]                     = "INTERNAL_FACE_EDGE_COORDINATES";
  field_name_string[field_enums::INTERNAL_FACE_EDGE_SIZE]                            = "INTERNAL_FACE_EDGE_SIZE";
  field_name_string[field_enums::EXTERNAL_FACE_EDGE_NORMAL]                          = "EXTERNAL_FACE_EDGE_NORMAL";
  field_name_string[field_enums::EXTERNAL_FACE_EDGE_COORDINATES]                     = "EXTERNAL_FACE_EDGE_COORDINATES";
  field_name_string[field_enums::EXTERNAL_FACE_EDGE_SIZE]                            = "EXTERNAL_FACE_EDGE_SIZE";
  field_name_string[field_enums::INTERNAL_CELL_COORDINATES]                          = "INTERNAL_CELL_COORDINATES";
  field_name_string[field_enums::INTERNAL_CELL_SIZE]                                 = "INTERNAL_CELL_SIZE";
  field_name_string[field_enums::QUAD_A]                                             = "QUAD_A";
  field_name_string[field_enums::QUAD_B]                                             = "QUAD_B";
  field_name_string[field_enums::QUAD_C]                                             = "QUAD_C";
  field_name_string[field_enums::QUAD_D]                                             = "QUAD_D";
  field_name_string[field_enums::QUAD_E]                                             = "QUAD_E";
  field_name_string[field_enums::QUAD_F]                                             = "QUAD_F";
  field_name_string[field_enums::QUAD_G]                                             = "QUAD_G";
  field_name_string[field_enums::QUAD_H]                                             = "QUAD_H";
  field_name_string[field_enums::QUAD_I]                                             = "QUAD_I";
  field_name_string[field_enums::QUAD_J]                                             = "QUAD_J";
  field_name_string[field_enums::QUAD_K]                                             = "QUAD_K";
  field_name_string[field_enums::QUAD_L]                                             = "QUAD_L";
  field_name_string[field_enums::AFFINE_A]                                           = "AFFINE_A";
  field_name_string[field_enums::AFFINE_B]                                           = "AFFINE_B";
  field_name_string[field_enums::AFFINE_C]                                           = "AFFINE_C";
  field_name_string[field_enums::AFFINE_D]                                           = "AFFINE_D";
  field_name_string[field_enums::AFFINE_E]                                           = "AFFINE_E";
  field_name_string[field_enums::AFFINE_F]                                           = "AFFINE_F";
  field_name_string[field_enums::AFFINE_G]                                           = "AFFINE_G";
  field_name_string[field_enums::AFFINE_H]                                           = "AFFINE_H";
  field_name_string[field_enums::AFFINE_I]                                           = "AFFINE_I";
  field_name_string[field_enums::FIELD_1]                                            = "FIELD_1";
  field_name_string[field_enums::FIELD_2]                                            = "FIELD_2";
  field_name_string[field_enums::FIELD_3]                                            = "FIELD_3";
  field_name_string[field_enums::FIELD_4]                                            = "FIELD_4";
  field_name_string[field_enums::FIELD_5]                                            = "FIELD_5";
  field_name_string[field_enums::FIELD_6]                                            = "FIELD_6";
  field_name_string[field_enums::FIELD_7]                                            = "FIELD_7";
  field_name_string[field_enums::FIELD_8]                                            = "FIELD_8";
  field_name_string[field_enums::FIELD_9]                                            = "FIELD_9";
  field_name_string[field_enums::FIELD_10]                                           = "FIELD_10";
  field_name_string[field_enums::U_X_DERIV]                                          = "U_X_DERIV";
  field_name_string[field_enums::U_Y_DERIV]                                          = "U_Y_DERIV";
  field_name_string[field_enums::V_X_DERIV]                                          = "V_X_DERIV";
  field_name_string[field_enums::V_Y_DERIV]                                          = "V_Y_DERIV";
  field_name_string[field_enums::DU_DX_EXACT]                                        = "DU_DX_EXACT";
  field_name_string[field_enums::DU_DY_EXACT]                                        = "DU_DY_EXACT";
  field_name_string[field_enums::DV_DX_EXACT]                                        = "DV_DX_EXACT";
  field_name_string[field_enums::DV_DY_EXACT]                                        = "DV_DY_EXACT";
  field_name_string[field_enums::STRAIN_CONTRIBS]                                    = "STRAIN_CONTRIBS";
  field_name_string[field_enums::GREEN_LAGRANGE_STRAIN_XX]                           = "GREEN_LAGRANGE_STRAIN_XX";
  field_name_string[field_enums::GREEN_LAGRANGE_STRAIN_YY]                           = "GREEN_LAGRANGE_STRAIN_YY";
  field_name_string[field_enums::GREEN_LAGRANGE_STRAIN_XY]                           = "GREEN_LAGRANGE_STRAIN_XY";
  field_name_string[field_enums::VSG_STRAIN_XX]                                      = "VSG_STRAIN_XX";
  field_name_string[field_enums::VSG_STRAIN_YY]                                      = "VSG_STRAIN_YY";
  field_name_string[field_enums::VSG_STRAIN_XY]                                      = "VSG_STRAIN_XY";
  field_name_string[field_enums::VSG_DUDX]                                           = "VSG_DUDX";
  field_name_string[field_enums::VSG_DUDY]                                           = "VSG_DUDY";
  field_name_string[field_enums::VSG_DVDX]                                           = "VSG_DVDX";
  field_name_string[field_enums::VSG_DVDY]                                           = "VSG_DVDY";
  field_name_string[field_enums::ALTITUDE]                                           = "ALTITUDE";
  field_name_string[field_enums::ALTITUDE_ABOVE_GROUND]                              = "ALTITUDE_ABOVE_GROUND";
  field_name_string[field_enums::GROUND_LEVEL]                                       = "GROUND_LEVEL";
  field_name_string[field_enums::NLVC_STRAIN_XX]                                     = "NLVC_STRAIN_XX";
  field_name_string[field_enums::NLVC_STRAIN_YY]                                     = "NLVC_STRAIN_YY";
  field_name_string[field_enums::NLVC_STRAIN_XY]                                     = "NLVC_STRAIN_XY";
  field_name_string[field_enums::NLVC_DUDX]                                          = "NLVC_DUDX";
  field_name_string[field_enums::NLVC_DUDY]                                          = "NLVC_DUDY";
  field_name_string[field_enums::NLVC_DVDX]                                          = "NLVC_DVDX";
  field_name_string[field_enums::NLVC_DVDY]                                          = "NLVC_DVDY";
  field_name_string[field_enums::EXACT_STRAIN_XX]                                    = "EXACT_STRAIN_XX";
  field_name_string[field_enums::EXACT_STRAIN_YY]                                    = "EXACT_STRAIN_YY";
  field_name_string[field_enums::EXACT_STRAIN_XY]                                    = "EXACT_STRAIN_XY";
  field_name_string[field_enums::ACCUMULATED_DISP]                                   = "ACCUMULATED_DISP";
  field_name_string[field_enums::DISP_ERROR]                                         = "DISP_ERROR";
  field_name_string[field_enums::VSG_STRAIN_XX_ERROR]                                = "VSG_STRAIN_XX_ERROR";
  field_name_string[field_enums::VSG_STRAIN_XY_ERROR]                                = "VSG_STRAIN_XY_ERROR";
  field_name_string[field_enums::VSG_STRAIN_YY_ERROR]                                = "VSG_STRAIN_YY_ERROR";
  field_name_string[field_enums::NLVC_STRAIN_XX_ERROR]                               = "NLVC_STRAIN_XX_ERROR";
  field_name_string[field_enums::NLVC_STRAIN_XY_ERROR]                               = "NLVC_STRAIN_XY_ERROR";
  field_name_string[field_enums::NLVC_STRAIN_YY_ERROR]                               = "NLVC_STRAIN_YY_ERROR";
  field_name_string[field_enums::UNCERTAINTY]                                        = "UNCERTAINTY";
  field_name_string[field_enums::UNCERTAINTY_ANGLE]                                  = "UNCERTAINTY_ANGLE";
  field_name_string[field_enums::STEREO_M_MAX]                                       = "STEREO_M_MAX";

  for (std::map<field_enums::Field_Type,std::string>::iterator pos = field_type_string.begin(); pos != field_type_string.end(); ++pos){
    string_field_type[pos->second] = pos->first;
  }
  for (std::map<field_enums::Field_Name,std::string>::iterator pos = field_name_string.begin(); pos != field_name_string.end(); ++pos){
    string_field_name[pos->second] = pos->first;
  }
  for (std::map<field_enums::Entity_Rank,std::string>::iterator pos = entity_rank_string.begin(); pos != entity_rank_string.end(); ++pos){
    string_entity_rank[pos->second] = pos->first;
  }
}

DICE_LIB_DLL_EXPORT
std::string tostring(const field_enums::Field_Type & field_type)
{
  create_string_maps();
  std::map<field_enums::Field_Type,std::string>::iterator pos=field_type_string.find(field_type);
    if (pos == field_type_string.end())
    {
      std::stringstream oss;
      oss << "Unknown field type: " << field_type << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
    }
    return pos->second;
}
DICE_LIB_DLL_EXPORT
std::string tostring(const field_enums::Field_Name & field_name)
{
  create_string_maps();
  std::map<field_enums::Field_Name,std::string>::iterator pos=field_name_string.find(field_name);
    if (pos == field_name_string.end())
    {
      std::stringstream oss;
      oss << "Unknown field name: " << field_name << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
    }
    return pos->second;
}
DICE_LIB_DLL_EXPORT
std::string tostring(const field_enums::Entity_Rank & entity_rank)
{
  create_string_maps();
  std::map<field_enums::Entity_Rank,std::string>::iterator pos=entity_rank_string.find(entity_rank);
    if (pos == entity_rank_string.end())
    {
      std::stringstream oss;
      oss << "Unknown field type: " << entity_rank << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
    }
    return pos->second;
}


DICE_LIB_DLL_EXPORT
field_enums::Field_Type string_to_field_type(const std::string & input_string)
{
  create_string_maps();
  std::map<std::string,field_enums::Field_Type,std::string>::iterator pos=string_field_type.find(input_string);
  if (pos == string_field_type.end())
  {
    std::stringstream oss;
    oss << "Unknown field type: " << input_string << std::endl;
    oss << "Valid options are: " << std::endl;
    for (std::map<std::string,field_enums::Field_Type,std::string>::iterator it = string_field_type.begin(); it != string_field_type.end(); ++it)
    {
      oss << "  " << it->first << std::endl;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  return pos->second;
}

DICE_LIB_DLL_EXPORT
std::list<std::string>
get_reverse_sorted_field_names()
{
  std::list<std::string> names_list;
  std::map<std::string,field_enums::Field_Name>::iterator string_name_it = string_field_name.begin();
  std::map<std::string,field_enums::Field_Name>::iterator string_name_end = string_field_name.end();
  for(;string_name_it!=string_name_end;++string_name_it)
  {
    names_list.push_back(string_name_it->first);
  }
  names_list.sort();
  names_list.reverse();
  return names_list;
}

DICE_LIB_DLL_EXPORT
void
update_field_names(const field_enums::Field_Name & field_name, const std::string & input_string)
{
  field_name_string[field_name] = input_string;
  std::map<field_enums::Field_Name,std::string>::iterator pos = field_name_string.find(field_name);
  string_field_name[pos->second] = pos->first;
}
int_t num_field_names()
{
  return string_field_name.size();
}

DICE_LIB_DLL_EXPORT
field_enums::Field_Name string_to_field_name(const std::string & input_string)
{
  create_string_maps();
  std::map<std::string,field_enums::Field_Name,std::string >::iterator pos=string_field_name.find(input_string);
  if (pos == string_field_name.end())
  {
    std::stringstream oss;
    oss << "Unknown field name: " << input_string << std::endl;
    oss << "Valid options are: " << std::endl;
    for (std::map<std::string,field_enums::Field_Name,std::string>::iterator it = string_field_name.begin(); it != string_field_name.end(); ++it)
    {
      oss << "  " << it->first << std::endl;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  return pos->second;
}

DICE_LIB_DLL_EXPORT
field_enums::Entity_Rank string_to_entity_rank(const std::string & input_string)
{
  create_string_maps();
  std::map<std::string,field_enums::Entity_Rank,std::string>::iterator pos=string_entity_rank.find(input_string);
  if (pos == string_entity_rank.end())
  {
    std::stringstream oss;
    oss << "Unknown field rank: " << input_string << std::endl;
    oss << "Valid options are: " << std::endl;
    for (std::map<std::string,field_enums::Entity_Rank,std::string>::iterator it = string_entity_rank.begin(); it != string_entity_rank.end(); ++it)
    {
      oss << "  " << it->first << std::endl;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  return pos->second;
}

field_enums::Field_Spec::Field_Spec():
    field_type_(NO_SUCH_FS.get_field_type()),
    name_(NO_SUCH_FS.get_name()),
    rank_(NO_SUCH_FS.get_rank()),
    state_(NO_SUCH_FS.get_state()),
    is_printable_(false),
    is_dof_(false)
{}

field_enums::Field_Spec::Field_Spec(const field_enums::Field_Type field_type,
  const field_enums::Field_Name label,
  const field_enums::Entity_Rank rank,
  const Field_State field_state,
  const bool is_printable,
  const bool is_dof):
    field_type_(field_type),
    name_(label),
    rank_(rank),
    state_(field_state),
    is_printable_(is_printable),
    is_dof_(is_dof)
{}

bool field_enums::Field_Spec::operator == (const Field_Spec& right) const {
  return name_ == right.get_name() && state_ == right.get_state();
}

bool field_enums::Field_Spec::operator != (const Field_Spec& right) const {
  return !(*this==right);
}

std::ostream& field_enums::Field_Spec::print(std::ostream& os) const {
  os << "Field: " << tostring(name_) << " Type: " << tostring(field_type_) << " Rank: " << tostring(rank_)
     << " State: " << state_
     << " Is printable: "
     <<  is_printable_ << " Is dof: " << is_dof_ << std::endl;
  return os;
}


} // namespace DICe

