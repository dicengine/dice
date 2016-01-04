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

#ifndef DICE_MESHENUMS_H_
#define DICE_MESHENUMS_H_

#include <DICe.h>

#include <stdlib.h>
#include <string>
#include <iostream>
#include <Teuchos_RCP.hpp>

namespace DICe{
namespace mesh{

enum Step_Status
{
  STEP_CONVERGED = 0,
  STEP_ITERATING,
  STEP_FAILED
};

enum Component
{
  NO_COMP=0,
  X_COMP,
  Y_COMP,
  Z_COMP
};

std::string tostring(const Component & comp);
Component string_to_component(const std::string & input_string_type);
std::string index_to_component_string(const unsigned index);
std::string index_to_component(const unsigned index);

enum Base_Element_Type
{
  NO_SUCH_ELEMENT_TYPE=0,
  HEX8,
  QUAD4,
  TETRA4,
  TETRA,
  TRI3,
  CVFEM_TRI,
  CVFEM_TETRA,
  SPHERE,
  PYRAMID5
};

std::string tostring(const Base_Element_Type & base_element_type);
Base_Element_Type string_to_base_element_type(const std::string & input_string_type);

enum Physics_Term
{
  NO_SUCH_PHYSICS_TERM=0,
  MASS,
  DIFF,
  ADV,
  SRC
};

std::string tostring(const Physics_Term & physics_term);
Physics_Term string_type_to_physics_term(const std::string & input_string_type);


namespace field_enums{
enum Field_Type
{
  NO_SUCH_FIELD_TYPE=0,
  SCALAR_FIELD_TYPE,
  VECTOR_FIELD_TYPE
};
enum Field_Name
{
  NO_SUCH_FIELD_NAME=0,
  INITIAL_COORDINATES,
  CURRENT_COORDINATES,
  INITIAL_CELL_COORDINATES,
  CURRENT_CELL_COORDINATES,
  INITIAL_CELL_SIZE,
  INITIAL_SUBELEMENT_SIZE,
  INITIAL_WEIGHTED_CELL_SIZE,
  INITIAL_CELL_RADIUS,
  CVFEM_AD_RESIDUAL,
  CVFEM_AD_LHS,
  CVFEM_AD_G_PHI,
  CVFEM_AD_GRAD_PHI,
  CVFEM_AD_LAMBDA,
  CVFEM_AD_GRAD_J,
  CVFEM_AD_IS_BOUNDARY,
  CVFEM_AD_IMAGE_PHI,
  CVFEM_AD_PHI,
  CVFEM_AD_PHI_0,
  CVFEM_AD_DIFFUSIVITY,
  CVFEM_AD_ADVECTION_VELOCITY,
  CVFEM_AD_ELEM_ADV_VEL,
  CVFEM_AD_NODE_ADV_VEL,
  ELAST_FEM_LAMBDA,
  ELAST_FEM_MU,
  ELAST_FEM_RESIDUAL,
  ELAST_FEM_LHS,
  ELAST_FEM_DISPLACEMENT,
  INTERNAL_FACE_EDGE_NORMAL,
  INTERNAL_FACE_EDGE_COORDINATES,
  INTERNAL_FACE_EDGE_SIZE,
  EXTERNAL_FACE_EDGE_NORMAL,
  EXTERNAL_FACE_EDGE_COORDINATES,
  EXTERNAL_FACE_EDGE_SIZE,
  INTERNAL_CELL_COORDINATES,
  INTERNAL_CELL_SIZE,
  BLOCK_ID,
  PROCESSOR_ID
};
enum Entity_Rank
{
  NO_SUCH_ENTITY_RANK=0,
  NODE_RANK,
  ELEMENT_RANK,
  SUBELEMENT_RANK,
  FACE_RANK,
  EDGE_RANK,
  INTERNAL_FACE_EDGE_RANK,
  EXTERNAL_FACE_EDGE_RANK,
  INTERNAL_CELL_RANK,
  BOND_RANK
};
enum Field_State
{
  NO_FIELD_STATE=0,
  STATE_N_MINUS_ONE,
  STATE_N,
  STATE_N_PLUS_ONE
};
}  //field enums

std::string tostring(const field_enums::Field_Type & field_type);
std::string tostring(const field_enums::Field_Name & field_name);
std::string tostring(const field_enums::Entity_Rank & field_rank);
unsigned toindex(const Component comp);
field_enums::Field_Type string_type_to_field_type(const std::string & input_string_type);
field_enums::Field_Name string_type_to_field_name(const std::string & input_string_type);
void update_field_names(const field_enums::Field_Name & field_name,
  const std::string & input_string);
int_t num_field_names();
std::list<std::string> get_reverse_sorted_field_names();
field_enums::Entity_Rank string_type_to_entity_rank(const std::string & input_string_type);

namespace field_enums{
/// A struct that holds all the necessary information to define a field
struct Field_Spec  {
private:
  /// field type
  Field_Type field_type_;
  /// field name
  Field_Name name_;
  /// field rank
  Entity_Rank rank_;
  /// field state
  Field_State state_;
  /// True if this field can be printed or output to exodus file
  bool is_printable_;
  /// True if this field is a degree of freedom for a particular physics
  bool is_dof_;
public:
  /// Constructor
  /// \param field_type The type of field
  /// \param label The name of the field
  /// \param rank The rank of the field
  /// \param field_state The state for this field
  /// \param is_printable True if this field can be output in the exodus mesh
  /// \param is_dof True if this field is a degree of freedon
  Field_Spec(const Field_Type field_type,
    const Field_Name label,
    const Entity_Rank rank,
    const Field_State field_state,
    const bool is_printable,
    const bool is_dof=false);

  /// Constructor with no arguments
  Field_Spec();

  /// Comparison operator
  /// \param right The field spec to compare to
  bool operator == (const Field_Spec& right) const;

  /// Comparison operator
  /// \param right The field spec to compare to
  bool operator != (const Field_Spec& right) const;

  /// Comparison operator
  /// \param right The field spec to compare to
  bool operator< (const Field_Spec & right) const
  {
    if( name_ == right.get_name())
      return state_ < right.get_state();
    else
      return name_ < right.get_name();
  }

  /// Print the field to the given os
  /// \param os The output stream
  std::ostream& print(std::ostream& os) const;

  /// Return the field type
  Field_Type get_field_type() const {
    return field_type_;
  }

  /// Return the field name
  Field_Name get_name() const {
    return name_;
  }

  /// Return the field state
  Field_State get_state() const {
    return state_;
  }

  /// Set the field's state
  void set_state(Field_State state) {
    state_ = state;
  }

  /// Return the name of the field as a string
  std::string get_name_label() const {
    return tostring(name_);
  }

  /// Return the rank of the field
  Entity_Rank get_rank() const {
    return rank_;
  }

  /// Returns true if the field is printable
  bool is_printable() const {
    return is_printable_;
  }

  /// Returns true if this field is a degree of freedom for a physics class
  bool is_dof() const {
    return is_dof_;
  }

  /// True if the two fields are compatible
  /// \param field_spec The field_spec to compare to
  bool is_compatible(const Field_Spec & field_spec)const{
    return (field_spec.get_field_type()==field_type_&&field_spec.get_rank()==rank_);
  }

  /// Returns true if the field can be a subrank of the given rank (for example internal faces or edges)
  bool is_subrank_of(const field_enums::Entity_Rank parent_rank)
  {
    if(rank_!=field_enums::INTERNAL_FACE_EDGE_RANK && rank_!=field_enums::INTERNAL_CELL_RANK && rank_!= EXTERNAL_FACE_EDGE_RANK) return false;
    if(parent_rank==field_enums::ELEMENT_RANK) return true;
    else return false;
  }
};

inline std::ostream& operator<<(std::ostream& os, const Field_Spec& fs) {
    return fs.print(os);
}

const Field_Spec NO_SUCH_FS(field_enums::NO_SUCH_FIELD_TYPE,field_enums::NO_SUCH_FIELD_NAME,field_enums::NO_SUCH_ENTITY_RANK,field_enums::NO_FIELD_STATE,false);
const Field_Spec PROCESSOR_ID_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::PROCESSOR_ID,field_enums::ELEMENT_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec BLOCK_ID_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::BLOCK_ID,field_enums::ELEMENT_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec INITIAL_COORDINATES_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::INITIAL_COORDINATES,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CURRENT_COORDINATES_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::CURRENT_COORDINATES,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec INITIAL_CELL_COORDINATES_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::INITIAL_CELL_COORDINATES,field_enums::ELEMENT_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CURRENT_CELL_COORDINATES_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::CURRENT_CELL_COORDINATES,field_enums::ELEMENT_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec INITIAL_CELL_SIZE_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::INITIAL_CELL_SIZE,field_enums::ELEMENT_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec INITIAL_SUBELEMENT_SIZE_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::INITIAL_SUBELEMENT_SIZE,field_enums::SUBELEMENT_RANK,field_enums::NO_FIELD_STATE,false);
const Field_Spec INITIAL_WEIGHTED_CELL_SIZE_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::INITIAL_WEIGHTED_CELL_SIZE,field_enums::ELEMENT_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec INITIAL_CELL_RADIUS_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::INITIAL_CELL_RADIUS,field_enums::ELEMENT_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_RESIDUAL_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_RESIDUAL,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_LHS_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_LHS,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_G_PHI_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_G_PHI,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_GRAD_PHI_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::CVFEM_AD_GRAD_PHI,field_enums::ELEMENT_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_LAMBDA_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_LAMBDA,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_LAMBDA_N_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_LAMBDA,field_enums::NODE_RANK,field_enums::STATE_N,false);
//const Field_Spec CVFEM_AD_GRAD_J_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::CVFEM_AD_GRAD_J,field_enums::INTERNAL_FACE_EDGE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_GRAD_J_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::CVFEM_AD_GRAD_J,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_IS_BOUNDARY_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_IS_BOUNDARY,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_PHI_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_PHI,field_enums::NODE_RANK,field_enums::STATE_N_PLUS_ONE,true,true);
const Field_Spec CVFEM_AD_PHI_N_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_PHI,field_enums::NODE_RANK,field_enums::STATE_N,false);
const Field_Spec CVFEM_AD_PHI_0_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_PHI_0,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_IMAGE_PHI_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_IMAGE_PHI,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_DIFFUSIVITY_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::CVFEM_AD_DIFFUSIVITY,field_enums::INTERNAL_FACE_EDGE_RANK,field_enums::NO_FIELD_STATE,false);
const Field_Spec CVFEM_AD_ADVECTION_VELOCITY_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::CVFEM_AD_ADVECTION_VELOCITY,field_enums::INTERNAL_FACE_EDGE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_ELEM_ADV_VEL_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::CVFEM_AD_ELEM_ADV_VEL,field_enums::ELEMENT_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec CVFEM_AD_NODE_ADV_VEL_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::CVFEM_AD_NODE_ADV_VEL,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec INTERNAL_FACE_EDGE_NORMAL_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::INTERNAL_FACE_EDGE_NORMAL,field_enums::INTERNAL_FACE_EDGE_RANK,field_enums::NO_FIELD_STATE,false);
const Field_Spec INTERNAL_FACE_EDGE_COORDINATES_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::INTERNAL_FACE_EDGE_COORDINATES,field_enums::INTERNAL_FACE_EDGE_RANK,field_enums::NO_FIELD_STATE,false);
const Field_Spec INTERNAL_FACE_EDGE_SIZE_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::INTERNAL_FACE_EDGE_SIZE,field_enums::INTERNAL_FACE_EDGE_RANK,field_enums::NO_FIELD_STATE,false);
//const Field_Spec EXTERNAL_FACE_EDGE_NORMAL_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::EXTERNAL_FACE_EDGE_NORMAL,field_enums::EXTERNAL_FACE_EDGE_RANK,field_enums::NO_FIELD_STATE,false);
//const Field_Spec EXTERNAL_FACE_EDGE_COORDINATES_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::EXTERNAL_FACE_EDGE_COORDINATES,field_enums::EXTERNAL_FACE_EDGE_RANK,field_enums::NO_FIELD_STATE,false);
//const Field_Spec EXTERNAL_FACE_EDGE_SIZE_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::EXTERNAL_FACE_EDGE_SIZE,field_enums::EXTERNAL_FACE_EDGE_RANK,field_enums::NO_FIELD_STATE,false);
const Field_Spec INTERNAL_CELL_COORDINATES_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::INTERNAL_CELL_COORDINATES,field_enums::INTERNAL_CELL_RANK,field_enums::NO_FIELD_STATE,false);
const Field_Spec INTERNAL_CELL_SIZE_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::INTERNAL_CELL_SIZE,field_enums::INTERNAL_CELL_RANK,field_enums::NO_FIELD_STATE,false);
const Field_Spec ELAST_FEM_LAMBDA_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::ELAST_FEM_LAMBDA,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec ELAST_FEM_MU_FS(field_enums::SCALAR_FIELD_TYPE,field_enums::ELAST_FEM_MU,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec ELAST_FEM_RESIDUAL_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::ELAST_FEM_RESIDUAL,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec ELAST_FEM_LHS_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::ELAST_FEM_LHS,field_enums::NODE_RANK,field_enums::NO_FIELD_STATE,true);
const Field_Spec ELAST_FEM_DISPLACEMENT_FS(field_enums::VECTOR_FIELD_TYPE,field_enums::ELAST_FEM_DISPLACEMENT,field_enums::NODE_RANK,field_enums::STATE_N_PLUS_ONE,true,true);

// TODO: There's probably a better way to keep track of the number of fields and their IDs
const int_t num_fields_defined = 37;

const field_enums::Field_Spec fs_spec_vec[num_fields_defined] = {
    NO_SUCH_FS,
    INITIAL_COORDINATES_FS,
    CURRENT_COORDINATES_FS,
    INITIAL_CELL_COORDINATES_FS,
    CURRENT_CELL_COORDINATES_FS,
    INITIAL_CELL_SIZE_FS,
    INITIAL_SUBELEMENT_SIZE_FS,
    INITIAL_WEIGHTED_CELL_SIZE_FS,
    INITIAL_CELL_RADIUS_FS,
    CVFEM_AD_RESIDUAL_FS,
    CVFEM_AD_IS_BOUNDARY_FS,
    CVFEM_AD_LHS_FS,
    CVFEM_AD_G_PHI_FS,
    CVFEM_AD_GRAD_PHI_FS,
    CVFEM_AD_LAMBDA_FS,
    CVFEM_AD_LAMBDA_N_FS,
    CVFEM_AD_GRAD_J_FS,
    CVFEM_AD_IMAGE_PHI_FS,
    CVFEM_AD_PHI_FS,
    CVFEM_AD_PHI_N_FS,
    CVFEM_AD_PHI_0_FS,
    CVFEM_AD_DIFFUSIVITY_FS,
    CVFEM_AD_ADVECTION_VELOCITY_FS,
    CVFEM_AD_ELEM_ADV_VEL_FS,
    CVFEM_AD_NODE_ADV_VEL_FS,
    ELAST_FEM_LAMBDA_FS,
    ELAST_FEM_MU_FS,
    ELAST_FEM_RESIDUAL_FS,
    ELAST_FEM_LHS_FS,
    ELAST_FEM_DISPLACEMENT_FS,
    INTERNAL_FACE_EDGE_NORMAL_FS,
    INTERNAL_FACE_EDGE_COORDINATES_FS,
    INTERNAL_FACE_EDGE_SIZE_FS,
//    EXTERNAL_FACE_EDGE_NORMAL_FS,
//    EXTERNAL_FACE_EDGE_COORDINATES_FS,
//    EXTERNAL_FACE_EDGE_SIZE_FS,
    INTERNAL_CELL_COORDINATES_FS,
    INTERNAL_CELL_SIZE_FS,
    BLOCK_ID_FS,
    PROCESSOR_ID_FS
    // don't forget to add one to num_fields_defined
};

const std::vector<field_enums::Field_Spec> FIELD_SPEC_VECTOR(fs_spec_vec,fs_spec_vec+num_fields_defined);
} //field enums

} //namespace mesh
} //namespace DICe

#endif /* DICE_MESHENUMS_H_ */
