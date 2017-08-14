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

#ifndef DICE_LOCALSHAPEFUNCTION_H
#define DICE_LOCALSHAPEFUNCTION_H

#include <DICe.h>
#include <DICe_Mesh.h>

/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe {

// forward declaration
class Schema;

/// \class DICe:::LocalShapeFunction
/// \brief A generic class that provides an abstraction of the local DIC shape function
class DICE_LIB_DLL_EXPORT
Local_Shape_Function {
public:
  /// base class constructor
  Local_Shape_Function():
  num_params_(0){};

  /// virtual destructor
  virtual ~Local_Shape_Function(){};

  /// the number of parameters in the shape function
  int_t num_params()const{
    return num_params_;
  }

  /// access to the parameter values
  std::vector<scalar_t> * parameters(){
    return &parameters_;
  }

  /// print the parameter values to DEBUG_MSG
  void print_parameters()const{
    std::map<DICe::mesh::field_enums::Field_Spec,size_t>::const_iterator it = spec_map_.begin();
    const std::map<DICe::mesh::field_enums::Field_Spec,size_t>::const_iterator it_end = spec_map_.end();
    std::stringstream dbg_msg;
    dbg_msg << "deformation parameters: ";
    for(;it!=it_end;++it)
      dbg_msg << it->first.get_name_label() << " " << parameters_[it->second];
    DEBUG_MSG(dbg_msg.str());
  }

  /// save off the parameters to the correct fields
  /// \param schema the schema that has the mesh with associated fields
  /// \param subset_gid the global id of the subset to save the fields for
  virtual void save_fields(Schema * schema,
    const int_t subset_gid);

  /// create the necessary fields for this shape function
  /// \param schema pointer to a schema
  void create_fields(Schema * schema);

  /// clears all the fields associated with this shape function
  /// \param schema pointer to a schema that holds the mesh with the fields
  virtual void reset_fields(Schema * schema)=0;

  /// clone the parameters values of another shape function
  /// \param shape_function the shape function to clone
  void clone(Teuchos::RCP<Local_Shape_Function> shape_function){
    assert(shape_function->num_params()==num_params_);
    for(int_t i=0;i<num_params_;++i)
      parameters_[i] = (*shape_function->parameters())[i];
  }

  /// gather the field values for each parameter
  /// \param schema pointer to a schema that has the fields
  /// \param subset_gid the id of the subset to use
  virtual void initialize_parameters_from_fields(Schema * schema,
    const int_t subset_gid)=0;

  /// map the input coordinates to the output coordinates
  /// \param x input x coordinate
  /// \param y input y coordinate
  /// \param cx input centroid coordinate (dummy variable for some shape functions)
  /// \param cy input centroid coordinate (dummy variable for some shape functions)
  /// \param out_x mapped x coordinate
  /// \param out_y mapped y coordinate
  virtual void map(const scalar_t & x,
    const scalar_t & y,
    const scalar_t & cx,
    const scalar_t & cy,
    scalar_t & out_x,
    scalar_t & out_y)=0;

  /// add the specified translation to the shape function parameter values
  /// \param u displacement in x
  /// \param v displacement in y
  virtual void add_translation(const scalar_t & u,
    const scalar_t & v)=0;

  /// replace the paramaters associated with u,v, and theta
  /// \param u input displacement in x
  /// \param v input displacement in y
  /// \param theta input rotation
  virtual void insert_motion(const scalar_t & u,
    const scalar_t & v,
    const scalar_t & theta)=0;

  /// replace the paramaters associated with u and v
  /// \param u input displacement in x
  /// \param v input displacement in y
  virtual void insert_motion(const scalar_t & u,
    const scalar_t & v)=0;

  /// clear the parameters
  virtual void clear()=0;

  /// converts the current map parameters to u, v, and theta
  /// \param x coordinate x (only used for some shape functions)
  /// \param y coordinate y (only used for some shape functions)
  /// \param out_u returns the displacement x value
  /// \param out_v returns the displacement y value
  /// \param out_theta returns the rotation
  virtual void map_to_u_v_theta(const scalar_t & x,
    const scalar_t & y,
    scalar_t & out_u,
    scalar_t & out_v,
    scalar_t & out_theta)=0;

  /// returns a reference to the given field spec location in the parameter vector
  /// \param spec the field spec
  scalar_t & parameter(const DICe::mesh::field_enums::Field_Spec & spec){
    assert(spec_map_.find(spec)!=spec_map_.end());
    return parameters_[spec_map_.find(spec)->second];
  }

  // TODO remove this (provided now for convenience)
  Teuchos::RCP<std::vector<scalar_t> > rcp(){
    return Teuchos::rcp(&parameters_,false);
  }

protected:
  /// a vector that holds the parameters of the shape function
  std::vector<scalar_t> parameters_;
  /// the total number of degrees of freedom
  int_t num_params_;
  /// stores the associated field with each DOF and the corresponding parameter index
  std::map<DICe::mesh::field_enums::Field_Spec,size_t> spec_map_;
};


/// \class DICe::Affine_Shape_Function
/// \brief six parameter seperable shape function (individual modes can be
/// turned on or off

class DICE_LIB_DLL_EXPORT
Affine_Shape_Function : public Local_Shape_Function{
public:

  /// constructor
  /// \param params parameters that define which components are active, etc.
  /// \param enable_rotation true if theta is enabled
  /// \param enable_normal_strain true if normal stretching is enabled
  /// \param enable_shear_strain true if shear stretching is enabled
  Affine_Shape_Function(const bool enable_rotation,
    const bool enable_normal_strain,
    const bool enable_shear_strain);

  /// virtual destructor
  virtual ~Affine_Shape_Function(){};

  virtual void clear();

  /// see base class description
  virtual void initialize_parameters_from_fields(Schema * schema,
    const int_t subset_gid);

  /// see base class description
  virtual void reset_fields(Schema * schema);

  /// see base class description
  virtual void map(const scalar_t & x,
    const scalar_t & y,
    const scalar_t & cx,
    const scalar_t & cy,
    scalar_t & out_x,
    scalar_t & out_y);

  /// see base class description
  virtual void add_translation(const scalar_t & u,
    const scalar_t & v);

  /// see base class description
  virtual void map_to_u_v_theta(const scalar_t & x,
    const scalar_t & y,
    scalar_t & out_u,
    scalar_t & out_v,
    scalar_t & out_theta);

  /// see base class description
  virtual void insert_motion(const scalar_t & u,
    const scalar_t & v,
    const scalar_t & theta);

  /// see base class description
  virtual void insert_motion(const scalar_t & u,
    const scalar_t & v);

private:
};

/// factory to create the right shape function
/// \param schema pointer to a schema to use to initialize the shape function
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Local_Shape_Function> shape_function_factory(Schema * schema);

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
