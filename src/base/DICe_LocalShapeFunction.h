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
#include <DICe_FieldEnums.h>
#include <DICe_CamSystem.h>

#include <Teuchos_RCP.hpp>


namespace DICe_LocalShapeFunction {
  enum Projection_Params {
    ZP = 0,
    THETA,
    PHI,
    MAX_CAM_PROJECTION_PARAMS,
    NO_SUCH_CAM_PROJECTION_PARAMS
  };

  const static char * projectionParamsStrings[] = {
    "ZP",
    "THETA",
    "PHI"
  };

}

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
    std::map<DICe::field_enums::Field_Spec,size_t>::const_iterator it = spec_map_.begin();
    const std::map<DICe::field_enums::Field_Spec,size_t>::const_iterator it_end = spec_map_.end();
    std::stringstream dbg_msg;
    dbg_msg << "deformation parameters: ";
    for(;it!=it_end;++it)
      dbg_msg << it->first.get_name_label() << " " << parameters_[it->second] << " ";
    DEBUG_MSG(dbg_msg.str());
  }

  /// save off the parameters to the correct fields
  /// \param schema the schema that has the mesh with associated fields
  /// \param subset_gid the global id of the subset to save the fields for
  virtual void save_fields(Schema * schema,
    const int_t subset_gid);

  /// clears all the fields associated with this shape function
  /// \param schema pointer to a schema that holds the mesh with the fields
  virtual void reset_fields(Schema * schema);

  /// clone the parameters values of another shape function
  /// \param shape_function the shape function to clone
  void clone(Teuchos::RCP<Local_Shape_Function> shape_function);

  /// gather the field values for each parameter
  /// \param schema pointer to a schema that has the fields
  /// \param subset_gid the id of the subset to use
  virtual void initialize_parameters_from_fields(Schema * schema,
    const int_t subset_gid);

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
  virtual void clear();

  /// converts the current map parameters to u, v, and theta
  /// \param cx input centroid coordinate (dummy variable for some shape functions)
  /// \param cy input centroid coordinate (dummy variable for some shape functions)
  /// \param out_u returns the displacement x value
  /// \param out_v returns the displacement y value
  /// \param out_theta returns the rotation
  virtual void map_to_u_v_theta(const scalar_t & cx,
    const scalar_t & cy,
    scalar_t & out_u,
    scalar_t & out_v,
    scalar_t & out_theta)=0;

  /// converts the mapping from referencing one centroid to another
  /// \param delta_x change in centroid x location
  /// \param delta_y change in centroid y location
  virtual void update_params_for_centroid_change(const scalar_t & delta_x,
    const scalar_t & delta_y)=0;

  /// returns a reference to the given field spec location in the parameter vector
  /// \param spec the field spec
  scalar_t & operator()(const DICe::field_enums::Field_Spec & spec){
    assert(spec_map_.find(spec)!=spec_map_.end());
    return parameters_[spec_map_.find(spec)->second];
  }

  /// returns a reference to the given field spec location in the parameter vector
  /// \param index index of the vector entry
  scalar_t & operator()(const size_t index){
    assert(index<parameters_.size());
    return parameters_[index];
  }

  /// returns the value the given field spec location in the parameter vector if it exists
  /// or zero if it does not
  /// \param spec the field spec
  scalar_t parameter(const DICe::field_enums::Field_Spec & spec)const{
    if(spec_map_.find(spec)==spec_map_.end()){
      return 0.0;
    }
    else
      return parameters_[spec_map_.find(spec)->second];
  }

  /// method that computes the residuals for this shape function
  /// \param x x-coordinate of the point to compute for
  /// \param y y-coordinate of the point to compute for
  /// \param cx input centroid coordinate (dummy variable for some shape functions)
  /// \param cy input centroid coordinate (dummy variable for some shape functions)
  /// \param gx x image gradient
  /// \param gy y image gradient
  /// \param residuals [out] the vector to store the residuals in
  /// \param use_ref_grads true if the gradients should be used from the reference image (so they need to be adjusted for the current def map)
  virtual void residuals(const scalar_t & x,
    const scalar_t & y,
    const scalar_t & cx,
    const scalar_t & cy,
    const scalar_t & gx,
    const scalar_t & gy,
    std::vector<scalar_t> & residuals,
    const bool use_ref_grads=false)=0;

  /// update the parameter values based on an input vector
  /// \param update reference to the update vector
  void update(const std::vector<scalar_t> & update);

  /// returns true if the solution is converged
  /// \param old_parameters vector of the previous guess for the parameters
  /// \param tol the solution tolerance
  virtual bool test_for_convergence(const std::vector<scalar_t> & old_parameters,
    const scalar_t & tol);

  // TODO remove this (provided now for convenience)
  Teuchos::RCP<std::vector<scalar_t> > rcp(){
    return Teuchos::rcp(&parameters_,false);
  }

  /// returns a pointer to the vector of delta values for this shape function
  Teuchos::RCP<std::vector<scalar_t> > deltas(){
    return Teuchos::rcp(&deltas_,false);
  }

  /// returns a pointer to the spec map
  std::map<DICe::field_enums::Field_Spec,size_t> * spec_map(){
    return &spec_map_;
  }

protected:
  /// a vector that holds the parameters of the shape function
  std::vector<scalar_t> parameters_;
  /// a vector that holds the deltas to use for a simplex method
  std::vector<scalar_t> deltas_;
  /// the total number of degrees of freedom
  int_t num_params_;
  /// stores the associated field with each DOF and the corresponding parameter index
  std::map<DICe::field_enums::Field_Spec,size_t> spec_map_;
};


/// \class DICe::Affine_Shape_Function
/// \brief six parameter seperable shape function (individual modes can be
/// turned on or off

class DICE_LIB_DLL_EXPORT
Affine_Shape_Function : public Local_Shape_Function{
public:

  /// constructor
  /// \param schema pointer to a schema used to initialize the shape function
  Affine_Shape_Function(Schema * schema);

  /// constructor with no schema, but boolean flags instead
  /// \param enable_rotation true if rotation parameter should be included
  /// \param enable_normal_strain true if normal strain should be included
  /// \param enable_shear_strain true if shear strain should be included
  Affine_Shape_Function(const bool enable_rotation,
    const bool enable_normal_strain,
    const bool enable_shear_strain,
    const scalar_t & delta_disp=1.0,
    const scalar_t & delta_theta=0.1);

  /// initializes the affine shape function
  /// \param enable_rotation true if rotation parameter should be included
  /// \param enable_normal_strain true if normal strain should be included
  /// \param enable_shear_strain true if shear strain should be included
  /// \param delta_disp the displacement delta to use for simplex optimization
  /// \param delta_theta the rotation delta to use for simplex optimization
  void init(const bool enable_rotation,
    const bool enable_normal_strain,
    const bool enable_shear_strain,
    const scalar_t & delta_disp=1.0,
    const scalar_t & delta_theta=0.1);

  /// virtual destructor
  virtual ~Affine_Shape_Function(){};

  /// see base class description
  virtual void initialize_parameters_from_fields(Schema * schema,
    const int_t subset_gid);

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
  virtual void map_to_u_v_theta(const scalar_t & cx,
    const scalar_t & cy,
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

  /// see base class description
  virtual void residuals(const scalar_t & x,
    const scalar_t & y,
    const scalar_t & cx,
    const scalar_t & cy,
    const scalar_t & gx,
    const scalar_t & gy,
    std::vector<scalar_t> & residuals,
    const bool use_ref_grads=false);

  /// see base class description
  virtual bool test_for_convergence(const std::vector<scalar_t> & old_parameters,
    const scalar_t & tol);

  /// see base class description
  virtual void update_params_for_centroid_change(const scalar_t & delta_x,
    const scalar_t & delta_y){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, this method has not been implemented yet for Affine_Shape_Function");
  };


private:
};

/// \class DICe::Quadratic_Shape_Function
/// \brief 12 parameter quadratic mapping for local shape function

class DICE_LIB_DLL_EXPORT
Quadratic_Shape_Function : public Local_Shape_Function{
public:

  /// constructor
  Quadratic_Shape_Function();

  /// virtual destructor
  virtual ~Quadratic_Shape_Function(){};

  /// clear the parameters
  virtual void clear();

  /// clears all the fields associated with this shape function
  /// \param schema pointer to a schema that holds the mesh with the fields
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
  virtual void map_to_u_v_theta(const scalar_t & cx,
    const scalar_t & cy,
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

  /// see base class description
  virtual void residuals(const scalar_t & x,
    const scalar_t & y,
    const scalar_t & cx,
    const scalar_t & cy,
    const scalar_t & gx,
    const scalar_t & gy,
    std::vector<scalar_t> & residuals,
    const bool use_ref_grads=false);

  /// see base class description
  virtual void save_fields(Schema * schema,
    const int_t subset_gid);

  /// see base class description
  virtual void update_params_for_centroid_change(const scalar_t & delta_x,
    const scalar_t & delta_y);
private:
};


/// factory to create the right shape function
/// \param schema pointer to a schema to use to initialize the shape function
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Local_Shape_Function> shape_function_factory(Schema * schema=NULL);

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
