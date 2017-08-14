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

#ifndef DICE_OBJECTIVE_H
#define DICE_OBJECTIVE_H

#include <iostream>

#include <DICe.h>

#include <DICe_Image.h>
#include <DICe_Subset.h>
#include <DICe_Schema.h>

#include <cassert>

namespace DICe {

/// Objective Factory
/// \param schema pointer to a schema
/// \param correlation_point_global_id the global id of the correlation point
/// returns a pointer to an objective
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Objective> objective_factory(Schema * schema,
  const int_t correlation_point_global_id);


/// \class DICe::Objective
/// \brief A container class for the subsets, optimization algorithm and initialization routine used to correlate a single point
///
/// A DICe::Objective contains all the necessary elements of a minimization objective, for example, to minimize the sum
/// squared difference of the intensity values between a subset taken from the reference and deformed image. The correlation
/// criteria, gamma, is defined here. Each objective has a
/// fast and robust method for determining the deformation map that minimizes gamma. The fast method is typically
/// image gradient based and the robust method is usually simplex based.
///
/// An objective has, as part of its member data, a reference DICe::Subset and a deformed DICe::Subset. These subsets represent
/// a small subsection of the image.
///
/// The degrees of freedom for an objective are the elements of the deformation map (displacement, rotation, stretch, etc.). A deformation
/// vector of size DICe_DEFORMATION_SIZE (see DICe.h for the ordering of the values) gets initialized in the initialize_from_previous_frame() or search()
/// methods of an objective. These values are then passed to the computeUpdateRobust() or computeUpdateFast() methods which contain
/// the bulk of the optimization algorithm. The sigma() function provides a representation of the uncertainty of the solution.

class DICE_LIB_DLL_EXPORT
Objective{

public:

  /// \brief Base class constructor uses a DICe::Schema to get parameter values and an index to denote the correlation point
  /// \param schema a DICe::Schema that holds all the correlation parameters
  /// \param correlation_point_global_id Control point local index (the local id of the control point, in parallel this is the id on the current processor)
  Objective(Schema * schema, const int_t correlation_point_global_id):
    schema_(schema),
    correlation_point_global_id_(correlation_point_global_id)
{
    if(correlation_point_global_id==-1)return;
    assert(schema_->is_initialized());
    // create the refSubset as member data
    // check to see if the schema has multishapes:
    if((*schema_->conformal_subset_defs()).find(correlation_point_global_id_)!=(*schema_->conformal_subset_defs()).end()){
      subset_ = Teuchos::rcp(new Subset(static_cast<int_t>(global_field_value(DICe::mesh::field_enums::SUBSET_COORDINATES_X_FS)),
        static_cast<int_t>(global_field_value(DICe::mesh::field_enums::SUBSET_COORDINATES_Y_FS)),
        (*schema_->conformal_subset_defs()).find(correlation_point_global_id_)->second));
    }
    // otherwise build up the subsets from x/y and w/h:
    else{
      assert(schema_->subset_dim()>0);
      subset_ = Teuchos::rcp(new Subset(static_cast<int_t>(global_field_value(DICe::mesh::field_enums::SUBSET_COORDINATES_X_FS)),
        static_cast<int_t>(global_field_value(DICe::mesh::field_enums::SUBSET_COORDINATES_Y_FS)),
        schema_->subset_dim(),schema_->subset_dim()));
    }
    assert(schema_->ref_img()!=Teuchos::null);
    subset_->initialize(schema_->ref_img());
}

  virtual ~Objective(){}

  /// \brief Correlation criteria
  /// \param deformation The deformation values for which to evaluate the correlation. These values define the mapping.
  scalar_t gamma( Teuchos::RCP<std::vector<scalar_t> > deformation) const;

  /// \brief Uncertainty measure for solution
  /// \param deformation The deformation map parameters for the current guess
  /// \param noise_level [out] Returned as the standard deviation estimate of the image noise sigma_g from Sutton et.al.
  scalar_t sigma( Teuchos::RCP<std::vector<scalar_t> > deformation,
    scalar_t & noise_level) const;

  /// \brief Measure of the slope of the optimization landscape or how deep the minimum well is
  /// \param deformation The deformation map parameters for the current guess
  scalar_t beta( Teuchos::RCP<std::vector<scalar_t> > deformation) const;

  /// \brief Gradient based optimization algorithm
  /// \param deformation [out] The deformation map parameters taken as input as the initial guess and returned as the converged solution
  /// \param num_iterations [out] The number of interations a particular frame took to execute
  virtual Status_Flag computeUpdateFast(Teuchos::RCP<std::vector<scalar_t> > deformation,
    int_t & num_iterations) = 0;

  /// \brief Simplex based optimization algorithm
  /// \param deformation [out] The deformation map parameters taken as input as the initial guess and returned as the converged solution
  /// \param num_iterations [out] The number of interations a particular frame took to execute
  /// \param override_tol set this value if for this particular subset, the tolerance should be changed
  Status_Flag computeUpdateRobust(Teuchos::RCP<std::vector<scalar_t> > deformation,
    int_t & num_iterations,
    const scalar_t & override_tol = -1.0);

  /// The number of degrees of freedom in the deformation vector
  int_t num_dofs()const{
    return dof_map_.size();
  }

  /// \brief Returns the map of degree of freedom field names (see dof_map_ description below)
  /// \param index The index of the degree of freedom map
  int_t dof_map(const size_t index)const{
      assert(index<dof_map_.size());
      return dof_map_[index];
  }

  /// \brief Returns the current value of the field specified. These values are stored in the schema
  /// \param spec Field_Spec that defines the requested field
  const mv_scalar_type & global_field_value(const DICe::mesh::field_enums::Field_Spec spec)const{
    return schema_->global_field_value(correlation_point_global_id_,spec);}


  /// Returns a pointer to the subset
  Teuchos::RCP<Subset> subset()const{
    return subset_;
  }

  /// Returns the sub image id of the subset
  int_t sub_image_id()const{
    return subset_->sub_image_id();
  }

  /// Returns the global id of the current correlation point
  int_t correlation_point_global_id()const{
    return correlation_point_global_id_;
  }

protected:

  /// Computes the difference from the exact solution and associated fields
  /// \param deformation pointer to the deformation parameters
  void computeUncertaintyFields(Teuchos::RCP<std::vector<scalar_t> > deformation);

  /// Pointer to the schema for this analysis
  Schema * schema_;
  /// Correlation point global id
  const int_t correlation_point_global_id_;
  /// Pointer to the subset
  Teuchos::RCP<Subset> subset_;
  /// Degree of freedom map. For most objectives, not all displacemen, rotation, etc. parameters are needed. This stores the index
  /// of only the degrees of freedom used in the current objective. The index maps the degree of freedom to the full set from DICe.h
  std::vector<int_t> dof_map_;
};

/// \class DICe::Objective_ZNSSD
/// \brief Sum squared differences DICe::Objective (with and without zero normalization)
/// the criteria is \f$ \gamma = \sum_i (\frac{G_i - \bar{G}}{\sqrt(\sum_i(G_i - \bar{G})^2)} - \frac{F_i - \bar{F}}{\sqrt(\sum_i(F_i - \bar{F})^2)})^2 \f$.
/// Normalization is activated when the user selects ZNSSD as the correlation_criteria. ZNSSD performs more
/// robustly in image sets where the lighting changes between frames.
///
class DICE_LIB_DLL_EXPORT
Objective_ZNSSD : public Objective
{

public:
  /// \brief Same constructor as for the base class (see base class documentation)
  ///
  /// The dof map for Objective_ZNSSD is set depending on which degrees of freedom the user has enabled (via setting an enable_<dof> parameter in the Schema).
  Objective_ZNSSD(Schema * schema,
    const int_t correlation_point_global_id):
    Objective(schema,correlation_point_global_id){
    // check that at least one of the shape functions is in use:
    TEUCHOS_TEST_FOR_EXCEPTION(!schema_->translation_enabled()&&
      !schema_->rotation_enabled()&&
      !schema_->normal_strain_enabled()&&
      !schema_->shear_strain_enabled(),std::runtime_error,"Error, no shape functions are activated");
    TEUCHOS_TEST_FOR_EXCEPTION(subset_==Teuchos::null,std::runtime_error,"");
    // populate the dof map since not all deformation dofs are used
    if(schema_->translation_enabled()){
      dof_map_.push_back(DOF_U);
      dof_map_.push_back(DOF_V);
    }
    if(schema_->rotation_enabled())
      dof_map_.push_back(DOF_THETA);
    if(schema_->normal_strain_enabled()){
      dof_map_.push_back(DOF_EX);
      dof_map_.push_back(DOF_EY);
    }
    if(schema_->shear_strain_enabled())
      dof_map_.push_back(DOF_GXY);
  }

  virtual ~Objective_ZNSSD(){}

  /// See base class documentation
  virtual Status_Flag computeUpdateFast(Teuchos::RCP<std::vector<scalar_t> > deformation,
    int_t & num_iterations);

  /// See base class documentation
  using Objective::computeUpdateRobust;

  /// See base class documentation
  using Objective::sigma;

  /// See base class documentation
  using Objective::beta;

  /// See base class documentation
  using Objective::gamma;

  /// See base class documentation
  using Objective::global_field_value;

  /// See base class documentation
  using Objective::num_dofs;

  /// See base class documentation
  using Objective::dof_map;

  /// See base class documentation
  using Objective::sub_image_id;
};



/// \class DICe::Objective_ZNSSD
/// \brief Sum squared differences DICe::Objective (with and without zero normalization) with AFFINE shape funcs
///
/// In the case of zero nomalization (ZNSSD),
/// the criteria is \f$ \gamma = \sum_i (\frac{G_i - \bar{G}}{\sqrt(\sum_i(G_i - \bar{G})^2)} - \frac{F_i - \bar{F}}{\sqrt(\sum_i(F_i - \bar{F})^2)})^2 \f$.
/// Normalization is activated when the user selects ZNSSD as the correlation_criteria. ZNSSD performs more
/// robustly in image sets where the lighting changes between frames.

class DICE_LIB_DLL_EXPORT
Objective_ZNSSD_Affine : public Objective
{

public:
  /// \brief Same constructor as for the base class (see base class documentation)
  ///
  /// The dof map for Objective_ZNSSD is set depending on which degrees of freedom the user has enabled (via setting an enable_<dof> parameter in the Schema).
  Objective_ZNSSD_Affine(Schema * schema,
    const int_t correlation_point_global_id):
    Objective(schema,correlation_point_global_id){
    // check that at least one of the shape functions is in use:
    TEUCHOS_TEST_FOR_EXCEPTION(!schema_->translation_enabled()&&
      !schema_->rotation_enabled()&&
      !schema_->normal_strain_enabled()&&
      !schema_->shear_strain_enabled(),std::runtime_error,"Error, no shape functions are activated");
    TEUCHOS_TEST_FOR_EXCEPTION(subset_==Teuchos::null,std::runtime_error,"");
    dof_map_.push_back(DOF_A);
    dof_map_.push_back(DOF_B);
    dof_map_.push_back(DOF_C);
    dof_map_.push_back(DOF_D);
    dof_map_.push_back(DOF_E);
    dof_map_.push_back(DOF_F);
    dof_map_.push_back(DOF_G);
    dof_map_.push_back(DOF_H);
    dof_map_.push_back(DOF_I);
  }

  virtual ~Objective_ZNSSD_Affine(){}

  /// See base class documentation
  virtual Status_Flag computeUpdateFast(Teuchos::RCP<std::vector<scalar_t> > deformation,
    int_t & num_iterations);

  /// See base class documentation
  using Objective::computeUpdateRobust;

  /// See base class documentation
  using Objective::sigma;

  /// See base class documentation
  using Objective::beta;

  /// See base class documentation
  using Objective::gamma;

  /// See base class documentation
  using Objective::global_field_value;

  /// See base class documentation
  using Objective::num_dofs;

  /// See base class documentation
  using Objective::dof_map;

  /// See base class documentation
  using Objective::sub_image_id;
};



}// End DICe Namespace

#endif
