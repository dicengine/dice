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

#ifndef DICE_OBJECTIVE_H
#define DICE_OBJECTIVE_H

#include <iostream>

#include <DICe.h>

#include <DICe_Image.h>
#include <DICe_Subset.h>
#include <DICe_Schema.h>

#include <cassert>

namespace DICe {

/// \class DICe::Objective
/// \brief A container class for the subsets, optimization algorithm and initialization routine used to correlate a single point
///
/// A DICe::Objective contains all the necessary elements of a minimization objective, for example, to minimize the sum
/// squared difference of the intensity values between a subset taken from the reference and deformed image. The correlation
/// criteria, gamma, is defined here as well as how the optimzation is seeded with an initial guess. Each objective has a
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
      subset_ = Teuchos::rcp(new Subset(static_cast<int_t>(local_field_value(COORDINATE_X)),
        static_cast<int_t>(local_field_value(COORDINATE_Y)),
        (*schema_->conformal_subset_defs()).find(correlation_point_global_id_)->second));
    }
    // otherwise build up the subsets from x/y and w/h:
    else{
      assert(schema_->subset_dim()>0);
      subset_ = Teuchos::rcp(new Subset(static_cast<int_t>(local_field_value(COORDINATE_X)),
        static_cast<int_t>(local_field_value(COORDINATE_Y)),
        schema_->subset_dim(),schema_->subset_dim()));
    }
    subset_->initialize(schema_->ref_img());
}

  virtual ~Objective(){}

  /// \brief Correlation criteria
  /// \param deformation The deformation values for which to evaluate the correlation. These values define the mapping.
  virtual scalar_t gamma( Teuchos::RCP<std::vector<scalar_t> > & deformation) const = 0;

  /// \brief Uncertainty measure for solution
  /// \param deformation The deformation map parameters for the current guess
  virtual scalar_t sigma( Teuchos::RCP<std::vector<scalar_t> > & deformation) const = 0;

  /// \brief Gradient based optimization algorithm
  /// \param deformation [out] The deformation map parameters taken as input as the initial guess and returned as the converged solution
  /// \param num_iterations [out] The number of interations a particular frame took to execute
  virtual Status_Flag computeUpdateFast(Teuchos::RCP<std::vector<scalar_t> > & deformation,
    int_t & num_iterations) = 0;

  /// \brief Simplex based optimization algorithm
  /// \param deformation [out] The deformation map parameters taken as input as the initial guess and returned as the converged solution
  /// \param num_iterations [out] The number of interations a particular frame took to execute
  /// \param override_tol set this value if for this particular subset, the tolerance should be changed
  virtual Status_Flag computeUpdateRobust(Teuchos::RCP<std::vector<scalar_t> > & deformation,
    int_t & num_iterations,
    const scalar_t & override_tol = -1.0) = 0;

  /// \brief Initialize deformation values using the previous frame's solution
  /// \param deformation [out] Returned as the deformation parameters from the previous step.
  /// If the previous step was not successful (the sigma value is -1) this method returns unsuccessfully.
  virtual Status_Flag initialize_from_previous_frame(Teuchos::RCP<std::vector<scalar_t> > & deformation) = 0;

  /// \brief Initialize deformation values by searching within a region of the current solution
  /// \param deformation [out] Returned as the deformation parameters found by searching the parameter space.
  /// \param precision_level The number of digits of precision to converge the solution (higher is more precise)
  /// \param return_value The resulting gamma value at the best solution
  virtual Status_Flag search(Teuchos::RCP<std::vector<scalar_t> > & deformation,
    const int_t precision_level,
    scalar_t & return_value) = 0;

  /// \brief Initialize the values by performing a search over a window with a given step
  /// \param deformation [out] returned with the initial guess
  /// \param window_size the size of the search window
  /// \param step_size the increment over which to search the window
  /// \param return_value the gamma value at the best initial guess
  virtual Status_Flag search_step(Teuchos::RCP<std::vector<scalar_t> > & deformation,
    const int_t window_size,
    const scalar_t step_size,
    scalar_t & return_value) = 0;

  /// \brief Initialize deformation values by using a neighbors current solution
  /// \param deformation [out] Returned as the deformation parameters taken from a neighboring point's current solution.
  /// If the neighbor's current frame was not successful (the sigma value is -1) this method returns unsuccessfully.
  virtual Status_Flag initialize_from_neighbor(Teuchos::RCP<std::vector<scalar_t> > & deformation) = 0;

  /// The number of degrees of freedom in the deformation vector
  int_t num_dofs()const{
    return dof_map_.size();
  }

  /// \brief Returns the map of degree of freedom field names (see dof_map_ description below)
  /// \param index The index of the degree of freedom map
  Field_Name dof_map(const size_t index)const{
      assert(index<dof_map_.size());
      return dof_map_[index];
  }

  /// \brief Returns the current local distributed value of the field specified. These values are stored in the schema
  /// \param name String name of the field (must match a valid enum in DICe.h)
#if DICE_TPETRA
  const scalar_t& local_field_value(const Field_Name name)const{
#else // Epetra does not have a scalar type so have to hard code double here
    const double& local_field_value(const Field_Name name)const{
#endif
    return schema_->local_field_value(correlation_point_global_id_,name);}

  /// \brief Returns the previous frame's value of the local distributed field specified. These values are stored in the schema.
  ///
  /// The solution at frame n - 1 is only stored for projection_method==VELCOITY_BASED
  /// \param name String name of the field (must match a valid enum in DICe.h)
#if DICE_TPETRA
    const scalar_t& local_field_value_nm1(const Field_Name name)const{
#else // Epetra does not have a scalar type so have to hard code double here
    const double& local_field_value_nm1(const Field_Name name)const{
#endif
    return schema_->local_field_value_nm1(correlation_point_global_id_,name);}

  /// Returns a pointer to the subset
  Teuchos::RCP<Subset> subset()const{
    return subset_;
  }

  /// Returns the global id of the current correlation point
  int_t correlation_point_global_id()const{
    return correlation_point_global_id_;
  }

protected:

  /// Pointer to the schema for this analysis
  Schema * schema_;
  /// Correlation point global id
  const int_t correlation_point_global_id_;
  /// Pointer to the subset
  Teuchos::RCP<Subset> subset_;
  /// Degree of freedom map. For most objectives, not all displacemen, rotation, etc. parameters are needed. This stores the index
  /// of only the degrees of freedom used in the current objective. The index maps the degree of freedom to the full set from DICe.h
  std::vector<Field_Name> dof_map_;
};

}// End DICe Namespace

#endif
