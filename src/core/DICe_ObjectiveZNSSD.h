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

#ifndef DICE_OBJECTIVE_ZNSSD_H
#define DICE_OBJECTIVE_ZNSSD_H

#include <iostream>

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Objective.h>
#include <DICe_Subset.h>

namespace DICe {

/// \class DICe::Objective_ZNSSD
/// \brief Sum squared differences DICe::Objective (with and without zero normalization)
///
/// This implementation of an objective has two varieties of correlation functions. The first
/// is sum squared differences (SSD) where the correlation criteria is
/// \f$ \gamma = \sum_i (G_i - F_i)^2 \f$ where \f$ F_i \f$ is the pixel intensity for a pixel in the reference
/// image and \f$ G_i \f$ is a pixel intensity from the deformed image.
///
/// In the case of zero nomalization (ZNSSD),
/// the criteria is \f$ \gamma = \sum_i (\frac{G_i - \bar{G}}{\sqrt(\sum_i(G_i - \bar{G})^2)} - \frac{F_i - \bar{F}}{\sqrt(\sum_i(F_i - \bar{F})^2)})^2 \f$.
/// Normalization is activated when the user selects ZNSSD as the correlation_criteria. ZNSSD performs more
/// robustly in image sets where the lighting changes between frames.
///
/// Since the full Newton sensitivies are not used in the case of ZNSSD, it can take a lot more iterations
/// to converge if the computeUpdateFast() method is used. For the computeUpdateRobust() method, the
/// performance is similar to SSD. It is recommended that if the normalization is used, the optimization_routine
/// be set to SIMPLEX.

// TODO implement full Newton terms

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
      dof_map_.push_back(DISPLACEMENT_X);
      dof_map_.push_back(DISPLACEMENT_Y);
    }
    if(schema_->rotation_enabled())
      dof_map_.push_back(ROTATION_Z);
    if(schema_->normal_strain_enabled()){
      dof_map_.push_back(NORMAL_STRAIN_X);
      dof_map_.push_back(NORMAL_STRAIN_Y);
    }
    if(schema_->shear_strain_enabled())
      dof_map_.push_back(SHEAR_STRAIN_XY);
  }

  virtual ~Objective_ZNSSD(){}

  /// See base class documentation
  virtual scalar_t gamma( Teuchos::RCP<std::vector<scalar_t> > & deformation)const;

  /// See base class documentation
  virtual scalar_t sigma( Teuchos::RCP<std::vector<scalar_t> > & deformation,
    scalar_t & noise_level) const;

  /// See base class documentation
  virtual scalar_t beta( Teuchos::RCP<std::vector<scalar_t> > & deformation) const;

  /// See base class documentation
  virtual Status_Flag computeUpdateFast(Teuchos::RCP<std::vector<scalar_t> > & deformation,
    int_t & num_iterations);

  /// See base class documentation
  virtual Status_Flag computeUpdateRobust(Teuchos::RCP<std::vector<scalar_t> > & deformation,
    int_t & num_iterations,
    const scalar_t & override_tol = -1.0);

  /// See base class documentation
  using Objective::local_field_value;

  /// See base class documentation
  using Objective::local_field_value_nm1;

  /// See base class documentation
  using Objective::num_dofs;

  /// See base class documentation
  using Objective::dof_map;

  /// See base class documentation
  using Objective::subset_;
};

}// End DICe Namespace

#endif
