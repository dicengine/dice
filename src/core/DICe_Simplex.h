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

#ifndef DICE_SIMPLEX_H
#define DICE_SIMPLEX_H

#include <DICe.h>
#include <DICe_Objective.h>


namespace DICe {

/// \class DICe::Simplex
/// \brief Non-gradient based optimization algorithm class
///
/// Image gradients are not needed or used in a simplex method. The simplex method
/// determines on an optimal solution (one that minimizes the objective's gamma function)
/// by evaluating gamma at simplex vertices and modifying the simplex until it
/// converges to a single point. This method is a lot slower than a gradient-based approach,
/// but is much more robust. The simplex method will almost always converge to a value that
/// that represents the best solution given the input parameters.

class DICE_LIB_DLL_EXPORT
Simplex {

public:

  /// \brief Default constructor
  /// \param obj Pointer to a DICe::Objective, used to gain access to the gamma() method of the objective
  /// \param params Paramters that define the varaitions on the initial guess, convergence tolerance and max number of iterations
  Simplex(const DICe::Objective * const obj,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// destructor
  virtual ~Simplex(){};

  /// \brief Returns the status of the algorithm when complete
  /// \param deformation [out] Taken as the initial guess for the first iteration and returned as the converged deformation solution at successful completion.
  /// \param deltas The variations on the initial guess used to construct the other simplex points.
  /// \param num_iterations [out] Returns the number of iterations that took place
  /// A return value of MAX_ITERATIONS_REACHED means that convergence did not occur in the allowed number of iterations.
  /// A return value of CORRELATION_SUCCESSFUL means that convergence was obtained for the solution stored in the deformation vector
  const Status_Flag minimize(Teuchos::RCP<std::vector<scalar_t> > & deformation,
    Teuchos::RCP<std::vector<scalar_t> > & deltas,
    int_t & num_iterations,
    const scalar_t & threshold = 1.0E-10);

private:
  /// Maximum allowed iterations for convergence
  int_t max_iterations_;
  /// The dimension of the simplex (one dim for each free parameter)
  int_t num_dim_;
  /// Convergence tolerance
  scalar_t tolerance_;
  /// Numerically small value
  scalar_t tiny_;
  /// Pointer to a DICe::Objective, used to gain access to objective methods like gamma()
  const DICe::Objective * const obj_;

};


}// End DICe Namespace

#endif
