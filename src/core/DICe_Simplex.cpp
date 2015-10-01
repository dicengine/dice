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

#include <DICe_Simplex.h>

#include <cassert>

namespace DICe {

Simplex::Simplex(const DICe::Objective * const obj,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  max_iterations_(1000),
  num_dim_(0),
  tolerance_(1.0E-8),
  tiny_(1.0E-10),
  obj_(obj)
{
  if(params!=Teuchos::null){
    if(params->isParameter(DICe::max_iterations)) max_iterations_ = params->get<int_t>(DICe::max_iterations);
    if(params->isParameter(DICe::tolerance)) tolerance_ = params->get<scalar_t>(DICe::tolerance);
  }
  // get num dims from the objective
  assert(obj_);
  num_dim_ = obj_->num_dofs();
  assert(num_dim_!=0);
  assert(num_dim_<=DICE_DEFORMATION_SIZE);
}

const Status_Flag
Simplex::minimize(Teuchos::RCP<std::vector<scalar_t> > & deformation,
  Teuchos::RCP<std::vector<scalar_t> > & deltas,
  int_t & num_iterations,
  const scalar_t & threshold){

  assert(deformation->size()==DICE_DEFORMATION_SIZE);
  assert(deltas->size()==num_dim_);

  DEBUG_MSG("Conducting multidimensional simplex minimization");
  DEBUG_MSG("Max iterations: " << max_iterations_ << " tolerance: " << tolerance_);
  DEBUG_MSG("Initial guess: ");
#ifdef DICE_DEBUG_MSG
  std::cout << " POINT 0: ";
  for(int_t j=0;j<DICE_DEFORMATION_SIZE;++j) std::cout << " " << (*deformation)[j];
  std::cout << std::endl;
#endif

  // allocate temp storage for routine and initialize the simplex vertices

  const int_t mpts = num_dim_ + 1;
  scalar_t * gamma_values = new scalar_t[mpts];
  std::vector< Teuchos::RCP<std::vector<scalar_t> > > points(mpts);
  for(int_t i=0;i<mpts;++i){
    points[i] = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));
    // set the points for the initial bounding simplex
    for(int_t j=0;j<DICE_DEFORMATION_SIZE;++j){
      (*points[i])[j] = (*deformation)[j];
    }
    if(i>0)
      (*points[i])[obj_->dof_map(i-1)] += (*deltas)[i-1];
#ifdef DICE_DEBUG_MSG
    std::cout << " SIMPLEX POINT: ";
    for(int_t j=0;j<DICE_DEFORMATION_SIZE;++j) std::cout << " " << (*points[i])[j];
    std::cout << std::endl;
#endif
    // evaluate gamma at these points
    gamma_values[i] = obj_->gamma(points[i]);
    DEBUG_MSG("Gamma value for this point: " << gamma_values[i]);
    if(i==0&&gamma_values[i]<threshold&&gamma_values[i]>=0.0){
      num_iterations = 0;
      DEBUG_MSG("Initial deformation guess is good enough (gamma < " << threshold << " for this guess)");
      return CORRELATION_SUCCESSFUL;
    }
  }

  // work variables

  int_t inhi,j;
  scalar_t ysave;
  int_t nfunk = 0;
  scalar_t * points_column_sums = new scalar_t[num_dim_];
  scalar_t * ptry = new scalar_t[num_dim_];

  // sum up the columns of the simplex vertices

  for (int_t j = 0; j < num_dim_; j++) {
    scalar_t sum = 0.0;
    for (int_t i = 0; i < mpts; i++) {
      sum += (*points[i])[obj_->dof_map(j)];
    }
    points_column_sums[j] = sum;
  }

  // simplex minimization routine
  int_t iteration = 0;
  for (iteration=0; iteration < max_iterations_; iteration++) {
    if( iteration >= max_iterations_-1){
      DEBUG_MSG("Simplex method max iterations exceeded");
      delete [] gamma_values;
      delete [] points_column_sums;
      delete [] ptry;
      num_iterations = iteration;
      return MAX_ITERATIONS_REACHED;
    }

    int_t ilo = 0;
    int_t ihi = gamma_values[0]>gamma_values[1] ? (inhi=1,0) : (inhi=0,1);
    for (int_t i = 0; i < mpts; i++) {
      if (gamma_values[i] <= gamma_values[ilo]) ilo = i;
        if (gamma_values[i] > gamma_values[ihi]) {
          inhi = ihi;
          ihi = i;
        } else if (gamma_values[i] > gamma_values[inhi] && i != ihi) inhi = i;
    }
    scalar_t rtol = 2.0*std::abs(gamma_values[ihi]-gamma_values[ilo])/(std::abs(gamma_values[ihi]) + std::abs(gamma_values[ilo]) + tiny_);
    if (rtol < tolerance_) {
      scalar_t dum = gamma_values[0];
      gamma_values[0] = gamma_values[ilo];
      gamma_values[ilo] = dum;
      for (int_t i = 0; i < num_dim_; i++) {
        scalar_t dum2 = (*points[0])[obj_->dof_map(i)];
        (*points[0])[obj_->dof_map(i)] = (*points[ilo])[obj_->dof_map(i)];
        (*points[ilo])[obj_->dof_map(i)] = dum2;
      }
      break;
    }
    nfunk += 2;

    scalar_t ytry,fac,fac1,fac2;

    fac = -1.0;
    fac1 = (1.0 - fac)/num_dim_;
    fac2 = fac1 - fac;

    for (int_t j = 0; j < num_dim_; j++)
      ptry[j] = points_column_sums[j]*fac1 - (*points[ihi])[obj_->dof_map(j)]*fac2;

    for(int_t n=0;n<num_dim_;++n)
      (*deformation)[obj_->dof_map(n)] = ptry[n];
    ytry = obj_->gamma(deformation);

    if (ytry < gamma_values[ihi]) {
      gamma_values[ihi] = ytry;
      for (int_t j = 0; j < num_dim_; j++) {
        points_column_sums[j] += ptry[j] - (*points[ihi])[obj_->dof_map(j)];
        (*points[ihi])[obj_->dof_map(j)] = ptry[j];
      }
    }
    if (ytry <= gamma_values[ilo]) {
      fac = 2.0;
      fac1 = (1.0 - fac)/num_dim_;
      fac2 = fac1 - fac;
      for (int_t j = 0; j < num_dim_; j++)
        ptry[j] = points_column_sums[j]*fac1 - (*points[ihi])[obj_->dof_map(j)]*fac2;

      for(int_t n=0;n<num_dim_;++n)
        (*deformation)[obj_->dof_map(n)] = ptry[n];
      ytry = obj_->gamma(deformation);

      if (ytry < gamma_values[ihi]) {
        gamma_values[ihi] = ytry;
        for (int_t j = 0; j < num_dim_; j++) {
          points_column_sums[j] += ptry[j] - (*points[ihi])[obj_->dof_map(j)];
          (*points[ihi])[obj_->dof_map(j)] = ptry[j];
        }
      }
    } else if (ytry >= gamma_values[inhi]) {
      ysave = gamma_values[ihi];
      fac = 0.5;
      fac1 = (1.0 - fac)/num_dim_;
      fac2 = fac1 - fac;
      for (int_t j = 0; j < num_dim_; j++)
        ptry[j] = points_column_sums[j]*fac1 - (*points[ihi])[obj_->dof_map(j)]*fac2;

      for(int_t n=0;n<num_dim_;++n)
        (*deformation)[obj_->dof_map(n)] = ptry[n];
      ytry = obj_->gamma(deformation);

      if (ytry < gamma_values[ihi]) {
        gamma_values[ihi] = ytry;
        for (int_t j = 0; j < num_dim_; j++) {
          points_column_sums[j] += ptry[j] - (*points[ihi])[obj_->dof_map(j)];
          (*points[ihi])[obj_->dof_map(j)] = ptry[j];
        }
      }
      if (ytry >= ysave) {
        for (int_t i = 0; i < mpts; i++) {
          if (i != ilo) {
            for (int_t j = 0; j < num_dim_; j++)
              (*points[i])[obj_->dof_map(j)] = points_column_sums[j] = 0.5*((*points[i])[obj_->dof_map(j)] + (*points[ilo])[obj_->dof_map(j)]);
            for(int_t n=0;n<num_dim_;++n)
              (*deformation)[obj_->dof_map(n)] = points_column_sums[n];
            gamma_values[i] = obj_->gamma(deformation);
          }
        }
        nfunk += num_dim_;

        for (int_t j = 0; j < num_dim_; j++) {
          scalar_t sum = 0.0;
          for (int_t i = 0; i < mpts; i++)
            sum += (*points[i])[obj_->dof_map(j)];
          points_column_sums[j] = sum;
        }
      }
    } else --nfunk;

#ifdef DICE_DEBUG_MSG
    std::cout << "Iteration " << iteration;
    for(int_t i=0;i<DICE_DEFORMATION_SIZE;++i) std::cout << " " << (*deformation)[i];
    std::cout << " nfunk: " << nfunk << " gamma: " << obj_->gamma(deformation) << " rtol: " << rtol << " tol: " << tolerance_ << std::endl;
#endif
  }

  delete [] gamma_values;
  delete [] points_column_sums;
  delete [] ptry;
  num_iterations = iteration;
  return CORRELATION_SUCCESSFUL;
}

}// End DICe Namespace
