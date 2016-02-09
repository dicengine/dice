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

#include <DICe_ObjectiveZNSSD.h>

#include <DICe_Simplex.h>

#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <iostream>
#include <iomanip>

#include <cassert>

namespace DICe {

scalar_t
Objective_ZNSSD::gamma( Teuchos::RCP<std::vector<scalar_t> > &deformation) const {

  assert(deformation->size()==DICE_DEFORMATION_SIZE);
  try{
    subset_->initialize(schema_->def_img(),DEF_INTENSITIES,deformation,schema_->interpolation_method());
  }
  catch (std::logic_error & err) {
    return -1.0;
  }
  scalar_t gamma = subset_->gamma();
  if(schema_->normalize_gamma_with_active_pixels()){
    int_t num_active_pixels = 0;
    for(int_t i=0;i<subset_->num_pixels();++i)
      if(subset_->is_active(i)) num_active_pixels++;
    if(num_active_pixels > 0)
      gamma /= num_active_pixels;
  }
  return gamma;
}

scalar_t
Objective_ZNSSD::beta(Teuchos::RCP<std::vector<scalar_t> > &deformation) const {
  // for beta we don't want the gamma values normalized by the number of pixels:
  const bool original_normalize_flag = schema_->normalize_gamma_with_active_pixels();
  schema_->set_normalize_gamma_with_active_pixels(false);
  std::vector<scalar_t> epsilon(3);
  epsilon[0] = 1.0E-1;
  epsilon[1] = 1.0E-1;
  epsilon[2] = 1.0E-1;
  std::vector<scalar_t> factor(3);
  factor[0] = 1.0E-3;
  factor[1] = 1.0E-3;
  factor[2] = 1.0E-1;
  std::vector<DICe::Field_Name> fields(3);
  fields[0] = DISPLACEMENT_X;
  fields[1] = DISPLACEMENT_Y;
  fields[2] = ROTATION_Z;
  const scalar_t gamma_0 = gamma(deformation);
  std::vector<scalar_t> dir_beta(3,0.0);
  Teuchos::RCP<std::vector<scalar_t> > temp_def = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE));
  for(size_t i=0;i<fields.size();++i){
    if(!schema_->rotation_enabled()&&i==2) continue;
    // reset def vector
    for(size_t j=0;j<DICE_DEFORMATION_SIZE;++j)
      (*temp_def)[j] = (*deformation)[j];
    // mod the def vector +
    (*temp_def)[fields[i]] += epsilon[i];
    const scalar_t gamma_p = gamma(temp_def);
    // mod the def vector -
    (*temp_def)[fields[i]] -= 2.0*epsilon[i];
    const scalar_t gamma_m = gamma(temp_def);

    if(std::abs(gamma_m - gamma_0)<1.0E-10||std::abs(gamma_p - gamma_0)<1.0E-10){
      // abort because the slope is so bad that beta is infinite
      DEBUG_MSG("Objective_ZNSSD::beta(): return value -1.0");
      // replace the correct normalization flag:
      schema_->set_normalize_gamma_with_active_pixels(original_normalize_flag);
      // re-initialize the subset with the original deformation solution
      subset_->initialize(schema_->def_img(),DEF_INTENSITIES,deformation,schema_->interpolation_method());
      return -1.0;
    }
    const scalar_t slope_m = std::abs(epsilon[i] / (gamma_m - gamma_0))*factor[i];
    const scalar_t slope_p = std::abs(epsilon[i] / (gamma_p - gamma_0))*factor[i];
    dir_beta[i] = (slope_m + slope_p)/2.0;
    //DEBUG_MSG("Simplex method dir_beta " << i << ": " << std::sqrt(dir_beta[i]*dir_beta[i]) << " gamma_p: " << gamma_p << " gamma_m: " << gamma_m);
  }
  scalar_t mag_dir_beta = 0.0;
  for(size_t i=0;i<dir_beta.size();++i){
    if(!schema_->rotation_enabled()&&i==2) continue;
    mag_dir_beta += dir_beta[i]*dir_beta[i];
  }
  mag_dir_beta = std::sqrt(mag_dir_beta);
  DEBUG_MSG("Objective_ZNSSD::beta(): return value " << mag_dir_beta);

  // replace the correct normalization flag:
  schema_->set_normalize_gamma_with_active_pixels(original_normalize_flag);

  // re-initialize the subset with the original deformation solution
  subset_->initialize(schema_->def_img(),DEF_INTENSITIES,deformation,schema_->interpolation_method());
  return mag_dir_beta;
}

scalar_t
Objective_ZNSSD::sigma( Teuchos::RCP<std::vector<scalar_t> > &deformation,
  scalar_t & noise_level) const {
  // if the gradients don't exist:
  if(!subset_->has_gradients())
    return 0.0;

  // compute the noise std dev. of the image:
  noise_level = subset_->noise_std_dev(schema_->def_img(),deformation);
  // sum up the grads in x and y:
  scalar_t sum_gx = 0.0;
  scalar_t sum_gy = 0.0;
  for(int_t i=0;i<subset_->num_pixels();++i){
    if(!subset_->is_active(i) || subset_->is_deactivated_this_step(i)) continue;
    sum_gx += subset_->grad_x(i)*subset_->grad_x(i);
    sum_gy += subset_->grad_y(i)*subset_->grad_y(i);
  }
  const scalar_t sum_grad = sum_gx > sum_gy ? sum_gy : sum_gx;
  // ensure that sum grad is greater than zero
  const scalar_t sigma = sum_grad>0.0 ? std::sqrt(2.0*noise_level*noise_level / sum_grad) : -1.0;
  DEBUG_MSG("Objective_ZNSSD::sigma(): Subset " << correlation_point_global_id_ << " sigma: " << sigma);
  return sigma;
}

Status_Flag
Objective_ZNSSD::computeUpdateRobust(Teuchos::RCP<std::vector<scalar_t> > & deformation,
  int_t & num_iterations,
  const scalar_t & override_tol){

  const scalar_t skip_threshold = override_tol==-1 ? schema_->skip_solve_gamma_threshold() : override_tol;

  Status_Flag status_flag;
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::max_iterations,  schema_->max_solver_iterations_robust());
  params->set(DICe::tolerance, schema_->robust_solver_tolerance());
  DICe::Simplex simplex(this,params);

  Teuchos::RCP<std::vector<scalar_t> > deltas = Teuchos::rcp(new std::vector<scalar_t>(num_dofs(),0.0));
  for(int_t i=0;i<num_dofs();++i){
    if(i<2) (*deltas)[i] = schema_->robust_delta_disp();
    else
      (*deltas)[i] = schema_->robust_delta_theta();
  }
  try{
    status_flag = simplex.minimize(deformation,deltas,num_iterations,skip_threshold);
  }
  catch (std::logic_error &err) {
    return CORRELATION_FAILED;
  }
  return status_flag;
}

Status_Flag
Objective_ZNSSD::computeUpdateFast(Teuchos::RCP<std::vector<scalar_t> > & deformation,
  int_t & num_iterations){
  assert(deformation->size()==DICE_DEFORMATION_SIZE);

  TEUCHOS_TEST_FOR_EXCEPTION(!subset_->has_gradients(),std::runtime_error,"Error, image gradients have not been computed but are needed here.");

  // TODO catch the case where the initial gamma is good enough (possibly do this at the image level, not subset?):

  // using type double here a lot because LAPACK doesn't support float.
  int_t N = 6; // [ u_x u_y theta dudx dvdy gxy ]
  scalar_t solve_tol_disp = schema_->fast_solver_tolerance();
  scalar_t solve_tol_theta = schema_->fast_solver_tolerance();
  const int_t max_solve_its = schema_->max_solver_iterations_fast();
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  double *GWORK = new double[10*N];
  int *IWORK = new int[LWORK];
  Teuchos::LAPACK<int_t,double> lapack;
  assert(N==6 && "  DICe ERROR: this DIC method is currently only approprate for 6 variables.");

  // Initialize storage:
  Teuchos::SerialDenseMatrix<int_t,double> H(N,N, true);
  Teuchos::ArrayRCP<double> q(N,0.0);
  Teuchos::RCP<std::vector<scalar_t> > def_old    = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0)); // save off the previous value to test for convergence
  Teuchos::RCP<std::vector<scalar_t> > def_update = Teuchos::rcp(new std::vector<scalar_t>(N,0.0)); // save off the previous value to test for convergence

  Teuchos::ArrayRCP<scalar_t> gradGx = subset_->grad_x_array();
  Teuchos::ArrayRCP<scalar_t> gradGy = subset_->grad_y_array();
  const scalar_t cx = subset_->centroid_x();
  const scalar_t cy = subset_->centroid_y();
  const scalar_t meanF = subset_->mean(REF_INTENSITIES);
  // these are used for regularization below
  const scalar_t prev_u = (*deformation)[DISPLACEMENT_X];
  const scalar_t prev_v = (*deformation)[DISPLACEMENT_Y];
  //const scalar_t prev_theta = (*deformation)[ROTATION_Z];

  // SOLVER ---------------------------------------------------------

  int_t solve_it = 0;
  for(;solve_it<=max_solve_its;++solve_it)
  {
    num_iterations = solve_it;
    // update the deformed image with the new deformation:
    try{
      subset_->initialize(schema_->def_img(),DEF_INTENSITIES,deformation,schema_->interpolation_method());
      //#ifdef DICE_DEBUG_MSG
      //    std::stringstream fileName;
      //    fileName << "defSubset_" << correlation_point_global_id_ << "_" << solve_it;
      //    def_subset_->write(fileName.str());
      //#endif
    }
    catch (std::logic_error & err) {
      return SUBSET_CONSTRUCTION_FAILED;
    }

    // compute the mean value of the subsets:
    const scalar_t meanG = subset_->mean(DEF_INTENSITIES);

    scalar_t dx=0.0,dy=0.0,Dx=0.0,Dy=0.0,delTheta=0.0,delEx=0.0,delEy=0.0,delGxy=0.0;
    scalar_t Gx=0.0,Gy=0.0, GmF=0.0;
    const scalar_t theta = (*deformation)[DICe::ROTATION_Z];
    const scalar_t dudx = (*deformation)[DICe::NORMAL_STRAIN_X];
    const scalar_t dvdy = (*deformation)[DICe::NORMAL_STRAIN_Y];
    const scalar_t gxy = (*deformation)[DICe::SHEAR_STRAIN_XY];
    const scalar_t cosTheta = std::cos(theta);
    const scalar_t sinTheta = std::sin(theta);

    for(int_t index=0;index<subset_->num_pixels();++index){
      if(subset_->is_deactivated_this_step(index)||!subset_->is_active(index)) continue;
      dx = subset_->x(index) - cx;
      dy = subset_->y(index) - cy;
      Dx = (1.0+dudx)*(dx) + gxy*(dy);
      Dy = (1.0+dvdy)*(dy) + gxy*(dx);
      GmF = (subset_->def_intensities(index) - meanG) - (subset_->ref_intensities(index) - meanF);
      Gx = gradGx[index];
      Gy = gradGy[index];
      delTheta = Gx*(-sinTheta*Dx - cosTheta*Dy) + Gy*(cosTheta*Dx - sinTheta*Dy);
      delEx = Gx*dx*cosTheta + Gy*dx*sinTheta;
      delEy = -Gx*dy*sinTheta + Gy*dy*cosTheta;
      delGxy = Gx*(cosTheta*dy - sinTheta*dx) + Gy*(sinTheta*dy + cosTheta*dx);

      q[0] += Gx*GmF;
      q[1] += Gy*GmF;
      q[2] += delTheta*GmF;
      q[3] += delEx*GmF;
      q[4] += delEy*GmF;
      q[5] += delGxy*GmF;

      H(0,0) += Gx*Gx;
      H(1,0) += Gy*Gx;
      H(2,0) += delTheta*Gx;
      H(3,0) += delEx*Gx;
      H(4,0) += delEy*Gx;
      H(5,0) += delGxy*Gx;

      H(0,1) += Gx*Gy;
      H(1,1) += Gy*Gy;
      H(2,1) += delTheta*Gy;
      H(3,1) += delEx*Gy;
      H(4,1) += delEy*Gy;
      H(5,1) += delGxy*Gy;

      H(0,2) += Gx*delTheta;
      H(1,2) += Gy*delTheta;
      H(2,2) += delTheta*delTheta;
      H(3,2) += delEx*delTheta;
      H(4,2) += delEy*delTheta;
      H(5,2) += delGxy*delTheta;

      H(0,3) += Gx*delEx;
      H(1,3) += Gy*delEx;
      H(2,3) += delTheta*delEx;
      H(3,3) += delEx*delEx;
      H(4,3) += delEy*delEx;
      H(5,3) += delGxy*delEx;

      H(0,4) += Gx*delEy;
      H(1,4) += Gy*delEy;
      H(2,4) += delTheta*delEy;
      H(3,4) += delEx*delEy;
      H(4,4) += delEy*delEy;
      H(5,4) += delGxy*delEy;

      H(0,5) += Gx*delGxy;
      H(1,5) += Gy*delGxy;
      H(2,5) += delTheta*delGxy;
      H(3,5) += delEx*delGxy;
      H(4,5) += delEy*delGxy;
      H(5,5) += delGxy*delGxy;
    }

    if(schema_->use_objective_regularization()){
      // add the penalty terms
      const scalar_t alpha = schema_->objective_regularization_factor();
      q[0] += alpha * ((*deformation)[DICe::DISPLACEMENT_X] - prev_u);
      q[1] += alpha * ((*deformation)[DICe::DISPLACEMENT_Y] - prev_v);
      //q[2] += alpha * ((*deformation)[DICe::ROTATION_Z] - prev_theta);
      H(0,0) += alpha;
      H(1,1) += alpha;
      // H(2,2) += alpha;
    }

    // determine the max value in the matrix:
    scalar_t maxH = 0.0;
    for(int_t i=0;i<H.numCols();++i)
      for(int_t j=0;j<H.numRows();++j)
        if(std::abs(H(i,j))>maxH) maxH = std::abs(H(i,j));

    // add ones for rows and columns of inactive shape functions
    if(!schema_->translation_enabled()){
      for(int_t i=0;i<N;++i){
        H(0,i) = 0.0; H(1,i) = 0.0;
        H(i,0) = 0.0; H(i,1) = 0.0;
      }
      H(0,0) = 1.0 * maxH;
      H(1,1) = 1.0 * maxH;
      q[0] = 0.0;
      q[1] = 0.0;
    }
    if(!schema_->rotation_enabled()){
      for(int_t i=0;i<N;++i){
        H(2,i) = 0.0; H(i,2) = 0.0;
      }
      H(2,2) = 1.0 * maxH;
      q[2] = 0.0;
    }
    if(!schema_->shear_strain_enabled()){
      for(int_t i=0;i<N;++i){
        H(5,i) = 0.0; H(i,5) = 0.0;
      }
      H(5,5) = 1.0 * maxH;
      q[5] = 0.0;
    }
    if(!schema_->normal_strain_enabled()){
      for(int_t i=0;i<N;++i){
        H(3,i) = 0.0; H(4,i) = 0.0;
        H(i,3) = 0.0; H(i,4) = 0.0;
      }
      H(3,3) = 1.0 * maxH;
      H(4,4) = 1.0 * maxH;
      q[3] = 0.0;
      q[4] = 0.0;
    }

    // TODO: remove for performance?
    // compute the 1-norm of H:
    std::vector<scalar_t> colTotals(N,0.0);
    for(int_t i=0;i<H.numCols();++i){
      for(int_t j=0;j<H.numRows();++j){
        colTotals[i]+=std::abs(H(j,i));
      }
    }
    double anorm = 0.0;
    for(int_t i=0;i<N;++i){
      if(colTotals[i] > anorm) anorm = colTotals[i];
    }

    // clear temp storage
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    for(int_t i=0;i<10*N;++i) GWORK[i] = 0.0;
    for(int_t i=0;i<LWORK;++i) IWORK[i] = 0;
    for(int_t i=0;i<N+1;++i) {IPIV[i] = 0;}
    double rcond=0.0; // reciporical condition number
    try
    {
      lapack.GETRF(N,N,H.values(),N,IPIV,&INFO);
      lapack.GECON('1',N,H.values(),N,anorm,&rcond,GWORK,IWORK,&INFO);
      DEBUG_MSG("Subset " << correlation_point_global_id_ << "    RCOND(H): "<< rcond);
      schema_->local_field_value(correlation_point_global_id_,DICe::CONDITION_NUMBER) = (rcond !=0.0) ? 1.0/rcond : 0.0;
      if(rcond < 1.0E-12) return HESSIAN_SINGULAR;
    }
    catch(std::exception &e){
      DEBUG_MSG( e.what() << '\n');
      return LINEAR_SOLVE_FAILED;
    }
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    try
    {
      lapack.GETRI(N,H.values(),N,IPIV,WORK,LWORK,&INFO);
    }
    catch(std::exception &e){
      DEBUG_MSG( e.what() << '\n');
      return LINEAR_SOLVE_FAILED;
    }


    // save off last step d
    for(int_t i=0;i<DICE_DEFORMATION_SIZE;++i)
      (*def_old)[i] = (*deformation)[i];

    for(int_t i=0;i<N;++i)
      (*def_update)[i] = 0.0;

    for(int_t i=0;i<N;++i)
      for(int_t j=0;j<N;++j)
        (*def_update)[i] += H(i,j)*(-1.0)*q[j];

    DEBUG_MSG("    Iterative updates: u " << (*def_update)[0] << " v " << (*def_update)[1] << " theta " <<
      (*def_update)[2] << " ex " << (*def_update)[3] << " ey " << (*def_update)[4] << " gxy " << (*def_update)[5]);

    (*deformation)[DICe::DISPLACEMENT_X] += (*def_update)[0];
    (*deformation)[DICe::DISPLACEMENT_Y] += (*def_update)[1];
    (*deformation)[DICe::ROTATION_Z] += (*def_update)[2];
    (*deformation)[DICe::NORMAL_STRAIN_X] += (*def_update)[3];
    (*deformation)[DICe::NORMAL_STRAIN_Y] += (*def_update)[4];
    (*deformation)[DICe::SHEAR_STRAIN_XY] += (*def_update)[5];

    DEBUG_MSG("Subset " << correlation_point_global_id_ << " -- iteration: " << solve_it << " u " << (*deformation)[DICe::DISPLACEMENT_X] <<
      " v " << (*deformation)[DICe::DISPLACEMENT_Y] << " theta " << (*deformation)[DICe::ROTATION_Z] <<
      " ex " << (*deformation)[DICe::NORMAL_STRAIN_X] << " ey " << (*deformation)[DICe::NORMAL_STRAIN_Y] <<
      " gxy " << (*deformation)[DICe::SHEAR_STRAIN_XY] <<
      " residual: (" << q[0] << "," << q[1] << "," << q[2] << ")");

    if(std::abs((*deformation)[DICe::DISPLACEMENT_X] - (*def_old)[DICe::DISPLACEMENT_X]) < solve_tol_disp
        && std::abs((*deformation)[DICe::DISPLACEMENT_Y] - (*def_old)[DICe::DISPLACEMENT_Y]) < solve_tol_disp
        && std::abs((*deformation)[DICe::ROTATION_Z] - (*def_old)[DICe::ROTATION_Z]) < solve_tol_theta){
      DEBUG_MSG("Subset " << correlation_point_global_id_ << " ** CONVERGED SOLUTION, u " << (*deformation)[DICe::DISPLACEMENT_X] <<
        " v " << (*deformation)[DICe::DISPLACEMENT_Y] <<
        " theta " << (*deformation)[DICe::ROTATION_Z]  << " ex " << (*deformation)[DICe::NORMAL_STRAIN_X] <<
        " ey " << (*deformation)[DICe::NORMAL_STRAIN_Y] << " gxy " << (*deformation)[DICe::SHEAR_STRAIN_XY]);
      break;
    }

    // zero out the storage
    H.putScalar(0.0);
    for(int_t i=0;i<N;++i)
      q[i] = 0.0;
  }

  // clean up storage for lapack:
  delete [] WORK;
  delete [] GWORK;
  delete [] IWORK;
  delete [] IPIV;

  if(solve_it>max_solve_its){
    return MAX_ITERATIONS_REACHED;
  }
  else return CORRELATION_SUCCESSFUL;
}

}// End DICe Namespace
