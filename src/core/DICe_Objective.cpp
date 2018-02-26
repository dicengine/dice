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

#include <DICe_Objective.h>
#include <DICe_ImageUtils.h>
#include <DICe_Simplex.h>

#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <iostream>
#include <iomanip>

#include <cassert>

namespace DICe {

using namespace field_enums;

scalar_t
Objective::gamma( Teuchos::RCP<Local_Shape_Function> shape_function) const {
  try{
    subset_->initialize(schema_->def_img(subset_->sub_image_id()),DEF_INTENSITIES,shape_function,schema_->interpolation_method());
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
      gamma /= num_active_pixels==0.0?1.0:num_active_pixels;
  }
  return gamma;
}

scalar_t
Objective::beta(Teuchos::RCP<Local_Shape_Function> shape_function) const {
  // for now return -1 for beta if affine shape functions are used

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
  scalar_t temp_u=0.0,temp_v=0.0,temp_t=0.0;
  const scalar_t gamma_0 = gamma(shape_function);
  std::vector<scalar_t> dir_beta(3,0.0);
  Teuchos::RCP<Local_Shape_Function> temp_lsf = shape_function_factory(schema_);
  for(size_t i=0;i<3;++i){
    temp_lsf->clone(shape_function);
    temp_lsf->map_to_u_v_theta(subset_->centroid_x(),subset_->centroid_y(),temp_u,temp_v,temp_t);
    if(!schema_->rotation_enabled()&&i==2) continue;
    if(i==0){
      temp_u += epsilon[i];
    }else if(i==1){
      temp_v += epsilon[i];
    }else if(i==2){
      temp_t += epsilon[i];
    }
    temp_lsf->insert_motion(temp_u,temp_v,temp_t);
    const scalar_t gamma_p = gamma(temp_lsf);
    if(i==0){
      temp_u -= 2*epsilon[i];
    }else if(i==1){
      temp_v -= 2*epsilon[i];
    }else if(i==2){
      temp_t -= 2*epsilon[i];
    }
    temp_lsf->insert_motion(temp_u,temp_v,temp_t);
    // mod the def vector -
    const scalar_t gamma_m = gamma(temp_lsf);
    if(std::abs(gamma_m - gamma_0)<1.0E-10||std::abs(gamma_p - gamma_0)<1.0E-10){
      // abort because the slope is so bad that beta is infinite
      DEBUG_MSG("Objective::beta(): return value -1.0");
      // replace the correct normalization flag:
      schema_->set_normalize_gamma_with_active_pixels(original_normalize_flag);
      // re-initialize the subset with the original deformation solution
      subset_->initialize(schema_->def_img(subset_->sub_image_id()),DEF_INTENSITIES,shape_function,schema_->interpolation_method());
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
  DEBUG_MSG("Objective::beta(): return value " << mag_dir_beta);

  // replace the correct normalization flag:
  schema_->set_normalize_gamma_with_active_pixels(original_normalize_flag);

  // re-initialize the subset with the original deformation solution
  subset_->initialize(schema_->def_img(subset_->sub_image_id()),DEF_INTENSITIES,shape_function,schema_->interpolation_method());
  return mag_dir_beta;
}

scalar_t
Objective::sigma( Teuchos::RCP<Local_Shape_Function> shape_function,
  scalar_t & noise_level) const {
  // if the gradients don't exist:
  if(!subset_->has_gradients())
    return 0.0;

  // compute the noise std dev. of the image:
  noise_level = subset_->noise_std_dev(schema_->def_img(subset_->sub_image_id()),shape_function);
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
  DEBUG_MSG("Objective::sigma(): Subset " << correlation_point_global_id_ << " sigma: " << sigma);
  return sigma;
}

void
Objective::computeUncertaintyFields(Teuchos::RCP<Local_Shape_Function> shape_function){

  if(correlation_point_global_id_<0)return;

  scalar_t norm_ut_2 = 0.0;
  scalar_t norm_uhat_2 = 0.0;
  scalar_t norm_error_dot_gphi_2 = 0.0;
  scalar_t norm_error_dot_jgphi_2 = 0.0;
  scalar_t norm_ut_dot_gphi_2 = 0.0;
  scalar_t norm_uhat_dot_gphi_2 = 0.0;

  scalar_t int_r_total_2 = 0.0;
  //scalar_t int_r_exact_2 = 0.0;
  scalar_t int_uhat_dot_g = 0.0;
  scalar_t int_uhat_dot_jg = 0.0;

  scalar_t sssig = 1.0;

  scalar_t u = 0.0;
  scalar_t v = 0.0;
  scalar_t t = 0.0;
  const scalar_t cx = schema_->global_field_value(correlation_point_global_id_,SUBSET_COORDINATES_X_FS);
  const scalar_t cy = schema_->global_field_value(correlation_point_global_id_,SUBSET_COORDINATES_Y_FS);
  shape_function->map_to_u_v_theta(cx,cy,u,v,t);
  const bool has_image_deformer = schema_->image_deformer()!=Teuchos::null;
  // put the exact solution into the n minus 1 field in case it didn't get populated by an image deformer
  if(has_image_deformer){
    scalar_t exact_u = 0.0;
    scalar_t exact_v = 0.0;
    schema_->image_deformer()->compute_deformation(cx,cy,exact_u,exact_v);
    schema_->mesh()->get_field(DICe::field_enums::MODEL_DISPLACEMENT_X_FS)->global_value(correlation_point_global_id_) = exact_u;
    schema_->mesh()->get_field(DICe::field_enums::MODEL_DISPLACEMENT_Y_FS)->global_value(correlation_point_global_id_) = exact_v;
    // field 8: subset error at center in x direction
    schema_->mesh()->get_field(DICe::field_enums::FIELD_8_FS)->global_value(correlation_point_global_id_) = std::abs(exact_u - u);
    // field 9: subset error at center in x direction
    schema_->mesh()->get_field(DICe::field_enums::FIELD_9_FS)->global_value(correlation_point_global_id_) = std::abs(exact_v - v);
  }

  for(int_t i=0;i<subset_->num_pixels();++i){
    scalar_t x = subset_->x(i);
    scalar_t y = subset_->y(i);
    const int_t offset_x = schema_->ref_img()->offset_x();
    const int_t offset_y = schema_->ref_img()->offset_y();
    scalar_t bx = 0.0;
    scalar_t by = 0.0;
    if(has_image_deformer)
      schema_->image_deformer()->compute_deformation(x,y,bx,by); // x and y should not be offset since they are global pixel coordinates
    scalar_t gx = subset_->grad_x(i);
    scalar_t gy = subset_->grad_y(i);
    //scalar_t lap = schema_->ref_img()->laplacian(x-offset_x,y-offset_y);
    sssig += gx*gx + gy*gy;

    scalar_t mag_ut_2 = bx*bx + by*by;
    //scalar_t one_over_mag_ut_2 = mag_ut_2 == 0.0 ? 0.0 : 1.0/mag_ut_2;
    norm_ut_2 += mag_ut_2;
    scalar_t mag_uhat_2 = u*u + v*v;
    //scalar_t one_over_mag_uhat_2 = mag_uhat_2 == 0.0 ? 0.0 : 1.0/mag_uhat_2;
    norm_uhat_2 += mag_uhat_2;
    //norm_error_2 += (bx - u)*(bx - u) + (by - v)*(by - v);
    scalar_t one_over_mag_grad_phi_2 = gx*gx + gy*gy < 1.0E-4 ? 0.0: 1.0 / (gx*gx + gy*gy);
    norm_ut_dot_gphi_2 += (bx*gx + by*gy)*(bx*gx + by*gy)*one_over_mag_grad_phi_2;
    //norm_ut_dot_jgphi_2 += (bx*-1.0*gy + by*gx)*(bx*-1.0*gy + by*gx)*one_over_mag_grad_phi_2;
    norm_uhat_dot_gphi_2 += (u*gx + v*gy)*(u*gx + v*gy)*one_over_mag_grad_phi_2;
    norm_error_dot_gphi_2 += ((u-bx)*gx + (v-by)*gy)*((u-bx)*gx + (v-by)*gy)*one_over_mag_grad_phi_2;
    norm_error_dot_jgphi_2 += ((v-by)*gx - (u-bx)*gy)*((v-by)*gx - (u-bx)*gy)*one_over_mag_grad_phi_2;

    scalar_t sub_r = schema_->def_img()->interpolate_keys_fourth(x - offset_x + u,y - offset_y + v) - (*schema_->ref_img())((int_t)(x-offset_x),(int_t)(y-offset_y));
    //int_r += sig*sig*lap*lap*one_over_mag_grad_phi_2;
    //scalar_t taylor = (*schema_->def_img())((int_t)(x-offset_x),(int_t)(y-offset_y)) - (*schema_->ref_img())((int_t)(x-offset_x),(int_t)(y-offset_y)) + u*gx + v*gy;
    //int_sub_r += sub_r*sub_r*one_over_mag_grad_phi_2;
    scalar_t sub_r_exact = schema_->def_img()->interpolate_keys_fourth(x-offset_x + bx,y-offset_y + by) - (*schema_->ref_img())((int_t)(x-offset_x),(int_t)(y-offset_y));
    //int_r_exact_2 += sub_r_exact*sub_r_exact*one_over_mag_grad_phi_2;
    int_r_total_2 += (sub_r - sub_r_exact)*(sub_r - sub_r_exact)*one_over_mag_grad_phi_2;
    int_uhat_dot_g += (u*gx + v*gy)/std::sqrt(gx*gx + gy*gy);
    int_uhat_dot_jg += (-1.0*u*gy + v*gx);
  }
  sssig /= subset_->num_pixels()==0.0?1.0:subset_->num_pixels();

  // populate the fields:
  // field 1: cos of angle between uhat and grad phi
  const scalar_t cos_theta_hat = norm_uhat_2 == 0.0 ? 1.0 : std::sqrt(norm_uhat_dot_gphi_2/norm_uhat_2);
  schema_->mesh()->get_field(DICe::field_enums::FIELD_1_FS)->global_value(correlation_point_global_id_) = cos_theta_hat;
  // field 2: cos of angle between the true motion and grad phi
  const scalar_t cos_theta = norm_ut_2 == 0.0 ? 1.0 : std::sqrt(norm_ut_dot_gphi_2/norm_ut_2);
  schema_->mesh()->get_field(DICe::field_enums::FIELD_2_FS)->global_value(correlation_point_global_id_) = cos_theta;
  // field 3: the total L2 error magnitude:
  schema_->mesh()->get_field(DICe::field_enums::FIELD_3_FS)->global_value(correlation_point_global_id_) = std::sqrt(norm_error_dot_gphi_2 + norm_error_dot_jgphi_2);
  // field 4: residual based exact error estimate in the direction of grad phi
  schema_->mesh()->get_field(DICe::field_enums::FIELD_4_FS)->global_value(correlation_point_global_id_) = cos_theta_hat==0.0?0.0:1.0/cos_theta_hat * std::sqrt(int_r_total_2);
  // field 5: the L2 error mag in direction of grad phi:
  schema_->mesh()->get_field(DICe::field_enums::FIELD_5_FS)->global_value(correlation_point_global_id_) = std::sqrt(norm_error_dot_gphi_2);
  // field 6: residual based exact error estimate in the direction of grad phi
  schema_->mesh()->get_field(DICe::field_enums::FIELD_6_FS)->global_value(correlation_point_global_id_) = std::sqrt(int_r_total_2);
  // field 7: SSSIG field
  schema_->mesh()->get_field(DICe::field_enums::FIELD_7_FS)->global_value(correlation_point_global_id_) = sssig;
// field 8: cos of angle between the error and grad phi
//  schema_->mesh()->get_field(DICe::field_enums::FIELD_8_FS)->global_value(correlation_point_global_id_) = std::sqrt(norm_error_dot_gphi_2/norm_error_2);
//  // field 9: cos of angle between ut and uhat
//  schema_->mesh()->get_field(DICe::field_enums::FIELD_9_FS)->global_value(correlation_point_global_id_) = std::sqrt(norm_ut_dot_uhat_2/norm_ut_2);
//  // field 10: cos of angle between ut and grad_phi
//  schema_->mesh()->get_field(DICe::field_enums::FIELD_10_FS)->global_value(correlation_point_global_id_) = std::sqrt(int_r_total_2);
//  // field 10: cos of angle between error and ut
//  schema_->mesh()->get_field(DICe::field_enums::FIELD_10_FS)->global_value(correlation_point_global_id_) = std::sqrt(int_sub_r_approx);
}

Status_Flag
Objective::computeUpdateRobust(Teuchos::RCP<Local_Shape_Function> shape_function,
  int_t & num_iterations,
  const scalar_t & override_tol){

  const scalar_t skip_threshold = override_tol==-1 ? schema_->skip_solve_gamma_threshold() : override_tol;

  Status_Flag status_flag;
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::max_iterations,  schema_->max_solver_iterations_robust());
  params->set(DICe::tolerance, schema_->robust_solver_tolerance());
  DICe::Subset_Simplex simplex(this,params);
  try{
    status_flag = simplex.minimize(shape_function,num_iterations,skip_threshold);
  }
  catch (std::logic_error &err) {
    return CORRELATION_FAILED;
  }
  return status_flag;
}

Status_Flag
Objective_ZNSSD::computeUpdateFast(Teuchos::RCP<Local_Shape_Function> shape_function,
  int_t & num_iterations){
  TEUCHOS_TEST_FOR_EXCEPTION(!subset_->has_gradients(),std::runtime_error,"Error, image gradients have not been computed but are needed here.");
  // TODO catch the case where the initial gamma is good enough (possibly do this at the image level, not subset?):
  int_t N = shape_function->num_params(); // one degree of freedom for each shape function parameter
  assert(N>=2);
  scalar_t tolerance = schema_->fast_solver_tolerance();
  const int_t max_solve_its = schema_->max_solver_iterations_fast();
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  // using type double here because LAPACK doesn't support float.
  double *WORK = new double[LWORK];
  Teuchos::LAPACK<int_t,double> lapack;

  // Initialize storage:
  Teuchos::SerialDenseMatrix<int_t,double> H(N,N, true);
  Teuchos::ArrayRCP<double> q(N,0.0);
  std::vector<scalar_t> residuals(N,0.0);
  std::vector<scalar_t> def_old(N,0.0);    // save off the previous value to test for convergence
  std::vector<scalar_t> def_update(N,0.0); // save off the previous value to test for convergence

  // note this creates a pointer to the array so
  // the values are updated each frame if compute_grad_def_images is on
  Teuchos::ArrayRCP<scalar_t> gradGx = subset_->grad_x_array();
  Teuchos::ArrayRCP<scalar_t> gradGy = subset_->grad_y_array();
  const scalar_t cx = subset_->centroid_x();
  const scalar_t cy = subset_->centroid_y();
  const scalar_t meanF = subset_->mean(REF_INTENSITIES);

  scalar_t old_u=0.0,old_v=0.0,old_t=0.0;
  shape_function->map_to_u_v_theta(cx,cy,old_u,old_v,old_t);
  DEBUG_MSG(std::setw(5) << "Iter" <<
    std::setw(12) << " u"  <<
    std::setw(12) << " du" <<
    std::setw(12) << " v"  <<
    std::setw(12) << " dv" <<
    std::setw(12) << " t"  <<
    std::setw(12) << " dt");

  int_t solve_it = 0;
  for(;solve_it<=max_solve_its;++solve_it){
    num_iterations = solve_it;

    // update the deformed image with the new deformation:
    try{
      subset_->initialize(schema_->def_img(subset_->sub_image_id()),DEF_INTENSITIES,shape_function,schema_->interpolation_method());
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
    // the gradients are taken from the def images rather than the ref
    const bool use_ref_grads = schema_->def_img()->has_gradients() ? false : true;

    scalar_t GmF = 0.0;
    for(int_t index=0;index<subset_->num_pixels();++index){
      if(subset_->is_deactivated_this_step(index)||!subset_->is_active(index)) continue;
      GmF = (subset_->def_intensities(index) - meanG) - (subset_->ref_intensities(index) - meanF);
      for(int_t i=0;i<N;++i)
        residuals[i] = 0.0;
      shape_function->residuals(subset_->x(index),subset_->y(index),cx,cy,gradGx[index],gradGy[index],residuals,use_ref_grads);
      for(int_t i=0;i<N;++i){
        q[i] += GmF*residuals[i];
        for(int_t j=0;j<N;++j)
          H(i,j) += residuals[i]*residuals[j];
      }
    }

    if(schema_->use_objective_regularization()){ // TODO test for affine shape functions too
      // add the penalty terms
      const scalar_t alpha = schema_->levenberg_marquardt_regularization_factor();
      H(0,0) += alpha;
      H(1,1) += alpha;
    }

    // compute the norm of H prior to taking the inverse:
    // Note: for this to work, the shape functions must always have their displacement degrees of freedom as the
    // first two parameters (assert that N>=2 above)
    const scalar_t det_h = H(0,0)*H(1,1) - H(1,0)*H(0,1);
    const scalar_t norm_H = std::sqrt(H(0,0)*H(0,0) + H(0,1)*H(0,1) + H(1,0)*H(1,0) + H(1,1)*H(1,1));
    scalar_t cond_2x2 = -1.0;
    if(det_h !=0.0){
      const scalar_t norm_Hi = det_h==0.0?0.0:std::sqrt((1.0/(det_h*det_h))*(H(0,0)*H(0,0) + H(0,1)*H(0,1) + H(1,0)*H(1,0) + H(1,1)*H(1,1)));
      cond_2x2 = norm_H * norm_Hi;
    }

    // clear temp storage
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    for(int_t i=0;i<N+1;++i) {IPIV[i] = 0;}
    try
    {
      lapack.GETRF(N,N,H.values(),N,IPIV,&INFO);
      if(correlation_point_global_id_>=0)
        schema_->global_field_value(correlation_point_global_id_,CONDITION_NUMBER_FS) = cond_2x2;
      if(cond_2x2 > 1.0E12) return HESSIAN_SINGULAR;
    }
    catch(std::exception &e){
      if(&e!=0){
        DEBUG_MSG( e.what() << '\n');
      }
      return LINEAR_SOLVE_FAILED;
    }
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    try
    {
      lapack.GETRI(N,H.values(),N,IPIV,WORK,LWORK,&INFO);
    }
    catch(std::exception &e){
      if(&e!=0){
        DEBUG_MSG( e.what() << '\n');
      }
      return LINEAR_SOLVE_FAILED;
    }
    // save off last step
    for(int_t i=0;i<N;++i)
      def_old[i] = (*shape_function)(i);
    for(int_t i=0;i<N;++i)
      def_update[i] = 0.0;
    for(int_t i=0;i<N;++i)
      for(int_t j=0;j<N;++j)
        def_update[i] += H(i,j)*(-1.0)*q[j];
    shape_function->update(def_update);

    scalar_t guess_u = 0.0,guess_v=0.0,guess_t=0.0;
    shape_function->map_to_u_v_theta(cx,cy,guess_u,guess_v,guess_t);
    std::ios  state(NULL);
    state.copyfmt(std::cout);
    DEBUG_MSG(std::setw(5) << solve_it <<
      std::setw(12) << std::scientific << std::setprecision(4) << guess_u <<
      std::setw(12) << guess_u - old_u <<
      std::setw(12) << guess_v <<
      std::setw(12) << guess_v - old_v <<
      std::setw(12) << guess_t <<
      std::setw(12) << guess_t - old_t);
    std::cout.copyfmt(state);
    shape_function->map_to_u_v_theta(cx,cy,old_u,old_v,old_t);

    const bool converged = shape_function->test_for_convergence(def_old,tolerance);
    if(converged){
      DEBUG_MSG("Subset " << correlation_point_global_id_ << " ** CONVERGED SOLUTION ");
      shape_function->print_parameters();
      computeUncertaintyFields(shape_function);
      break;
    }

    // zero out the storage
    H.putScalar(0.0);
    for(int_t i=0;i<N;++i)
      q[i] = 0.0;
  } // end solve iteration loop

  // clean up storage for lapack:
  delete [] WORK;
  delete [] IPIV;

  if(solve_it>max_solve_its){
    return MAX_ITERATIONS_REACHED;
  }
  else return CORRELATION_SUCCESSFUL;
}

}// End DICe Namespace
