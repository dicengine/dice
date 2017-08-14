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

using namespace mesh::field_enums;

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Objective> objective_factory(Schema * schema,
  const int_t correlation_point_global_id){
  if(schema->affine_matrix_enabled())
    return Teuchos::rcp(new Objective_ZNSSD_Affine(schema,correlation_point_global_id));
  else
    return Teuchos::rcp(new Objective_ZNSSD(schema,correlation_point_global_id));
}


scalar_t
Objective::gamma( Teuchos::RCP<std::vector<scalar_t> > deformation) const {
  try{
    subset_->initialize(schema_->def_img(subset_->sub_image_id()),DEF_INTENSITIES,deformation,schema_->interpolation_method());
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
Objective::beta(Teuchos::RCP<std::vector<scalar_t> > deformation) const {
  // for now return -1 for beta if affine shape functions are used
  // TODO hook beta up for affine shape functions
  if(deformation->size()!=DICE_DEFORMATION_SIZE) return -1.0;

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
  std::vector<Affine_Dof> fields(3);
  fields[0] = DOF_U;
  fields[1] = DOF_V;
  fields[2] = DOF_THETA;
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
      DEBUG_MSG("Objective::beta(): return value -1.0");
      // replace the correct normalization flag:
      schema_->set_normalize_gamma_with_active_pixels(original_normalize_flag);
      // re-initialize the subset with the original deformation solution
      subset_->initialize(schema_->def_img(subset_->sub_image_id()),DEF_INTENSITIES,deformation,schema_->interpolation_method());
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
  subset_->initialize(schema_->def_img(subset_->sub_image_id()),DEF_INTENSITIES,deformation,schema_->interpolation_method());
  return mag_dir_beta;
}

scalar_t
Objective::sigma( Teuchos::RCP<std::vector<scalar_t> > deformation,
  scalar_t & noise_level) const {
  // if the gradients don't exist:
  if(!subset_->has_gradients())
    return 0.0;

  // compute the noise std dev. of the image:
  noise_level = subset_->noise_std_dev(schema_->def_img(subset_->sub_image_id()),deformation);
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
Objective::computeUncertaintyFields(Teuchos::RCP<std::vector<scalar_t> > deformation){

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
  scalar_t x_prime = 0.0;
  scalar_t y_prime = 0.0;
  const scalar_t cx = schema_->global_field_value(correlation_point_global_id_,SUBSET_COORDINATES_X_FS);
  const scalar_t cy = schema_->global_field_value(correlation_point_global_id_,SUBSET_COORDINATES_Y_FS);
  if(deformation->size()==DICE_DEFORMATION_SIZE){
    u = (*deformation)[DOF_U];
    v = (*deformation)[DOF_V];
  }else if(deformation->size()==DICE_DEFORMATION_SIZE_AFFINE){
    TEUCHOS_TEST_FOR_EXCEPTION((*deformation)[8]==0.0,std::runtime_error,"");
    map_affine(cx,cy,x_prime,y_prime,deformation);
    u = x_prime - cx;
    v = y_prime - cy;
  }else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, unknown deformation vector size.");
  }

  const bool has_image_deformer = schema_->image_deformer()!=Teuchos::null;

  // put the exact solution into the n minus 1 field in case it didn't get populated by an image deformer
  if(has_image_deformer){
    scalar_t exact_u = 0.0;
    scalar_t exact_v = 0.0;
    schema_->image_deformer()->compute_deformation(cx,cy,exact_u,exact_v);
    schema_->mesh()->get_field(DICe::mesh::field_enums::MODEL_SUBSET_DISPLACEMENT_X_FS)->global_value(correlation_point_global_id_) = exact_u;
    schema_->mesh()->get_field(DICe::mesh::field_enums::MODEL_SUBSET_DISPLACEMENT_Y_FS)->global_value(correlation_point_global_id_) = exact_v;
    // field 8: subset error at center in x direction
    schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_8_FS)->global_value(correlation_point_global_id_) = std::abs(exact_u - u);
    // field 9: subset error at center in x direction
    schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_9_FS)->global_value(correlation_point_global_id_) = std::abs(exact_v - v);
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
  sssig /= subset_->num_pixels();

  // populate the fields:
  // field 1: cos of angle between uhat and grad phi
  const scalar_t cos_theta_hat = norm_uhat_2 == 0.0 ? 1.0 : std::sqrt(norm_uhat_dot_gphi_2/norm_uhat_2);
  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_1_FS)->global_value(correlation_point_global_id_) = cos_theta_hat;
  // field 2: cos of angle between the true motion and grad phi
  const scalar_t cos_theta = norm_ut_2 == 0.0 ? 1.0 : std::sqrt(norm_ut_dot_gphi_2/norm_ut_2);
  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_2_FS)->global_value(correlation_point_global_id_) = cos_theta;
  // field 3: the total L2 error magnitude:
  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_3_FS)->global_value(correlation_point_global_id_) = std::sqrt(norm_error_dot_gphi_2 + norm_error_dot_jgphi_2);
  // field 4: residual based exact error estimate in the direction of grad phi
  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_4_FS)->global_value(correlation_point_global_id_) = 1.0/cos_theta_hat * std::sqrt(int_r_total_2);
  // field 5: the L2 error mag in direction of grad phi:
  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_5_FS)->global_value(correlation_point_global_id_) = std::sqrt(norm_error_dot_gphi_2);
  // field 6: residual based exact error estimate in the direction of grad phi
  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_6_FS)->global_value(correlation_point_global_id_) = std::sqrt(int_r_total_2);
  // field 7: SSSIG field
  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_7_FS)->global_value(correlation_point_global_id_) = sssig;
// field 8: cos of angle between the error and grad phi
//  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_8_FS)->global_value(correlation_point_global_id_) = std::sqrt(norm_error_dot_gphi_2/norm_error_2);
//  // field 9: cos of angle between ut and uhat
//  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_9_FS)->global_value(correlation_point_global_id_) = std::sqrt(norm_ut_dot_uhat_2/norm_ut_2);
//  // field 10: cos of angle between ut and grad_phi
//  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_10_FS)->global_value(correlation_point_global_id_) = std::sqrt(int_r_total_2);
//  // field 10: cos of angle between error and ut
//  schema_->mesh()->get_field(DICe::mesh::field_enums::FIELD_10_FS)->global_value(correlation_point_global_id_) = std::sqrt(int_sub_r_approx);
}

Status_Flag
Objective::computeUpdateRobust(Teuchos::RCP<std::vector<scalar_t> > deformation,
  int_t & num_iterations,
  const scalar_t & override_tol){

  const scalar_t skip_threshold = override_tol==-1 ? schema_->skip_solve_gamma_threshold() : override_tol;

  Status_Flag status_flag;
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::max_iterations,  schema_->max_solver_iterations_robust());
  params->set(DICe::tolerance, schema_->robust_solver_tolerance());
  DICe::Subset_Simplex simplex(this,params);

  Teuchos::RCP<std::vector<scalar_t> > deltas = Teuchos::rcp(new std::vector<scalar_t>(num_dofs(),0.0));

  if(num_dofs()==DICE_DEFORMATION_SIZE_AFFINE){
    (*deltas)[DOF_A] = 0.0001;
    (*deltas)[DOF_B] = 1.0E-5;
    (*deltas)[DOF_C] = 1.0;
    (*deltas)[DOF_D] = 1.0E-5;
    (*deltas)[DOF_E] = 0.0001;
    (*deltas)[DOF_F] = 1.0;
    (*deltas)[DOF_G] = 1.0E-5;
    (*deltas)[DOF_H] = 1.0E-5;
    (*deltas)[DOF_I] = 1.0E-5;
    //    for(int_t i=0;i<num_dofs();++i)
    //      (*deltas)[i] = 0.00001;
    //    (*deltas)[AFFINE_C] = schema_->robust_delta_disp();
    //    (*deltas)[AFFINE_F] = schema_->robust_delta_disp();
  }else{
    for(int_t i=0;i<num_dofs();++i){
      if(i<2) (*deltas)[i] = schema_->robust_delta_disp();
      else
        (*deltas)[i] = schema_->robust_delta_theta();
    }
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
Objective_ZNSSD::computeUpdateFast(Teuchos::RCP<std::vector<scalar_t> > deformation,
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
  //double *GWORK = new double[10*N];
  //int *IWORK = new int[LWORK];
  Teuchos::LAPACK<int_t,double> lapack;
  assert(N==6 && "  DICe ERROR: this DIC method is currently only approprate for 6 variables.");

  // Initialize storage:
  Teuchos::SerialDenseMatrix<int_t,double> H(N,N, true);
  Teuchos::ArrayRCP<double> q(N,0.0);
  Teuchos::RCP<std::vector<scalar_t> > def_old    = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0)); // save off the previous value to test for convergence
  Teuchos::RCP<std::vector<scalar_t> > def_update = Teuchos::rcp(new std::vector<scalar_t>(N,0.0)); // save off the previous value to test for convergence

  // note this creates a pointer to the array so
  // the values are updated each frame if compute_grad_def_images is on
  Teuchos::ArrayRCP<scalar_t> gradGx = subset_->grad_x_array();
  Teuchos::ArrayRCP<scalar_t> gradGy = subset_->grad_y_array();
  const scalar_t cx = subset_->centroid_x();
  const scalar_t cy = subset_->centroid_y();
  const scalar_t meanF = subset_->mean(REF_INTENSITIES);
  // these are used for regularization below
  //const scalar_t prev_u = (*deformation)[DOF_U];
  //const scalar_t prev_v = (*deformation)[DOF_V];
  //const scalar_t prev_theta = (*deformation)[DOF_THETA];

  // SOLVER ---------------------------------------------------------
  DEBUG_MSG(std::setw(5) << "Iter" <<
    std::setw(12) << " Ru" <<
    std::setw(12) << " Rv" <<
    std::setw(12) << " Rt" <<
    std::setw(12) << " u"  <<
    std::setw(12) << " du" <<
    std::setw(12) << " v"  <<
    std::setw(12) << " dv" <<
    std::setw(12) << " t"  <<
    std::setw(12) << " dt");

  int_t solve_it = 0;
  for(;solve_it<=max_solve_its;++solve_it)
  {
    num_iterations = solve_it;
    // update the deformed image with the new deformation:
    try{
      subset_->initialize(schema_->def_img(subset_->sub_image_id()),DEF_INTENSITIES,deformation,schema_->interpolation_method());
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
    //DEBUG_MSG("Subset " << correlation_point_global_id_ << " def mean: " << meanG);

    // the gradients are taken from the def images rather than the ref
    const bool use_ref_grads = schema_->def_img()->has_gradients() ? false : true;

    scalar_t dx=0.0,dy=0.0,Dx=0.0,Dy=0.0,delTheta=0.0,delEx=0.0,delEy=0.0,delGxy=0.0;
    scalar_t gx = 0.0, gy= 0.0;
    scalar_t Gx=0.0,Gy=0.0, GmF=0.0;
    const scalar_t theta = (*deformation)[DICe::DOF_THETA];
    const scalar_t dudx = (*deformation)[DICe::DOF_EX];
    const scalar_t dvdy = (*deformation)[DICe::DOF_EY];
    const scalar_t gxy = (*deformation)[DICe::DOF_GXY];
    const scalar_t cosTheta = std::cos(theta);
    const scalar_t sinTheta = std::sin(theta);

    for(int_t index=0;index<subset_->num_pixels();++index){
      if(subset_->is_deactivated_this_step(index)||!subset_->is_active(index)) continue;
      dx = subset_->x(index) - cx;
      dy = subset_->y(index) - cy;
      Dx = (1.0+dudx)*(dx) + gxy*(dy);
      Dy = (1.0+dvdy)*(dy) + gxy*(dx);
      GmF = (subset_->def_intensities(index) - meanG) - (subset_->ref_intensities(index) - meanF);
      gx = gradGx[index];
      gy = gradGy[index];
      Gx = use_ref_grads ? cosTheta*gx - sinTheta*gy : gx;
      Gy = use_ref_grads ? sinTheta*gx + cosTheta*gy : gy;

      delTheta = Gx*(-sinTheta*Dx - cosTheta*Dy) + Gy*(cosTheta*Dx - sinTheta*Dy);
      //deldelTheta = Gx*(-cosTheta*Dx + sinTheta*Dy) + Gy*(-sinTheta*Dx - cosTheta*Dy);
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
      H(2,2) += delTheta*delTheta;// + delTheta*deldelTheta;
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

    // compute the norm of H prior to taking the inverse:
    const scalar_t det_h = H(0,0)*H(1,1) - H(1,0)*H(0,1);
    const scalar_t norm_H = std::sqrt(H(0,0)*H(0,0) + H(0,1)*H(0,1) + H(1,0)*H(1,0) + H(1,1)*H(1,1));
    scalar_t cond_2x2 = -1.0;
    if(det_h !=0.0){
      const scalar_t norm_Hi = std::sqrt((1.0/(det_h*det_h))*(H(0,0)*H(0,0) + H(0,1)*H(0,1) + H(1,0)*H(1,0) + H(1,1)*H(1,1)));
      cond_2x2 = norm_H * norm_Hi;
    }

    if(schema_->use_objective_regularization()){
      // add the penalty terms
      const scalar_t alpha = schema_->levenberg_marquardt_regularization_factor();
      //q[0] += alpha * ((*deformation)[DICe::DOF_U] - prev_u);
      //q[1] += alpha * ((*deformation)[DICe::DOF_V] - prev_v);
      //q[2] += alpha * ((*deformation)[DICe::DOF_THETA] - prev_theta);
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
    //std::vector<scalar_t> colTotals(N,0.0);
    //for(int_t i=0;i<H.numCols();++i){
    //  for(int_t j=0;j<H.numRows();++j){
    //    colTotals[i]+=std::abs(H(j,i));
    //  }
    //}
    //double anorm = 0.0;
    //for(int_t i=0;i<N;++i){
    //  if(colTotals[i] > anorm) anorm = colTotals[i];
    //}

    // clear temp storage
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    //for(int_t i=0;i<10*N;++i) GWORK[i] = 0.0;
    //for(int_t i=0;i<LWORK;++i) IWORK[i] = 0;
    for(int_t i=0;i<N+1;++i) {IPIV[i] = 0;}
    //double rcond=0.0; // reciporical condition number
    try
    {
      lapack.GETRF(N,N,H.values(),N,IPIV,&INFO);
      //lapack.GECON('1',N,H.values(),N,anorm,&rcond,GWORK,IWORK,&INFO);
      //DEBUG_MSG("Subset " << correlation_point_global_id_ << "    RCOND(H): "<< rcond);
      schema_->global_field_value(correlation_point_global_id_,CONDITION_NUMBER_FS) = cond_2x2;
      //schema_->global_field_value(correlation_point_global_id_,DICe::CONDITION_NUMBER) = (rcond !=0.0) ? 1.0/rcond : 0.0;
      //if(rcond < 1.0E-12) return HESSIAN_SINGULAR;
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

    //DEBUG_MSG("    Iterative updates: u " << (*def_update)[0] << " v " << (*def_update)[1] << " theta " <<
    //  (*def_update)[2] << " ex " << (*def_update)[3] << " ey " << (*def_update)[4] << " gxy " << (*def_update)[5]);

    (*deformation)[DICe::DOF_U] += (*def_update)[0];
    (*deformation)[DICe::DOF_V] += (*def_update)[1];
    (*deformation)[DICe::DOF_THETA] += (*def_update)[2];
    (*deformation)[DICe::DOF_EX] += (*def_update)[3];
    (*deformation)[DICe::DOF_EY] += (*def_update)[4];
    (*deformation)[DICe::DOF_GXY] += (*def_update)[5];

    std::ios  state(NULL);
    state.copyfmt(std::cout);
    DEBUG_MSG(std::setw(5) << solve_it <<
      //std::setw(15) << resid_norm <<
      std::setw(12) << std::scientific << std::setprecision(4) << q[0] <<
      std::setw(12) << q[1] <<
      std::setw(12) << q[2] <<
      std::setw(12) << (*deformation)[DICe::DOF_U] <<
      std::setw(12) << (*def_update)[0] <<
      std::setw(12) << (*deformation)[DICe::DOF_V] <<
      std::setw(12) << (*def_update)[1] <<
      std::setw(12) << (*deformation)[DICe::DOF_THETA] <<
      std::setw(12) << (*def_update)[2]);
    std::cout.copyfmt(state);

    //DEBUG_MSG("Subset " << correlation_point_global_id_ << " -- iteration: " << solve_it << " u " << (*deformation)[DICe::DOF_U] <<
    //  " v " << (*deformation)[DICe::DOF_V] << " theta " << (*deformation)[DICe::DOF_THETA] <<
    //  " ex " << (*deformation)[DICe::DOF_EX] << " ey " << (*deformation)[DICe::DOF_EY] <<
    //  " gxy " << (*deformation)[DICe::DOF_GXY] <<
    //  " residual: (" << q[0] << "," << q[1] << "," << q[2] << ")");

    if(std::abs((*deformation)[DICe::DOF_U] - (*def_old)[DICe::DOF_U]) < solve_tol_disp
        && std::abs((*deformation)[DICe::DOF_V] - (*def_old)[DICe::DOF_V]) < solve_tol_disp
        && std::abs((*deformation)[DICe::DOF_THETA] - (*def_old)[DICe::DOF_THETA]) < solve_tol_theta){
      DEBUG_MSG("Subset " << correlation_point_global_id_ << " ** CONVERGED SOLUTION, u " << (*deformation)[DICe::DOF_U] <<
        " v " << (*deformation)[DICe::DOF_V] <<
        " theta " << (*deformation)[DICe::DOF_THETA]  << " ex " << (*deformation)[DICe::DOF_EX] <<
        " ey " << (*deformation)[DICe::DOF_EY] << " gxy " << (*deformation)[DICe::DOF_GXY]);
      //if(schema_->image_deformer()!=Teuchos::null && schema_->mesh()->get_comm()->get_size()==1){
      computeUncertaintyFields(deformation);
      //}
      break;
    }

    // zero out the storage
    H.putScalar(0.0);
    for(int_t i=0;i<N;++i)
      q[i] = 0.0;
  }

  // clean up storage for lapack:
  delete [] WORK;
  //delete [] GWORK;
  //delete [] IWORK;
  delete [] IPIV;

  if(solve_it>max_solve_its){
    return MAX_ITERATIONS_REACHED;
  }
  else return CORRELATION_SUCCESSFUL;
}


Status_Flag
Objective_ZNSSD_Affine::computeUpdateFast(Teuchos::RCP<std::vector<scalar_t> > deformation,
  int_t & num_iterations){
  // using type double here a lot because LAPACK doesn't support float.
  int_t N = DICE_DEFORMATION_SIZE_AFFINE; // [ AFFINE_A ... AFFINE_I ]
  assert((int_t)deformation->size()==N);
  TEUCHOS_TEST_FOR_EXCEPTION(!subset_->has_gradients(),std::runtime_error,"Error, image gradients have not been computed but are needed here.");

  scalar_t solve_tol = schema_->fast_solver_tolerance();
  const int_t max_solve_its = schema_->max_solver_iterations_fast();
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  Teuchos::LAPACK<int_t,double> lapack;

  // Initialize storage:
  Teuchos::SerialDenseMatrix<int_t,double> tangent(N,N, true);
  Teuchos::ArrayRCP<double> q(N,0.0);
  Teuchos::RCP<std::vector<scalar_t> > def_old    = Teuchos::rcp(new std::vector<scalar_t>(N,0.0)); // save off the previous value to test for convergence
  Teuchos::RCP<std::vector<scalar_t> > def_update = Teuchos::rcp(new std::vector<scalar_t>(N,0.0));

  // note this creates a pointer to the array so
  // the values are updated each frame if compute_grad_def_images is on
  Teuchos::ArrayRCP<scalar_t> gradGx = subset_->grad_x_array();
  Teuchos::ArrayRCP<scalar_t> gradGy = subset_->grad_y_array();
  const scalar_t meanF = subset_->mean(REF_INTENSITIES);

  // SOLVER ---------------------------------------------------------
  DEBUG_MSG(std::setw(5) << "Iter" <<
    std::setw(12) << " a" <<
    std::setw(12) << " b" <<
    std::setw(12) << " c" <<
    std::setw(12) << " d"  <<
    std::setw(12) << " e" <<
    std::setw(12) << " f"  <<
    std::setw(12) << " g" <<
    std::setw(12) << " h"  <<
    std::setw(12) << " i");

  int_t solve_it = 0;
  for(;solve_it<=max_solve_its;++solve_it)
  {
    num_iterations = solve_it;
    // update the deformed image with the new deformation:
    try{
      subset_->initialize(schema_->def_img(subset_->sub_image_id()),DEF_INTENSITIES,deformation,schema_->interpolation_method());
    }
    catch (std::logic_error & err) {
      return SUBSET_CONSTRUCTION_FAILED;
    }

    // compute the mean value of the subsets:
    const scalar_t meanG = subset_->mean(DEF_INTENSITIES);

    // the gradients are taken from the def images rather than the ref
    const bool use_ref_grads = schema_->def_img()->has_gradients() ? false : true;

    scalar_t dwxdx=0.0,dwxdy=0.0,dwydx=0.0,dwydy=0.0;
    scalar_t dwxdx_i=0.0,dwxdy_i=0.0,dwydx_i=0.0,dwydy_i=0.0;
    scalar_t det_w = 0.0;
    scalar_t term_1=0.0,term_2=0.0,term_3=0.0;
    scalar_t x=0.0,y=0.0;
    scalar_t gx = 0.0, gy= 0.0;
    scalar_t Gx=0.0,Gy=0.0, GmF=0.0;
    Teuchos::ArrayRCP<double> resids(N,0.0);

    const scalar_t A = (*deformation)[DOF_A];
    const scalar_t B = (*deformation)[DOF_B];
    const scalar_t C = (*deformation)[DOF_C];
    const scalar_t D = (*deformation)[DOF_D];
    const scalar_t E = (*deformation)[DOF_E];
    const scalar_t F = (*deformation)[DOF_F];
    const scalar_t G = (*deformation)[DOF_G];
    const scalar_t H = (*deformation)[DOF_H];
    const scalar_t I = (*deformation)[DOF_I];
    TEUCHOS_TEST_FOR_EXCEPTION(I==0.0,std::runtime_error,"");

    for(int_t index=0;index<subset_->num_pixels();++index){
      if(subset_->is_deactivated_this_step(index)||!subset_->is_active(index)) continue;
      x = subset_->x(index);
      y = subset_->y(index);
      GmF = (subset_->def_intensities(index) - meanG) - (subset_->ref_intensities(index) - meanF);
      gx = gradGx[index];
      gy = gradGy[index];
      Gx = gx;
      Gy = gy;
      term_1 = (A*x+B*y+C);
      term_2 = (D*x+E*y+F);
      term_3 = (G*x+H*y+I);
      TEUCHOS_TEST_FOR_EXCEPTION(term_3==0.0,std::runtime_error,"");
      if(use_ref_grads){
        dwxdx = A/term_3 - G*term_1/(term_3*term_3);
        dwxdy = B/term_3 - H*term_1/(term_3*term_3);
        dwydx = D/term_3 - G*term_2/(term_3*term_3);
        dwydy = E/term_3 - H*term_2/(term_3*term_3);
        TEUCHOS_TEST_FOR_EXCEPTION(std::abs(dwxdx*dwydy - dwxdy*dwydx) < 1.0E-12,std::runtime_error,"Error, det(dwdx) is zero.");
        det_w = 1.0/(dwxdx*dwydy - dwxdy*dwydx);
        dwxdx_i = det_w * dwydy;
        dwxdy_i = -1.0 * det_w * dwxdy;
        dwydx_i = -1.0 * det_w * dwydx;
        dwydy_i = det_w * dwxdx;
        Gx = dwxdx_i*gx + dwxdy_i*gy;
        Gy = dwydx_i*gx + dwydy_i*gy;
      }
      resids[0] = Gx*x/term_3;
      resids[1] = Gx*y/term_3;
      resids[2] = Gx/term_3;
      resids[3] = Gy*x/term_3;
      resids[4] = Gy*y/term_3;
      resids[5] = Gy/term_3;
      resids[6] = -1.0*(Gx*term_1/(term_3*term_3) + Gy*term_2/(term_3*term_3))*x;
      resids[7] = -1.0*(Gx*term_1/(term_3*term_3) + Gy*term_2/(term_3*term_3))*y;
      resids[8] = -1.0*(Gx*term_1/(term_3*term_3) + Gy*term_2/(term_3*term_3));
      for(int_t i=0;i<q.size();++i){
        q[i] += GmF*resids[i];
        for(int_t j=0;j<q.size();++j)
          tangent(i,j) += resids[i]*resids[j];
      }
    }
    // clear temp storage
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    for(int_t i=0;i<N+1;++i) {IPIV[i] = 0;}
    try
    {
      lapack.GETRF(N,N,tangent.values(),N,IPIV,&INFO);
      schema_->global_field_value(correlation_point_global_id_,CONDITION_NUMBER_FS) = -1.0;
    }
    catch(std::exception &e){
      DEBUG_MSG( e.what() << '\n');
      return LINEAR_SOLVE_FAILED;
    }
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    try
    {
      lapack.GETRI(N,tangent.values(),N,IPIV,WORK,LWORK,&INFO);
    }
    catch(std::exception &e){
      DEBUG_MSG( e.what() << '\n');
      return LINEAR_SOLVE_FAILED;
    }
    // save off last step d
    for(int_t i=0;i<N;++i)
      (*def_old)[i] = (*deformation)[i];

    for(int_t i=0;i<N;++i)
      (*def_update)[i] = 0.0;

    for(int_t i=0;i<N;++i)
      for(int_t j=0;j<N;++j)
        (*def_update)[i] += tangent(i,j)*(-1.0)*q[j];

    for(int_t i=0;i<N;++i)
      (*deformation)[i] += (*def_update)[i];

    std::ios  state(NULL);
    state.copyfmt(std::cout);
    DEBUG_MSG(std::setw(5) << solve_it <<
      //std::setw(15) << resid_norm <<
      std::setw(12) << std::scientific << std::setprecision(4) << (*deformation)[DOF_A] <<
      std::setw(12) << (*deformation)[DOF_B] <<
      std::setw(12) << (*deformation)[DOF_C] <<
      std::setw(12) << (*deformation)[DOF_D] <<
      std::setw(12) << (*deformation)[DOF_E] <<
      std::setw(12) << (*deformation)[DOF_F] <<
      std::setw(12) << (*deformation)[DOF_G] <<
      std::setw(12) << (*deformation)[DOF_H] <<
      std::setw(12) << (*deformation)[DOF_I]);
    std::cout.copyfmt(state);

    bool all_converged = true;
    for(int_t i=0;i<N;++i)
      if(std::abs((*deformation)[i] - (*def_old)[i]) > solve_tol)
        all_converged = false;

    if(all_converged){
      DEBUG_MSG("Subset " << correlation_point_global_id_ << " ** CONVERGED SOLUTION, a " << (*deformation)[DOF_A] <<
        " b " << (*deformation)[DOF_B] <<
        " c " << (*deformation)[DOF_C] <<
        " d " << (*deformation)[DOF_D] <<
        " e " << (*deformation)[DOF_E] <<
        " f " << (*deformation)[DOF_F] <<
        " g " << (*deformation)[DOF_G] <<
        " h " << (*deformation)[DOF_H] <<
        " i " << (*deformation)[DOF_I]);
      computeUncertaintyFields(deformation);
      //}
      break;
    }

    // zero out the storage
    tangent.putScalar(0.0);
    for(int_t i=0;i<N;++i)
      q[i] = 0.0;
  }

  // clean up storage for lapack:
  delete [] WORK;
  delete [] IPIV;

  if(solve_it>max_solve_its){
    return MAX_ITERATIONS_REACHED;
  }
  else return CORRELATION_SUCCESSFUL;
}


}// End DICe Namespace
