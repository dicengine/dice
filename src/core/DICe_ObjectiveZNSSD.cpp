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
  return subset_->gamma();
}

scalar_t
Objective_ZNSSD::sigma( Teuchos::RCP<std::vector<scalar_t> > &deformation) const {

  // if the gradients don't exist or the optimization method is SIMPLEX based return 0.0;
   if(!subset_->has_gradients()||schema_->optimization_method()==DICe::SIMPLEX) return 0.0;

   assert(deformation->size()==DICE_DEFORMATION_SIZE);

   // TODO address is_active

   int_t N = 2;
   int *IPIV = new int[N+1];
   double *EIGS = new double[N+1];
   int LWORK = N*N;
   int QWORK = 3*N;
   int INFO = 0;
   double *WORK = new double[LWORK];
   double *SWORK = new double[QWORK];
   Teuchos::LAPACK<int_t,double> lapack;
   assert(N==2);

   // Initialize storage:
   Teuchos::SerialDenseMatrix<int_t,double> H(N,N, true);
   Teuchos::ArrayRCP<scalar_t> gradGx = subset_->grad_x_array();
   Teuchos::ArrayRCP<scalar_t> gradGy = subset_->grad_y_array();

   // update the deformed image with the new deformation:
   try{
     subset_->initialize(schema_->def_img(),DEF_INTENSITIES,deformation,schema_->interpolation_method());
   }
   catch (std::logic_error & err) {return -1.0;}

   scalar_t Gx=0.0,Gy=0.0;
   for(int_t index=0;index<subset_->num_pixels();++index){
     // skip the deactivated pixels
     // TODO if(!is_active[index]||is_deactivated_this_step[index])continue;
     Gx = gradGx[index];
     Gy = gradGy[index];
     H(0,0) += Gx*Gx;
     H(1,0) += Gy*Gx;
     H(0,1) += Gx*Gy;
     H(1,1) += Gy*Gy;
   }
   // clear temp storage
   for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
   for(int_t i=0;i<QWORK;++i) SWORK[i] = 0.0;
   for(int_t i=0;i<N+1;++i) {IPIV[i] = 0; EIGS[i] = 0.0;}
   try{
     lapack.GETRF(N,N,H.values(),N,IPIV,&INFO);
   }
   catch(std::exception &e){DEBUG_MSG( e.what() << '\n'); return -1.0;}
   for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
   try{
     lapack.GETRI(N,H.values(),N,IPIV,WORK,LWORK,&INFO);
   }
   catch(std::exception &e){DEBUG_MSG( e.what() << '\n'); return -1.0;}

   // now compute the eigenvalues for H^-1 as an estimate of sigma:
   lapack.SYEV('N','U',N,H.values(),N,EIGS,SWORK,QWORK,&INFO);
   DEBUG_MSG("Subset " << correlation_point_global_id_ << " Eigenvalues of H^-1: " << EIGS[0] << " " << EIGS[1]);
   const scalar_t maxEig = std::max(EIGS[0],EIGS[1]);

   // 95% confidence interval
   const scalar_t sigma = 2.0*std::sqrt(maxEig*5.991);
   DEBUG_MSG("Subset " << correlation_point_global_id_ << " sigma: " << sigma);

   // clean up storage for lapack:
   delete [] WORK;
   delete [] SWORK;
   delete [] IPIV;
   delete [] EIGS;

   return sigma;
}

Status_Flag
Objective_ZNSSD::initialize_from_previous_frame(Teuchos::RCP<std::vector<scalar_t> > & deformation){

  assert(deformation->size()==DICE_DEFORMATION_SIZE);

  // 1: check if there exists a value from the previous step (image in a series)
  const scalar_t sigma = local_field_value(DICe::SIGMA);
  if(sigma!=-1.0){// && sigma!=0.0)
    const Projection_Method projection = schema_->projection_method();
    if(schema_->translation_enabled()){
      DEBUG_MSG("Subset " << correlation_point_global_id_ << " Translation is enabled.");
      if(schema_->image_frame() > 2 && projection == VELOCITY_BASED){
        (*deformation)[DICe::DISPLACEMENT_X] = local_field_value(DICe::DISPLACEMENT_X) + (local_field_value(DICe::DISPLACEMENT_X)-local_field_value_nm1(DICe::DISPLACEMENT_X));
        (*deformation)[DICe::DISPLACEMENT_Y] = local_field_value(DICe::DISPLACEMENT_Y) + (local_field_value(DICe::DISPLACEMENT_Y)-local_field_value_nm1(DICe::DISPLACEMENT_Y));
      }
      else{
        (*deformation)[DICe::DISPLACEMENT_X] = local_field_value(DICe::DISPLACEMENT_X);
        (*deformation)[DICe::DISPLACEMENT_Y] = local_field_value(DICe::DISPLACEMENT_Y);
      }
    }
    if(schema_->rotation_enabled()){
      DEBUG_MSG("Subset " << correlation_point_global_id_ << " Rotation is enabled.");
      if(schema_->image_frame() > 2 && projection == VELOCITY_BASED){
        (*deformation)[DICe::ROTATION_Z] = local_field_value(DICe::ROTATION_Z) + (local_field_value(DICe::ROTATION_Z)-local_field_value_nm1(DICe::ROTATION_Z));
      }
      else{
        (*deformation)[DICe::ROTATION_Z] = local_field_value(DICe::ROTATION_Z);
      }
    }
    if(schema_->normal_strain_enabled()){
      DEBUG_MSG("Subset " << correlation_point_global_id_ << " Normal strain is enabled.");
      (*deformation)[DICe::NORMAL_STRAIN_X] = local_field_value(DICe::NORMAL_STRAIN_X);
      (*deformation)[DICe::NORMAL_STRAIN_Y] = local_field_value(DICe::NORMAL_STRAIN_Y);
    }
    if(schema_->shear_strain_enabled()){
      DEBUG_MSG("Subset " << correlation_point_global_id_ << " Shear strain is enabled.");
      (*deformation)[DICe::SHEAR_STRAIN_XY] = local_field_value(DICe::SHEAR_STRAIN_XY);
    }

    DEBUG_MSG("Projection Method: " << projection);
    // TODO Remove this output:
    DEBUG_MSG("Subset " << correlation_point_global_id_ << " solution from prev. step: u " << local_field_value(DICe::DISPLACEMENT_X)
      << " v " << local_field_value(DICe::DISPLACEMENT_Y)
      << " theta " << local_field_value(DICe::ROTATION_Z)
      << " e_x " << local_field_value(DICe::NORMAL_STRAIN_X)
      << " e_y " << local_field_value(DICe::NORMAL_STRAIN_Y)
      << " g_xy " << local_field_value(DICe::SHEAR_STRAIN_XY));

    DEBUG_MSG("Subset " << correlation_point_global_id_ << " solution from nm1 step: u " << local_field_value_nm1(DICe::DISPLACEMENT_X)
      << " v " << local_field_value_nm1(DICe::DISPLACEMENT_Y)
      << " theta " << local_field_value_nm1(DICe::ROTATION_Z)
      << " e_x " << local_field_value_nm1(DICe::NORMAL_STRAIN_X)
      << " e_y " << local_field_value_nm1(DICe::NORMAL_STRAIN_Y)
      << " g_xy " << local_field_value_nm1(DICe::SHEAR_STRAIN_XY));

    DEBUG_MSG("Subset " << correlation_point_global_id_ << " init. with values: u " << (*deformation)[DICe::DISPLACEMENT_X]
                        << " v " << (*deformation)[DICe::DISPLACEMENT_Y]
                        << " theta " << (*deformation)[DICe::ROTATION_Z]
                        << " e_x " << (*deformation)[DICe::NORMAL_STRAIN_X]
                        << " e_y " << (*deformation)[DICe::NORMAL_STRAIN_Y]
                        << " g_xy " << (*deformation)[DICe::SHEAR_STRAIN_XY]);
    return INITIALIZE_USING_PREVIOUS_FRAME_SUCCESSFUL;
  }
  return INITIALIZE_FAILED;
}

Status_Flag
Objective_ZNSSD::initialize_from_neighbor( Teuchos::RCP<std::vector<scalar_t> > &deformation) {

  assert(deformation->size()==DICE_DEFORMATION_SIZE);

  // try a neighbor's value of displacement x and y:
  // this doesn't work for the first subset
  //assert(correlation_point_global_id_>0);

  const int_t neighbor_gid = local_field_value(DICe::NEIGHBOR_ID);//correlation_point_global_id_ - 1;
  if(neighbor_gid==-1) // must use previous value instead (could be a seed location)
    return initialize_from_previous_frame(deformation);
  assert(neighbor_gid>=0);

  // for now require that the neighbor is on this processor
  // TODO enable cross-processor use of neighbor values:
  const int_t neighbor_lid = schema_->get_local_id(neighbor_gid);
  TEUCHOS_TEST_FOR_EXCEPTION(neighbor_lid<0,std::runtime_error,"Error: Only neighbors on this processor can be used for initialization");

  DEBUG_MSG("Subset " << correlation_point_global_id_ << " will use neighbor values from " << neighbor_gid);

  const scalar_t sigma = schema_->local_field_value(neighbor_gid,DICe::SIGMA);
  if(sigma!=-1.0){
    if(schema_->translation_enabled()){
      (*deformation)[DICe::DISPLACEMENT_X] = schema_->local_field_value(neighbor_gid,DICe::DISPLACEMENT_X);
      (*deformation)[DICe::DISPLACEMENT_Y] = schema_->local_field_value(neighbor_gid,DICe::DISPLACEMENT_Y);
    }
    if(schema_->rotation_enabled()){
      (*deformation)[DICe::ROTATION_Z] = schema_->local_field_value(neighbor_gid,DICe::ROTATION_Z);
    }
    // add these later:
    if(schema_->normal_strain_enabled()){
      (*deformation)[DICe::NORMAL_STRAIN_X] = schema_->local_field_value(neighbor_gid,DICe::NORMAL_STRAIN_X);
      (*deformation)[DICe::NORMAL_STRAIN_Y] = schema_->local_field_value(neighbor_gid,DICe::NORMAL_STRAIN_Y);
    }
    if(schema_->shear_strain_enabled()){
      (*deformation)[DICe::SHEAR_STRAIN_XY] = schema_->local_field_value(neighbor_gid,DICe::SHEAR_STRAIN_XY);
    }

    DEBUG_MSG("Subset " << correlation_point_global_id_ << " init. from neighbor values: u " << (*deformation)[DICe::DISPLACEMENT_X]
                        << " v " << (*deformation)[DICe::DISPLACEMENT_Y]
                        << " theta " << (*deformation)[DICe::ROTATION_Z]
                        << " e_x " << (*deformation)[DICe::NORMAL_STRAIN_X]
                        << " e_y " << (*deformation)[DICe::NORMAL_STRAIN_Y]
                        << " g_xy " << (*deformation)[DICe::SHEAR_STRAIN_XY]);

    return INITIALIZE_USING_NEIGHBOR_VALUE_SUCCESSFUL;
  }
  return INITIALIZE_FAILED;
}

Status_Flag
Objective_ZNSSD::search_step(Teuchos::RCP<std::vector<scalar_t> > & deformation,
  const int_t window_size,
  const scalar_t step_size,
  scalar_t & return_value) {
  assert(deformation->size()==DICE_DEFORMATION_SIZE);
  Teuchos::RCP<std::vector<scalar_t> > trial_def = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));
  const int_t num_steps = window_size / step_size;
  scalar_t disp_x = 0.0;
  scalar_t disp_y = 0.0;
  scalar_t min_gamma = 4.0;
  scalar_t min_x = 0.0, min_y = 0.0;
  for(int_t y=-num_steps/2;y<=num_steps/2;++y){
    disp_y = (*deformation)[DISPLACEMENT_Y] + y*step_size;
    (*trial_def)[DICe::DISPLACEMENT_Y] = disp_y;
    for(int_t x=-num_steps/2;x<=num_steps/2;++x){
      disp_x = (*deformation)[DISPLACEMENT_X] + x*step_size;
      (*trial_def)[DICe::DISPLACEMENT_X] = disp_x;
      // TODO turn off deactivated pixels
      // ref_subset_->reset_is_deactivated_this_step();
      // ref_subset_->turn_off_obstructed_pixels(trial_def);
      // evaluate gamma for this displacement
      const scalar_t gammaTrial = gamma(trial_def);
      //std::cout << " Trial x " << disp_x << " trial y " << disp_y << " trial theta " << theta << " trial gamma " << gammaTrial << std::endl;
      if(gammaTrial < min_gamma){
        //std::cout << " min found!! " << std::endl;
        min_gamma = gammaTrial;
        min_x = disp_x;
        min_y = disp_y;
      }
    }
  }
  DEBUG_MSG("Subset " << correlation_point_global_id_ << " search step returned best gamma at u " << min_x << " v " << min_y << " gamma " << min_gamma);

  // put the values into the deformation vector:
  (*deformation)[DISPLACEMENT_X] = min_x;
  (*deformation)[DISPLACEMENT_Y] = min_y;
  return_value = min_gamma;
  if(min_gamma < 0.6) return SEARCH_SUCCESSFUL;
  else return SEARCH_FAILED;
}

Status_Flag
Objective_ZNSSD::search(Teuchos::RCP<std::vector<scalar_t> > & deformation,
  const int_t precision_level,
  scalar_t & return_value) {
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, method has not been implemented.");
//  DEBUG_MSG("Subset " << correlation_point_global_id_ << " conducting localized SEARCH to initialize around point u " <<
//    (*deformation)[DISPLACEMENT_X] << " v " << (*deformation)[DISPLACEMENT_Y] <<
//    " theta " << (*deformation)[ROTATION_Z]);
//
//  assert(deformation->size()==DICE_DEFORMATION_SIZE);
//
//  // get the tolerances
//  const scalar_t disp_jump = schema_->disp_jump_tol();
//  const scalar_t theta_jump = schema_->theta_jump_tol();
//  const int_t num_disp_steps = 10; // TODO revisit how many steps to take
//  const int_t num_theta_steps = 10;
//  const scalar_t disp_step = disp_jump/num_disp_steps;
//  const scalar_t theta_step = theta_jump/num_theta_steps;
//
//  Teuchos::RCP<std::vector<scalar_t> > trial_def = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));
//  // temp subset for use in turning off pixels that are obstructed or fail the intensity deviation test:
//  Teuchos::RCP<DICe::Subset> trial_subset = Teuchos::rcp(new Subset(ref_subset_));
//
//  scalar_t disp_x = 0.0;
//  scalar_t disp_y = 0.0;
//  scalar_t theta = 0.0;
//  scalar_t min_gamma = 4.0;
//  scalar_t min_x = 0.0, min_y = 0.0, min_t = 0.0;
//  for(int_t y=0;y<num_disp_steps;++y){
//    disp_y = (*deformation)[DISPLACEMENT_Y] - disp_jump/2.0 + y*disp_step;
//    (*trial_def)[DICe::DISPLACEMENT_Y] = disp_y;
//    for(int_t x=0;x<num_disp_steps;++x){
//      disp_x = (*deformation)[DISPLACEMENT_X] - disp_jump/2.0 + x*disp_step;
//      (*trial_def)[DICe::DISPLACEMENT_X] = disp_x;
//      for(int_t t=0;t<num_theta_steps;++t){
//        theta = (*deformation)[ROTATION_Z] - theta_jump/2.0 + t*theta_step;
//        (*trial_def)[DICe::ROTATION_Z] = theta;
//
//        ref_subset_->reset_is_deactivated_this_step();
//        ref_subset_->turn_off_obstructed_pixels(trial_def);
//        trial_subset->initialize(trial_def,schema_->interpolation_method(),schema_->def_img());
//        // evaluate gamma for this displacement
//        const scalar_t gammaTrial = gamma(trial_def);
//        //std::cout << " Trial x " << disp_x << " trial y " << disp_y << " trial theta " << theta << " trial gamma " << gammaTrial << std::endl;
//        if(gammaTrial < min_gamma){
//          //std::cout << " min found!! " << std::endl;
//          min_gamma = gammaTrial;
//          min_x = disp_x;
//          min_y = disp_y;
//          min_t = theta;
//        }
//      }
//    }
//  }
//  DEBUG_MSG("Subset " << correlation_point_global_id_ << " localized search returned best gamma at u " << min_x << " v " << min_y << " theta " << min_t << " gamma " << min_gamma);
//  return_value = min_gamma;
//  if(min_gamma < 0.6) return SEARCH_SUCCESSFUL;
//  else return SEARCH_FAILED;
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
      // TODO TODO TODO check for is deactivated this step
      //if(ref_subset_->is_deactivated_this_step(index)||!ref_subset_->is_active(index)) continue;
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
