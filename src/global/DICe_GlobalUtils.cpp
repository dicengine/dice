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

#include <DICe_GlobalUtils.h>
#include <DICe_Global.h>
#include <DICe_MeshIO.h>
#include <DICe_Schema.h>
#include <DICe_Subset.h>

#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#ifdef DICE_TPETRA
  #include <BelosTpetraAdapter.hpp>
#else
  #include <BelosEpetraAdapter.hpp>
#endif

namespace DICe {

namespace global{

void div_symmetric_strain(const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & coeff,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * inv_jac,
  const scalar_t * DN,
  scalar_t * elem_stiffness){
  const int_t B_dim = 2*spa_dim - 1;
  scalar_t B[B_dim*num_funcs*spa_dim];

  // compute the B matrix
  DICe::global::calc_B(DN,inv_jac,num_funcs,spa_dim,B);

  // compute B'*B
  for(int_t i=0;i<num_funcs*spa_dim;++i){
    for(int_t j=0;j<B_dim;++j){
      for(int_t k=0;k<num_funcs*spa_dim;++k){
        elem_stiffness[i*num_funcs*spa_dim + k] +=
            coeff*B[j*num_funcs*spa_dim+i]*B[j*num_funcs*spa_dim + k] * gp_weight * J;
      }
    }
  }
}

void mms_image_grad_tensor(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness){
  TEUCHOS_TEST_FOR_EXCEPTION(mms_problem==Teuchos::null,std::runtime_error,
    "Error, the pointer to the mms problem must be valid");
  // compute the image stiffness terms
  scalar_t d_phi_dt = 0.0, grad_phi_x = 0.0, grad_phi_y = 0.0;
  mms_problem->phi_derivatives(x,y,d_phi_dt,grad_phi_x,grad_phi_y);
  //std::cout << " x " << x << " y " << y << " grad_phi_x " << grad_phi_x << " grad_phi_y " << grad_phi_y << std::endl;
  // image stiffness terms
  for(int_t i=0;i<num_funcs;++i){
    const int_t row1 = (i*spa_dim) + 0;
    const int_t row2 = (i*spa_dim) + 1;
    for(int_t j=0;j<num_funcs;++j){
      elem_stiffness[row1*num_funcs*spa_dim + j*spa_dim+0]
                     += N[i]*(grad_phi_x*grad_phi_x)*N[j]*gp_weight*J;
      elem_stiffness[row1*num_funcs*spa_dim + j*spa_dim+1]
                     += N[i]*(grad_phi_x*grad_phi_y)*N[j]*gp_weight*J;
      elem_stiffness[row2*num_funcs*spa_dim + j*spa_dim+0]
                     += N[i]*(grad_phi_y*grad_phi_x)*N[j]*gp_weight*J;
      elem_stiffness[row2*num_funcs*spa_dim + j*spa_dim+1]
                     += N[i]*(grad_phi_y*grad_phi_y)*N[j]*gp_weight*J;
    }
  }
}

void mms_force(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & coeff,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  std::set<Global_EQ_Term> * eq_terms,
  scalar_t * elem_force){
  TEUCHOS_TEST_FOR_EXCEPTION(mms_problem==Teuchos::null,std::runtime_error,
    "Error, the pointer to the mms problem must be valid");

  scalar_t fx = 0.0;
  scalar_t fy = 0.0;
  mms_problem->force(x,y,coeff,eq_terms,fx,fy);

  //compute the force terms for this point
  for(int_t i=0;i<num_funcs;++i){
    elem_force[i*spa_dim+0] += fx*N[i]*gp_weight*J;
    elem_force[i*spa_dim+1] += fy*N[i]*gp_weight*J;
  }
}


void mms_image_time_force(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_force){
  TEUCHOS_TEST_FOR_EXCEPTION(mms_problem==Teuchos::null,std::runtime_error,
    "Error, the pointer to the mms problem must be valid");
  // compute the image force terms
  scalar_t d_phi_dt = 0.0, grad_phi_x = 0.0, grad_phi_y = 0.0;
  mms_problem->phi_derivatives(x,y,d_phi_dt,grad_phi_x,grad_phi_y);
  for(int_t i=0;i<num_funcs;++i){
    elem_force[i*spa_dim+0] -= d_phi_dt*grad_phi_x*N[i]*gp_weight*J;
    elem_force[i*spa_dim+1] -= d_phi_dt*grad_phi_y*N[i]*gp_weight*J;
  }
}

void image_time_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & bx,
  const scalar_t & by,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_force){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  // compute the image force terms
  const intensity_t phi_0 = alg->ref_img()->interpolate_bicubic(x-bx,y-by);
  const intensity_t phi = alg->def_img()->interpolate_bicubic(x,y);
  const scalar_t d_phi_dt = phi - phi_0;
  const scalar_t grad_phi_x = alg->grad_x()->interpolate_bicubic(x-bx,y-by);
  const scalar_t grad_phi_y = alg->grad_y()->interpolate_bicubic(x-bx,y-by);
  for(int_t i=0;i<num_funcs;++i){
    elem_force[i*spa_dim+0] -= d_phi_dt*grad_phi_x*N[i]*gp_weight*J;
    elem_force[i*spa_dim+1] -= d_phi_dt*grad_phi_y*N[i]*gp_weight*J;
  }
}

void image_grad_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");
  // compute the image stiffness terms
  const scalar_t grad_phi_x = alg->grad_x()->interpolate_bicubic(x,y);
  const scalar_t grad_phi_y = alg->grad_y()->interpolate_bicubic(x,y);

  // image stiffness terms
  for(int_t i=0;i<num_funcs;++i){
    const int_t row1 = (i*spa_dim) + 0;
    const int_t row2 = (i*spa_dim) + 1;
    for(int_t j=0;j<num_funcs;++j){
      elem_stiffness[row1*num_funcs*spa_dim + j*spa_dim+0]
                     += N[i]*(grad_phi_x*grad_phi_x)*N[j]*gp_weight*J;
      elem_stiffness[row1*num_funcs*spa_dim + j*spa_dim+1]
                     += N[i]*(grad_phi_x*grad_phi_y)*N[j]*gp_weight*J;
      elem_stiffness[row2*num_funcs*spa_dim + j*spa_dim+0]
                     += N[i]*(grad_phi_y*grad_phi_x)*N[j]*gp_weight*J;
      elem_stiffness[row2*num_funcs*spa_dim + j*spa_dim+1]
                     += N[i]*(grad_phi_y*grad_phi_y)*N[j]*gp_weight*J;
    }
  }
}

void tikhonov_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  const scalar_t alpha2 = alg->alpha2();

  // image stiffness terms
  for(int_t i=0;i<num_funcs;++i){
    const int_t row1 = (i*spa_dim) + 0;
    const int_t row2 = (i*spa_dim) + 1;
    for(int_t j=0;j<num_funcs;++j){
      elem_stiffness[row1*num_funcs*spa_dim + j*spa_dim+0]
                     += N[i]*alpha2*N[j]*gp_weight*J;
      elem_stiffness[row2*num_funcs*spa_dim + j*spa_dim+1]
                     += N[i]*alpha2*N[j]*gp_weight*J;
    }
  }
}

void lumped_tikhonov_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  const scalar_t alpha2 = alg->alpha2();

  // image stiffness terms
  for(int_t i=0;i<num_funcs;++i){
    const int_t row1 = (i*spa_dim) + 0;
    const int_t row2 = (i*spa_dim) + 1;
    for(int_t j=0;j<num_funcs;++j){
      elem_stiffness[row1*num_funcs*spa_dim + row1]
                     += N[i]*alpha2*N[j]*gp_weight*J;
      elem_stiffness[row2*num_funcs*spa_dim + row2]
                     += N[i]*alpha2*N[j]*gp_weight*J;
    }
  }
}



void div_velocity(const int_t spa_dim,
  const int_t t3_num_funcs,
  const int_t t6_num_funcs,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * inv_jac,
  const scalar_t * DN6,
  const scalar_t * N3,
  scalar_t * elem_div_stiffness){
  scalar_t vec_invjTDNT[t6_num_funcs*spa_dim];
  for(int_t n=0;n<t6_num_funcs;++n){
    vec_invjTDNT[n*spa_dim+0] = inv_jac[0]*DN6[n*spa_dim+0] + inv_jac[2]*DN6[n*spa_dim+1];
    vec_invjTDNT[n*spa_dim+1] = inv_jac[1]*DN6[n*spa_dim+0] + inv_jac[3]*DN6[n*spa_dim+1];
  }
  // div velocity stiffness terms
  for(int_t i=0;i<t3_num_funcs;++i){
    for(int_t j=0;j<t6_num_funcs;++j){
      for(int_t dim=0;dim<spa_dim;++dim){
        elem_div_stiffness[i*t6_num_funcs*spa_dim + j*spa_dim+dim]
                           -= N3[i]*vec_invjTDNT[j*spa_dim+dim]*gp_weight*J;
      }
    }
  }
}

void subset_velocity(Global_Algorithm * alg,
  const int_t & c_x, // closest pixel in x
  const int_t & c_y, // closest pixel in y
  const int_t & subset_size,
  scalar_t & b_x,
  scalar_t & b_y){

  // create a subset:
  Teuchos::RCP<Subset> subset = Teuchos::rcp(new Subset(c_x,c_y,subset_size,subset_size));
  subset->initialize(alg->schema()->ref_img(),REF_INTENSITIES); // get the schema ref image rather than the alg since the alg is already normalized

  // using type double here a lot because LAPACK doesn't support float.
  int_t N = 2; // [ u_x u_y ]
  scalar_t solve_tol_disp = alg->schema()->fast_solver_tolerance();
  const int_t max_solve_its = alg->schema()->max_solver_iterations_fast();
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  double *GWORK = new double[10*N];
  int *IWORK = new int[LWORK];
  Teuchos::LAPACK<int_t,double> lapack;

  // Initialize storage:
  Teuchos::SerialDenseMatrix<int_t,double> H(N,N, true);
  Teuchos::ArrayRCP<double> q(N,0.0);
  Teuchos::RCP<std::vector<scalar_t> > deformation = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0)); // save off the previous value to test for convergence
  Teuchos::RCP<std::vector<scalar_t> > def_old     = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0)); // save off the previous value to test for convergence
  Teuchos::RCP<std::vector<scalar_t> > def_update  = Teuchos::rcp(new std::vector<scalar_t>(N,0.0)); // save off the previous value to test for convergence

  Teuchos::ArrayRCP<scalar_t> gradGx = subset->grad_x_array();
  Teuchos::ArrayRCP<scalar_t> gradGy = subset->grad_y_array();
  const scalar_t meanF = subset->mean(REF_INTENSITIES);

  // SOLVER ---------------------------------------------------------

  int_t solve_it = 0;
  for(;solve_it<=max_solve_its;++solve_it)
  {
    // update the deformed image with the new deformation:
    subset->initialize(alg->schema()->def_img(),DEF_INTENSITIES,deformation); // get the schema def image rather than the alg since the alg is already normalized

    // compute the mean value of the subsets:
    const scalar_t meanG = subset->mean(DEF_INTENSITIES);

    scalar_t Gx=0.0,Gy=0.0, GmF=0.0;
    for(int_t index=0;index<subset->num_pixels();++index){
      if(subset->is_deactivated_this_step(index)||!subset->is_active(index)) continue;
      GmF = (subset->def_intensities(index) - meanG) - (subset->ref_intensities(index) - meanF);
      Gx = gradGx[index];
      Gy = gradGy[index];

      q[0] += Gx*GmF;
      q[1] += Gy*GmF;
      H(0,0) += Gx*Gx;
      H(0,1) += Gx*Gy;
      H(1,0) += Gy*Gx;
      H(1,1) += Gy*Gy;
    }

    // determine the max value in the matrix:
    scalar_t maxH = 0.0;
    for(int_t i=0;i<H.numCols();++i)
      for(int_t j=0;j<H.numRows();++j)
        if(std::abs(H(i,j))>maxH) maxH = std::abs(H(i,j));

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
      //DEBUG_MSG("Subset at cx " << c_x << " cy " << c_y  << "    RCOND(H): "<< rcond);
      TEUCHOS_TEST_FOR_EXCEPTION(rcond < 1.0E-12,std::runtime_error,"Hessian singular");
    }
    catch(std::exception &e){
      DEBUG_MSG( e.what() << '\n');
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"subset boundary initializer condition number estimate failed");
    }
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    try
    {
      lapack.GETRI(N,H.values(),N,IPIV,WORK,LWORK,&INFO);
    }
    catch(std::exception &e){
      DEBUG_MSG( e.what() << '\n');
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"subset boundary initializer matrix solve failed");
    }

    // save off last step d
    for(int_t i=0;i<DICE_DEFORMATION_SIZE;++i)
      (*def_old)[i] = (*deformation)[i];

    for(int_t i=0;i<N;++i)
      (*def_update)[i] = 0.0;

    for(int_t i=0;i<N;++i)
      for(int_t j=0;j<N;++j)
        (*def_update)[i] += H(i,j)*(-1.0)*q[j];

    //DEBUG_MSG("    Iterative updates: u " << (*def_update)[0] << " v " << (*def_update)[1]);

    (*deformation)[DICe::DISPLACEMENT_X] += (*def_update)[0];
    (*deformation)[DICe::DISPLACEMENT_Y] += (*def_update)[1];

    //DEBUG_MSG("Subset at cx " << c_x << " cy " << c_y  << " -- iteration: " << solve_it << " u " << (*deformation)[DICe::DISPLACEMENT_X] <<
    //  " v " << (*deformation)[DICe::DISPLACEMENT_Y] << " theta " << (*deformation)[DICe::ROTATION_Z] <<
    //  " ex " << (*deformation)[DICe::NORMAL_STRAIN_X] << " ey " << (*deformation)[DICe::NORMAL_STRAIN_Y] <<
    //  " gxy " << (*deformation)[DICe::SHEAR_STRAIN_XY] << ")");

    if(std::abs((*deformation)[DICe::DISPLACEMENT_X] - (*def_old)[DICe::DISPLACEMENT_X]) < solve_tol_disp
        && std::abs((*deformation)[DICe::DISPLACEMENT_Y] - (*def_old)[DICe::DISPLACEMENT_Y]) < solve_tol_disp){
      DEBUG_MSG("subset_velocity(): solution at cx " << c_x << " cy " << c_y << ": b_x " << (*deformation)[DICe::DISPLACEMENT_X] <<
        " b_y " << (*deformation)[DICe::DISPLACEMENT_Y]);
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

  if(solve_it>=max_solve_its){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Subset_velocity(): max iterations reached");
  }

  b_x = (*deformation)[DICe::DISPLACEMENT_X];
  b_y = (*deformation)[DICe::DISPLACEMENT_Y];
}


void optical_flow_velocity(Global_Algorithm * alg,
  const int_t & c_x, // closest pixel in x
  const int_t & c_y, // closest pixel in y
  scalar_t & b_x,
  scalar_t & b_y){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  const int_t window_size = 21; // TODO make sure this is greater than the buffer
  const int_t half_window_size = window_size / 2;
  static scalar_t coeffs[] = {0.0039,0.0111,0.0286,0.0657,0.1353,0.2494,0.4111,0.6065,0.8007,0.9460,1.0000,
         0.9460,0.8007,0.6065,0.4111,0.2494,0.1353,0.0657,0.0286,0.0111,0.0039};
  //const int_t window_size = 13; // TODO make sure this is greater than the buffer
  //const int_t half_window_size = window_size / 2;
  //static scalar_t coeffs[] = {0.51, 0.64,0.84,0.91,0.96,0.99,1.0,0.99,0.96,0.91,0.84,0.64,0.51};
  static scalar_t window_coeffs[window_size][window_size];
  for(int_t j=0;j<window_size;++j){
    for(int_t i=0;i<window_size;++i){
      window_coeffs[i][j] = coeffs[i]*coeffs[j];
    }
  }

  // do the optical flow about these points...
  const int N = 2;
  Teuchos::SerialDenseMatrix<int,double> H(N,N, true);
  Teuchos::ArrayRCP<double> q(N,0.0);
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  Teuchos::LAPACK<int,double> lapack;

  // clear the values of H and q
  H(0,0) = 0.0;
  H(1,0) = 0.0;
  H(0,1) = 0.0;
  H(1,1) = 0.0;
  q[0] = 0.0;
  q[1] = 0.0;
  scalar_t Ix = 0.0;
  scalar_t Iy = 0.0;
  scalar_t It = 0.0;
  scalar_t w_coeff = 0.0;
  int_t x=0,y=0;
  // loop over subset pixels in the deformed location
  for(int_t j=0;j<window_size;++j){
    y = c_y - half_window_size + j;
    for(int_t i=0;i<window_size;++i){
      x = c_x - half_window_size + i;
      Ix = (*alg->grad_x())(x,y);
      Iy = (*alg->grad_y())(x,y);
      It = (*alg->def_img())(x,y) - (*alg->ref_img())(x,y); // FIXME this should be prev image
      w_coeff = window_coeffs[i][j];
      H(0,0) += Ix*Ix*w_coeff*w_coeff;
      H(1,0) += Ix*Iy*w_coeff*w_coeff;
      H(0,1) += Iy*Ix*w_coeff*w_coeff;
      H(1,1) += Iy*Iy*w_coeff*w_coeff;
      q[0] += Ix*It*w_coeff*w_coeff;
      q[1] += Iy*It*w_coeff*w_coeff;
    }
  }
  // do the inversion:
  for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
  for(int_t i=0;i<N+1;++i) {IPIV[i] = 0;}
  lapack.GETRF(N,N,H.values(),N,IPIV,&INFO);
  lapack.GETRI(N,H.values(),N,IPIV,WORK,LWORK,&INFO);
  // compute u and v
  b_x = -H(0,0)*q[0] - H(0,1)*q[1];
  b_y = -H(1,0)*q[0] - H(1,1)*q[1];
  DEBUG_MSG("optical_flow_velocity(): at point " << c_x << " " << c_y << " b_x "  << b_x << " b_y " << b_y);

  delete [] IPIV;
  delete [] WORK;
}


void calc_jacobian(const scalar_t * xcap,
  const scalar_t * DN,
  scalar_t * jacobian,
  scalar_t * inv_jacobian,
  scalar_t & J,
  int_t num_elem_nodes,
  int_t dim ){

  for(int_t i=0;i<dim*dim;++i){
    jacobian[i] = 0.0;
    inv_jacobian[i] = 0.0;
  }
  J = 0.0;

//  std::cout << " xcap " << std::endl;
//  for(int_t k=0;k<num_elem_nodes;++k){
//    for(int_t i=0;i<dim;++i){
//      std::cout << xcap[k*dim + i] << " " ;
//    }
//    std::cout << std::endl;
//  }
//  std::cout << " DN " << std::endl;
//  for(int_t k=0;k<num_elem_nodes;++k){
//    for(int_t i=0;i<dim;++i){
//      std::cout << DN[k*dim + i] << " " ;
//    }
//    std::cout << std::endl;
//  }
  for(int_t i=0;i<dim;++i)
    for(int_t j=0;j<dim;++j)
      for(int_t k=0;k<num_elem_nodes;++k)
        jacobian[i*dim+j] += xcap[k*dim + i] * DN[k*dim+j];
//  std::cout << " jacobian " << std::endl;
//  for(int_t i=0;i<dim;++i){
//    for(int_t j=0;j<dim;++j)
//      std::cout << " " << jacobian[i*dim+j];
//    std::cout << std::endl;
//  }

  if(dim==2){
    J = jacobian[0]*jacobian[3] - jacobian[1]*jacobian[2];
    TEUCHOS_TEST_FOR_EXCEPTION(J<=0.0,std::runtime_error,
      "Error: determinant 0.0 encountered or negative det");
    inv_jacobian[0] =  jacobian[3] / J;
    inv_jacobian[1] = -jacobian[1] / J;
    inv_jacobian[2] = -jacobian[2] / J;
    inv_jacobian[3] =  jacobian[0] / J;
  }
  else if(dim==3){
    J =   jacobian[0]*jacobian[4]*jacobian[8] + jacobian[1]*jacobian[5]*jacobian[6] + jacobian[2]*jacobian[3]*jacobian[7]
        - jacobian[6]*jacobian[4]*jacobian[2] - jacobian[7]*jacobian[5]*jacobian[0] - jacobian[8]*jacobian[3]*jacobian[1];
    TEUCHOS_TEST_FOR_EXCEPTION(J<=0.0,std::runtime_error,
      "Error: determinant 0.0 encountered or negative det");
    inv_jacobian[0] = ( -jacobian[5]*jacobian[7] + jacobian[4]*jacobian[8]) /  J;
    inv_jacobian[1] = (  jacobian[2]*jacobian[7] - jacobian[1]*jacobian[8]) /  J;
    inv_jacobian[2] = ( -jacobian[2]*jacobian[4] + jacobian[1]*jacobian[5]) /  J;
    inv_jacobian[3] = (  jacobian[5]*jacobian[6] - jacobian[3]*jacobian[8]) /  J;
    inv_jacobian[4] = ( -jacobian[2]*jacobian[6] + jacobian[0]*jacobian[8]) /  J;
    inv_jacobian[5] = (  jacobian[2]*jacobian[3] - jacobian[0]*jacobian[5]) /  J;
    inv_jacobian[6] = ( -jacobian[4]*jacobian[6] + jacobian[3]*jacobian[7]) /  J;
    inv_jacobian[7] = (  jacobian[1]*jacobian[6] - jacobian[0]*jacobian[7]) /  J;
    inv_jacobian[8] = ( -jacobian[1]*jacobian[3] + jacobian[0]*jacobian[4]) /  J;
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"Error, invalid dimension");

//  std::cout << " J " << J << std::endl;
//  std::cout << " inv_jacobian " << std::endl;
//  for(int_t i=0;i<dim;++i){
//    for(int_t j=0;j<dim;++j)
//      std::cout << " " << inv_jacobian[i*dim+j];
//    std::cout << std::endl;
//  }
};

void calc_B(const scalar_t * DN,
  const scalar_t * inv_jacobian,
  const int_t num_elem_nodes,
  const int_t dim,
  scalar_t * solid_B){

  scalar_t dN[dim*num_elem_nodes];
  for(int_t i=0;i<dim*num_elem_nodes;++i)
    dN[i] = 0.0;

//  std::cout << " DN " << std::endl;
//  for(int_t j=0;j<num_elem_nodes;++j){
//    for(int_t i=0;i<dim;++i)
//      std::cout << " " << DN[j*dim+i];
//    std::cout << std::endl;
//  }

  // compute j_inv_transpose * DN_transpose
  for(int_t i=0;i<dim;++i)
    for(int_t j=0;j<num_elem_nodes;++j)
      for(int_t k=0;k<dim;++k)
        dN[i*num_elem_nodes+j] += inv_jacobian[k*dim + i] * DN[j*dim+k];

//  std::cout << " dN " << std::endl;
//  for(int_t i=0;i<dim;++i){
//    for(int_t j=0;j<num_elem_nodes;++j)
//      std::cout << " " << dN[i*num_elem_nodes+j];
//    std::cout << std::endl;
//  }

  const int_t stride = dim*num_elem_nodes;
  int_t placer = 0;
  if(dim ==3){
    for(int_t i=0;i<6*dim*num_elem_nodes;++i)
      solid_B[i] = 0.0;
    for(int_t i=0;i<num_elem_nodes;i++){
      placer = i*dim;     // hold the place in B for each node
      solid_B[0*stride + placer + 0] = dN[0*num_elem_nodes + i];
      solid_B[1*stride + placer + 1] = dN[1*num_elem_nodes + i];
      solid_B[2*stride + placer + 2] = dN[2*num_elem_nodes + i];
      solid_B[3*stride + placer + 0] = dN[1*num_elem_nodes + i];
      solid_B[3*stride + placer + 1] = dN[0*num_elem_nodes + i];
      solid_B[4*stride + placer + 1] = dN[2*num_elem_nodes + i];
      solid_B[4*stride + placer + 2] = dN[1*num_elem_nodes + i];
      solid_B[5*stride + placer + 0] = dN[2*num_elem_nodes + i];
      solid_B[5*stride + placer + 2] = dN[0*num_elem_nodes + i];
    };
  };

  if(dim==2){
    for(int_t i=0;i<3*dim*num_elem_nodes;++i)
      solid_B[i] = 0.0;
    for(int_t i=0;i<num_elem_nodes;i++){
      placer = i*dim;     // hold the place in B for each node
      solid_B[0*stride + placer + 0] = dN[0*num_elem_nodes + i];
      solid_B[1*stride + placer + 1] = dN[1*num_elem_nodes + i];
      solid_B[2*stride + placer + 0] = dN[1*num_elem_nodes + i];
      solid_B[2*stride + placer + 1] = dN[0*num_elem_nodes + i];
    };
  };

//  std::cout << " B " << std::endl;
//  for(int_t i=0;i<6;++i){
//    for(int_t j=0;j<num_elem_nodes*dim;++j)
//      std::cout << " " << solid_B[i*(num_elem_nodes*dim)+j];
//    std::cout << std::endl;
//  }

}

void calc_mms_force_elasticity(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & alpha,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & force_x,
  scalar_t & force_y){
  const scalar_t beta = m*DICE_PI/L;
  force_x = alpha*beta*beta*cos(beta*y)*sin(beta*x);
  force_y = -alpha*beta*beta*cos(beta*x)*sin(beta*y);
}

void calc_mms_vel_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & b_x,
  scalar_t & b_y){

  const scalar_t beta = m*DICE_PI/L;
  b_x = sin(beta*x)*cos(beta*y);
  b_y = -cos(beta*x)*sin(beta*y);
  //b_x = x;
  //b_y = -y;
  //b_x = x + y*y;
  //b_y = x*x - y;

}

void calc_mms_lap_vel_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & lap_b_x,
  scalar_t & lap_b_y){
  const scalar_t beta = m*DICE_PI/L;
  lap_b_x = -beta*beta*cos(beta*y)*sin(beta*x);
  lap_b_y = beta*beta*cos(beta*x)*sin(beta*y);
  //lap_b_x = 0.0;
  //lap_b_y = 0.0;
  //lap_b_x = 2.0;
  //lap_b_y = 2.0;
}

void calc_mms_phi_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  const scalar_t & g,
  scalar_t & phi){
  const scalar_t gamma = g*DICE_PI/L;
  phi = sin(gamma*x)*cos(gamma*y+DICE_PI/2.0);
  //const scalar_t gamma = g*DICE_PI/L;
  //phi = -1.0/gamma*(std::cos(gamma*x)*std::cos(gamma*(x-L)) + std::cos(gamma*y)*std::cos(y-L));
}

void calc_mms_phi_terms_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & m,
  const scalar_t & L,
  const scalar_t & g,
  scalar_t & d_phi_dt,
  scalar_t & grad_phi_x,
  scalar_t & grad_phi_y){

  scalar_t b_x = 0.0;
  scalar_t b_y = 0.0;
  calc_mms_vel_rich(x,y,L,m,b_x,b_y);
  scalar_t mod_x = x - b_x;
  scalar_t mod_y = y - b_y;

  scalar_t phi_0 = 0.0;
  calc_mms_phi_rich(x,y,L,g,phi_0);
  scalar_t phi = 0.0;
  calc_mms_phi_rich(mod_x,mod_y,L,g,phi);
  d_phi_dt = phi - phi_0;

  const scalar_t gamma = g*DICE_PI/L;
  grad_phi_x = gamma*cos(gamma*x)*cos(DICE_PI/2.0 + gamma*y);
  grad_phi_y = -gamma*sin(gamma*x)*sin(DICE_PI/2.0 + gamma*y);
}

void calc_mms_bc_simple(const scalar_t & x,
  const scalar_t & y,
  scalar_t & b_x,
  scalar_t & b_y){
  //b_x = x + y;
  //b_y = x - y;
  b_x = x + y*y;
  b_y = x*x - y;
  //b_x = 0.0001;
  //b_y = 0.0001;
}

void calc_mms_bc_2(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  scalar_t & b_x,
  scalar_t & b_y){
  b_x = std::cos(x*DICE_PI/L);
  b_y = 0.0;
}


void calc_mms_force_simple(const scalar_t & alpha,
  scalar_t & force_x,
  scalar_t & force_y){
  //force_x = 0.0;
  //force_y = 0.0;
  force_x = -2.0*alpha;
  force_y = -2.0*alpha;
}

}// end global namespace


}// End DICe Namespace
