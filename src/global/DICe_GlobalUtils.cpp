// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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

template <typename S>
DICE_LIB_DLL_EXPORT
void div_symmetric_strain(const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & coeff,
  const S & J,
  const S & gp_weight,
  const S * inv_jac,
  const S * DN,
  S * elem_stiffness){
  const int_t B_dim = 2*spa_dim - 1;
  std::vector<S> B(B_dim*num_funcs*spa_dim);

  // compute the B matrix
  DICe::global::calc_B(DN,inv_jac,num_funcs,spa_dim,&B[0]);

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
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void div_symmetric_strain(const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & coeff,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * inv_jac,
  const precision_t * DN,
  precision_t * elem_stiffness);
#endif
template DICE_LIB_DLL_EXPORT void div_symmetric_strain(const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & coeff,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * inv_jac,
  const scalar_t * DN,
  scalar_t * elem_stiffness);

template <typename S>
DICE_LIB_DLL_EXPORT
void mms_image_grad_tensor(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const S & x,
  const S & y,
  const S & J,
  const S & gp_weight,
  const S * N,
  S * elem_stiffness){
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
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void mms_image_grad_tensor(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & x,
  const precision_t & y,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * N,
  precision_t * elem_stiffness);
#endif
template DICE_LIB_DLL_EXPORT void mms_image_grad_tensor(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness);

template <typename S>
DICE_LIB_DLL_EXPORT
void mms_force(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & coeff,
  const S & J,
  const S & gp_weight,
  const S * N,
  std::set<Global_EQ_Term> * eq_terms,
  S * elem_force){
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
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void mms_force(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & coeff,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * N,
  std::set<Global_EQ_Term> * eq_terms,
  precision_t * elem_force);
#endif
template DICE_LIB_DLL_EXPORT void mms_force(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & coeff,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  std::set<Global_EQ_Term> * eq_terms,
  scalar_t * elem_force);

template <typename S>
DICE_LIB_DLL_EXPORT
void mms_image_time_force(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const S & x,
  const S & y,
  const S & J,
  const S & gp_weight,
  const S * N,
  S * elem_force){
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
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void mms_image_time_force(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & x,
  const precision_t & y,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * N,
  precision_t * elem_force);
#endif
template DICE_LIB_DLL_EXPORT void mms_image_time_force(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_force);

template <typename S>
DICE_LIB_DLL_EXPORT
void image_gray_diff(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const S & x,
  const S & y,
  const S & bx,
  const S & by,
  const S & J,
  const S & gp_weight,
  const S * N,
  S * elem_gray_diff){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");
  // compute the image force terms
  const S phi_0 = alg->schema()->ref_img()->interpolate_bicubic(x-bx,y-by);
  const S phi = alg->schema()->def_img()->interpolate_bicubic(x,y);
  const S d_phi_dt = std::abs(phi - phi_0);
  for(int_t i=0;i<num_funcs;++i){
    elem_gray_diff[i] += d_phi_dt*N[i]*gp_weight*J;
  }
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void image_gray_diff(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & x,
  const precision_t & y,
  const precision_t & bx,
  const precision_t & by,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * N,
  precision_t * elem_gray_diff);
#endif
template DICE_LIB_DLL_EXPORT void image_gray_diff(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & bx,
  const scalar_t & by,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_gray_diff);

template <typename S>
DICE_LIB_DLL_EXPORT
void image_time_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const S & x,
  const S & y,
  const S & bx,
  const S & by,
  const S & J,
  const S & gp_weight,
  const S * N,
  S * elem_force){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  // compute the image force terms
  const S phi_0 = alg->schema()->ref_img()->interpolate_bicubic(x-bx,y-by);
  const S phi = alg->schema()->def_img()->interpolate_bicubic(x,y);
  const S d_phi_dt = phi - phi_0;
  const S grad_phi_x = alg->grad_x()->interpolate_bicubic(x-bx,y-by);
  const S grad_phi_y = alg->grad_y()->interpolate_bicubic(x-bx,y-by);
  for(int_t i=0;i<num_funcs;++i){
    elem_force[i*spa_dim+0] -= d_phi_dt*grad_phi_x*N[i]*gp_weight*J;
    elem_force[i*spa_dim+1] -= d_phi_dt*grad_phi_y*N[i]*gp_weight*J;
  }
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void image_time_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & x,
  const precision_t & y,
  const precision_t & bx,
  const precision_t & by,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * N,
  precision_t * elem_force);
#endif
template DICE_LIB_DLL_EXPORT void image_time_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & bx,
  const scalar_t & by,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_force);

template <typename S>
DICE_LIB_DLL_EXPORT
void image_grad_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const S & x,
  const S & y,
  const S & bx,
  const S & by,
  const S & J,
  const S & gp_weight,
  const S * N,
  S * elem_stiffness){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");
  // compute the image stiffness terms
  const S grad_phi_x = alg->grad_x()->interpolate_bicubic(x-bx,y-by);
  const S grad_phi_y = alg->grad_y()->interpolate_bicubic(x-bx,y-by);

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
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void image_grad_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & x,
  const precision_t & y,
  const precision_t & bx,
  const precision_t & by,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * N,
  precision_t * elem_stiffness);
#endif
template DICE_LIB_DLL_EXPORT void image_grad_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & bx,
  const scalar_t & by,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness);

template <typename S>
DICE_LIB_DLL_EXPORT
void image_grad_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const S & x,
  const S & y,
  const S & bx,
  const S & by,
  const S & J,
  const S & gp_weight,
  const S * N,
  S * elem_force){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  const S grad_phi_x = alg->grad_x()->interpolate_bicubic(x-bx,y-by);
  const S grad_phi_y = alg->grad_y()->interpolate_bicubic(x-bx,y-by);

  // image stiffness terms
  for(int_t i=0;i<num_funcs;++i){
    elem_force[i*spa_dim+0]
               -= (grad_phi_x*grad_phi_x*bx + grad_phi_x*grad_phi_y*by)*N[i]*gp_weight*J;
    elem_force[i*spa_dim+1]
               -= (grad_phi_y*grad_phi_x*bx + grad_phi_y*grad_phi_y*by)*N[i]*gp_weight*J;
  }
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void image_grad_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & x,
  const precision_t & y,
  const precision_t & bx,
  const precision_t & by,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * N,
  precision_t * elem_force);
#endif
template DICE_LIB_DLL_EXPORT void image_grad_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & bx,
  const scalar_t & by,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_force);

template <typename S>
DICE_LIB_DLL_EXPORT
void tikhonov_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const S & J,
  const S & gp_weight,
  const S * N,
  const S & tau,
  S * elem_stiffness){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  const S alpha2 = alg->alpha2();

  // image stiffness terms
  for(int_t i=0;i<num_funcs;++i){
    const int_t row1 = (i*spa_dim) + 0;
    const int_t row2 = (i*spa_dim) + 1;
    for(int_t j=0;j<num_funcs;++j){
      elem_stiffness[row1*num_funcs*spa_dim + j*spa_dim+0]
                     += (1.0 - tau*alpha2)*N[i]*alpha2*N[j]*gp_weight*J;
      elem_stiffness[row2*num_funcs*spa_dim + j*spa_dim+1]
                     += (1.0 - tau*alpha2)*N[i]*alpha2*N[j]*gp_weight*J;
    }
  }
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void tikhonov_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * N,
  const precision_t & tau,
  precision_t * elem_stiffness);
#endif
template DICE_LIB_DLL_EXPORT void tikhonov_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  const scalar_t & tau,
  scalar_t * elem_stiffness);

template <typename S>
DICE_LIB_DLL_EXPORT
void tikhonov_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const S & bx,
  const S & by,
  const S & J,
  const S & gp_weight,
  const S * N,
  S * elem_force){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  const S alpha2 = alg->alpha2();

  // compute the image force terms
  for(int_t i=0;i<num_funcs;++i){
    elem_force[i*spa_dim+0] -= alpha2*bx*N[i]*gp_weight*J;
    elem_force[i*spa_dim+1] -= alpha2*by*N[i]*gp_weight*J;
  }
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void tikhonov_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & bx,
  const precision_t & by,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * N,
  precision_t * elem_force);
#endif
template DICE_LIB_DLL_EXPORT void tikhonov_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & bx,
  const scalar_t & by,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_force);

template <typename S>
DICE_LIB_DLL_EXPORT
void lumped_tikhonov_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const S & J,
  const S & gp_weight,
  const S * N,
  S * elem_stiffness){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  const S alpha2 = alg->alpha2();

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
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void lumped_tikhonov_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * N,
  precision_t * elem_stiffness);
#endif
template DICE_LIB_DLL_EXPORT void lumped_tikhonov_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness);

template <typename S>
DICE_LIB_DLL_EXPORT
void div_velocity(const int_t spa_dim,
  const int_t num_funcs,
  const S & J,
  const S & gp_weight,
  const S * inv_jac,
  const S * DN,
  const S * N,
  const scalar_t & alpha2,
  const S & tau,
  S * elem_div_stiffness){
  std::vector<S> vec_invjTDNT(num_funcs*spa_dim);
  for(int_t n=0;n<num_funcs;++n){
    vec_invjTDNT[n*spa_dim+0] = inv_jac[0]*DN[n*spa_dim+0] + inv_jac[2]*DN[n*spa_dim+1];
    vec_invjTDNT[n*spa_dim+1] = inv_jac[1]*DN[n*spa_dim+0] + inv_jac[3]*DN[n*spa_dim+1];
  }

  // div velocity stiffness terms
  for(int_t i=0;i<num_funcs;++i){
    for(int_t j=0;j<num_funcs;++j){
      for(int_t dim=0;dim<spa_dim;++dim){
        elem_div_stiffness[i*num_funcs*spa_dim + j*spa_dim+dim]
                           -= N[i]*vec_invjTDNT[j*spa_dim+dim]*gp_weight*J;
      }
    }
  }
  for(int_t i=0;i<num_funcs;++i){
    for(int_t j=0;j<num_funcs;++j){
        elem_div_stiffness[i*num_funcs*spa_dim + j*spa_dim+0] -= tau*alpha2*N[j]*vec_invjTDNT[i*spa_dim+0]*gp_weight*J;
        elem_div_stiffness[i*num_funcs*spa_dim + j*spa_dim+1] -= tau*alpha2*N[j]*vec_invjTDNT[i*spa_dim+1]*gp_weight*J;
    }
  }
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void div_velocity(const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * inv_jac,
  const precision_t * DN,
  const precision_t * N,
  const scalar_t & alpha2,
  const precision_t & tau,
  precision_t * elem_div_stiffness);
#endif
template DICE_LIB_DLL_EXPORT void div_velocity(const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * inv_jac,
  const scalar_t * DN,
  const scalar_t * N,
  const scalar_t & alpha2,
  const scalar_t & tau,
  scalar_t * elem_div_stiffness);

template <typename S>
DICE_LIB_DLL_EXPORT
void stab_lagrange(const int_t spa_dim,
  const int_t num_funcs,
  const S & J,
  const S & gp_weight,
  const S * inv_jac,
  const S * DN,
  const S & tau,
  S * elem_stab_stiffness){
  std::vector<S> invjTDNT(num_funcs*spa_dim);
  for(int_t n=0;n<num_funcs;++n){
    invjTDNT[n*spa_dim+0] = inv_jac[0]*DN[n*spa_dim+0] + inv_jac[2]*DN[n*spa_dim+1];
    invjTDNT[n*spa_dim+1] = inv_jac[1]*DN[n*spa_dim+0] + inv_jac[3]*DN[n*spa_dim+1];
  }
  // stab lagrange stiffness terms
  for(int_t i=0;i<num_funcs;++i){
    for(int_t j=0;j<num_funcs;++j){
        elem_stab_stiffness[i*num_funcs + j]
           -= tau*(invjTDNT[i*2]*invjTDNT[j*2] + invjTDNT[i*2+1]*invjTDNT[j*2+1])*gp_weight*J;
    }
  }
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void stab_lagrange(const int_t spa_dim,
  const int_t num_funcs,
  const precision_t & J,
  const precision_t & gp_weight,
  const precision_t * inv_jac,
  const precision_t * DN,
  const precision_t & tau,
  precision_t * elem_stab_stiffness);
#endif
template DICE_LIB_DLL_EXPORT void stab_lagrange(const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * inv_jac,
  const scalar_t * DN,
  const scalar_t & tau,
  scalar_t * elem_stab_stiffness);

template <typename S>
DICE_LIB_DLL_EXPORT
void subset_velocity(Global_Algorithm * alg,
  const int_t & c_x, // closest pixel in x
  const int_t & c_y, // closest pixel in y
  const int_t & subset_size,
  S & b_x,
  S & b_y){

  // create a subset:
  Teuchos::RCP<Subset> subset = Teuchos::rcp(new Subset(c_x,c_y,subset_size,subset_size));
  subset->initialize(alg->schema()->ref_img(),REF_INTENSITIES); // get the schema ref image rather than the alg since the alg is already normalized

  // using type double here a lot because LAPACK doesn't support float.
  Teuchos::RCP<Local_Shape_Function> shape_function = Teuchos::rcp(new Affine_Shape_Function(false,false,false));
  int_t N = shape_function->num_params();
  scalar_t solve_tol_disp = alg->schema()->fast_solver_tolerance();
  const int_t max_solve_its = alg->schema()->max_solver_iterations_fast();
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  int *IWORK = new int[LWORK];
  Teuchos::LAPACK<int_t,double> lapack;


  // Initialize storage:
  Teuchos::SerialDenseMatrix<int_t,double> H(N,N, true);
  Teuchos::ArrayRCP<double> q(N,0.0);
  std::vector<scalar_t> def_old(N,0.0); // save off the previous value to test for convergence
  std::vector<scalar_t> def_update(N,0.0); // save off the previous value to test for convergence
  std::vector<scalar_t> residuals(N,0.0);
  // initialize the displacement field with the incoming values
  shape_function->insert_motion(b_x,b_y);
  for(int_t i=0;i<N;++i)
    def_old[i] = (*shape_function)(i);

  Teuchos::ArrayRCP<scalar_t> gradGx = subset->grad_x_array();
  Teuchos::ArrayRCP<scalar_t> gradGy = subset->grad_y_array();
  const scalar_t meanF = subset->mean(REF_INTENSITIES);

  // SOLVER ---------------------------------------------------------

  int_t solve_it = 0;
  for(;solve_it<=max_solve_its;++solve_it)
  {
    // update the deformed image with the new deformation:
    subset->initialize(alg->schema()->def_img(),DEF_INTENSITIES,shape_function); // get the schema def image rather than the alg since the alg is already normalized

    // compute the mean value of the subsets:
    const scalar_t meanG = subset->mean(DEF_INTENSITIES);

    scalar_t Gx=0.0,Gy=0.0, GmF=0.0;
    for(int_t index=0;index<subset->num_pixels();++index){
      if(subset->is_deactivated_this_step(index)||!subset->is_active(index)) continue;
      GmF = (subset->def_intensities(index) - meanG) - (subset->ref_intensities(index) - meanF);
      Gx = gradGx[index];
      Gy = gradGy[index];
      shape_function->residuals(subset->x(index),subset->y(index),subset->centroid_x(),subset->centroid_y(),Gx,Gy,residuals,false);
      for(int_t i=0;i<N;++i){
        q[i] += GmF*residuals[i];
        for(int_t j=0;j<N;++j)
          H(i,j) += residuals[i]*residuals[j];
      }
    }

    // clear temp storage
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    for(int_t i=0;i<LWORK;++i) IWORK[i] = 0;
    for(int_t i=0;i<N+1;++i) {IPIV[i] = 0;}
    try
    {
      lapack.GETRF(N,N,H.values(),N,IPIV,&INFO);
    }
    catch(std::exception &e){
      std::cout << e.what() << '\n';
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"subset boundary initializer condition number estimate failed");
    }
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    try
    {
      lapack.GETRI(N,H.values(),N,IPIV,WORK,LWORK,&INFO);
    }
    catch(std::exception &e){
      std::cout << e.what() << '\n';
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"subset boundary initializer matrix solve failed");
    }

    // save off last step d
    for(int_t i=0;i<N;++i)
      def_old[i] = (*shape_function)(i);
    for(int_t i=0;i<N;++i)
      def_update[i] = 0.0;
    for(int_t i=0;i<N;++i)
      for(int_t j=0;j<N;++j)
        def_update[i] += H(i,j)*(-1.0)*q[j];
    shape_function->update(def_update);

    scalar_t print_u=0.0,print_v=0.0,print_t=0.0;
    shape_function->map_to_u_v_theta(subset->centroid_x(),subset->centroid_y(),print_u,print_v,print_t);
    if(shape_function->test_for_convergence(def_old,solve_tol_disp)){
      DEBUG_MSG("subset_velocity(): solution at cx " << c_x << " cy " << c_y << ": b_x_in " << b_x << " b_x " << print_u <<
        " b_y_in " << b_y << " b_y " << print_v);
      break;
    }

    // zero out the storage
    H.putScalar(0.0);
    for(int_t i=0;i<N;++i)
      q[i] = 0.0;
  }

  // clean up storage for lapack:
  delete [] WORK;
  delete [] IWORK;
  delete [] IPIV;

  if(solve_it>=max_solve_its){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Subset_velocity(): max iterations reached");
  }
  scalar_t out_u=0.0,out_v=0.0,out_t=0.0;
  shape_function->map_to_u_v_theta(subset->centroid_x(),subset->centroid_y(),out_u,out_v,out_t);
  b_x = out_u;
  b_y = out_v;
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void subset_velocity(Global_Algorithm * alg,
  const int_t & c_x, // closest pixel in x
  const int_t & c_y, // closest pixel in y
  const int_t & subset_size,
  precision_t & b_x,
  precision_t & b_y);
#endif
template DICE_LIB_DLL_EXPORT void subset_velocity(Global_Algorithm * alg,
  const int_t & c_x, // closest pixel in x
  const int_t & c_y, // closest pixel in y
  const int_t & subset_size,
  scalar_t & b_x,
  scalar_t & b_y);

template <typename S>
DICE_LIB_DLL_EXPORT
void optical_flow_velocity(Global_Algorithm * alg,
  const int_t & c_x, // closest pixel in x
  const int_t & c_y, // closest pixel in y
  S & b_x,
  S & b_y){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  const int_t window_size = 21; // TODO make sure this is greater than the buffer
  const int_t half_window_size = window_size / 2;
  static S coeffs[] = {0.0039,0.0111,0.0286,0.0657,0.1353,0.2494,0.4111,0.6065,0.8007,0.9460,1.0000,
         0.9460,0.8007,0.6065,0.4111,0.2494,0.1353,0.0657,0.0286,0.0111,0.0039};
  //const int_t window_size = 13; // TODO make sure this is greater than the buffer
  //const int_t half_window_size = window_size / 2;
  //static S coeffs[] = {0.51, 0.64,0.84,0.91,0.96,0.99,1.0,0.99,0.96,0.91,0.84,0.64,0.51};
  static S window_coeffs[window_size][window_size];
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
  S Ix = 0.0;
  S Iy = 0.0;
  S It = 0.0;
  S w_coeff = 0.0;
  int_t x=0,y=0;
  // loop over subset pixels in the deformed location
  for(int_t j=0;j<window_size;++j){
    y = c_y - half_window_size + j;
    for(int_t i=0;i<window_size;++i){
      x = c_x - half_window_size + i;
      Ix = (*alg->grad_x())(x,y);
      Iy = (*alg->grad_y())(x,y);
      It = (*alg->schema()->def_img())(x,y) - (*alg->schema()->ref_img())(x,y); // FIXME this should be prev image
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
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void optical_flow_velocity(Global_Algorithm * alg,
  const int_t & c_x, // closest pixel in x
  const int_t & c_y, // closest pixel in y
  precision_t & b_x,
  precision_t & b_y);
#endif
template DICE_LIB_DLL_EXPORT void optical_flow_velocity(Global_Algorithm * alg,
  const int_t & c_x, // closest pixel in x
  const int_t & c_y, // closest pixel in y
  scalar_t & b_x,
  scalar_t & b_y);

template <typename S>
DICE_LIB_DLL_EXPORT
S compute_tau_tri3(const Global_Formulation & formulation,
  const scalar_t & alpha2,
  const S * natural_coords,
  const S & J,
  S * inv_jac){

  assert(alpha2!=0.0);
  const S tau_1 = 0.225*0.5*J;
  const S tau_2 = 0.14464285714286*0.5*J;
  const S tau_3 = (inv_jac[0]*inv_jac[0] + inv_jac[2]*inv_jac[2] + inv_jac[0]*inv_jac[1] +
                          inv_jac[2]*inv_jac[3] + inv_jac[1]*inv_jac[1] + inv_jac[3]*inv_jac[3])*4.05*J;
  const S be = 27.0 * natural_coords[0]*natural_coords[1]*(1.0 - natural_coords[0] - natural_coords[1]);
  const S tau = formulation==LEHOUCQ_TURNER ? be*tau_1/alpha2*tau_2 : be*tau_1/alpha2*tau_3;

  return tau;
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT precision_t compute_tau_tri3(const Global_Formulation & formulation,
  const scalar_t & alpha2,
  const precision_t * natural_coords,
  const precision_t & J,
  precision_t * inv_jac);
#endif
template DICE_LIB_DLL_EXPORT scalar_t compute_tau_tri3(const Global_Formulation & formulation,
  const scalar_t & alpha2,
  const scalar_t * natural_coords,
  const scalar_t & J,
  scalar_t * inv_jac);


template <typename S>
DICE_LIB_DLL_EXPORT
void calc_jacobian(const S * xcap,
  const S * DN,
  S * jacobian,
  S * inv_jacobian,
  S & J,
  int_t num_elem_nodes,
  int_t dim ){

  for(int_t i=0;i<dim*dim;++i){
    jacobian[i] = 0.0;
    inv_jacobian[i] = 0.0;
  }
  J = 0.0;

  for(int_t i=0;i<dim;++i)
    for(int_t j=0;j<dim;++j)
      for(int_t k=0;k<num_elem_nodes;++k)
        jacobian[i*dim+j] += xcap[k*dim + i] * DN[k*dim+j];

  if(dim==2){
    J = jacobian[0]*jacobian[3] - jacobian[1]*jacobian[2];
    if(J<=0.0){
      std::cout << "Error: invalid determinant, value = " << J << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
      "Error: determinant 0.0 encountered or negative det");
    }
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
};
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void calc_jacobian(const precision_t * xcap,
  const precision_t * DN,
  precision_t * jacobian,
  precision_t * inv_jacobian,
  precision_t & J,
  int_t num_elem_nodes,
  int_t dim );
#endif
template DICE_LIB_DLL_EXPORT void calc_jacobian(const scalar_t * xcap,
  const scalar_t * DN,
  scalar_t * jacobian,
  scalar_t * inv_jacobian,
  scalar_t & J,
  int_t num_elem_nodes,
  int_t dim );

template <typename S>
DICE_LIB_DLL_EXPORT
void calc_B(const S * DN,
  const S * inv_jacobian,
  const int_t num_elem_nodes,
  const int_t dim,
  S * solid_B){

  std::vector<S> dN(dim*num_elem_nodes);
  for(int_t i=0;i<dim*num_elem_nodes;++i)
    dN[i] = 0.0;

  // compute j_inv_transpose * DN_transpose
  for(int_t i=0;i<dim;++i)
    for(int_t j=0;j<num_elem_nodes;++j)
      for(int_t k=0;k<dim;++k)
        dN[i*num_elem_nodes+j] += inv_jacobian[k*dim + i] * DN[j*dim+k];

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
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void calc_B(const precision_t * DN,
  const precision_t * inv_jacobian,
  const int_t num_elem_nodes,
  const int_t dim,
  precision_t * solid_B);
#endif
template DICE_LIB_DLL_EXPORT void calc_B(const scalar_t * DN,
  const scalar_t * inv_jacobian,
  const int_t num_elem_nodes,
  const int_t dim,
  scalar_t * solid_B);

template <typename S>
DICE_LIB_DLL_EXPORT
void calc_mms_force_elasticity(const S & x,
  const S & y,
  const S & alpha,
  const S & L,
  const S & m,
  S & force_x,
  S & force_y){
  assert(L!=0.0);
  const S beta = m*DICE_PI/L;
  force_x = alpha*beta*beta*cos(beta*y)*sin(beta*x);
  force_y = -alpha*beta*beta*cos(beta*x)*sin(beta*y);
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void calc_mms_force_elasticity(const precision_t & x,
  const precision_t & y,
  const precision_t & alpha,
  const precision_t & L,
  const precision_t & m,
  precision_t & force_x,
  precision_t & force_y);
#endif
template DICE_LIB_DLL_EXPORT void calc_mms_force_elasticity(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & alpha,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & force_x,
  scalar_t & force_y);

template <typename S>
DICE_LIB_DLL_EXPORT
void calc_mms_vel_rich(const S & x,
  const S & y,
  const S & L,
  const S & m,
  S & b_x,
  S & b_y){
  assert(L!=0.0);
  const S beta = m*DICE_PI/L;
  b_x = sin(beta*x)*cos(beta*y);
  b_y = -cos(beta*x)*sin(beta*y);
  //b_x = x;
  //b_y = -y;
  //b_x = x + y*y;
  //b_y = x*x - y;

}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void calc_mms_vel_rich(const precision_t & x,
  const precision_t & y,
  const precision_t & L,
  const precision_t & m,
  precision_t & b_x,
  precision_t & b_y);
#endif
template DICE_LIB_DLL_EXPORT void calc_mms_vel_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & b_x,
  scalar_t & b_y);

template <typename S>
DICE_LIB_DLL_EXPORT
void calc_mms_lap_vel_rich(const S & x,
  const S & y,
  const S & L,
  const S & m,
  S & lap_b_x,
  S & lap_b_y){
  assert(L!=0.0);
  const S beta = m*DICE_PI/L;
  lap_b_x = -beta*beta*cos(beta*y)*sin(beta*x);
  lap_b_y = beta*beta*cos(beta*x)*sin(beta*y);
  //lap_b_x = 0.0;
  //lap_b_y = 0.0;
  //lap_b_x = 2.0;
  //lap_b_y = 2.0;
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void calc_mms_lap_vel_rich(const precision_t & x,
  const precision_t & y,
  const precision_t & L,
  const precision_t & m,
  precision_t & lap_b_x,
  precision_t & lap_b_y);
#endif
template DICE_LIB_DLL_EXPORT void calc_mms_lap_vel_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & lap_b_x,
  scalar_t & lap_b_y);

template <typename S>
DICE_LIB_DLL_EXPORT
void calc_mms_phi_rich(const S & x,
  const S & y,
  const S & L,
  const S & g,
  S & phi){
  assert(L!=0.0);
  const S gamma = g*DICE_PI/L;
  phi = sin(gamma*x)*cos(gamma*y+DICE_PI/2.0);
  //const S gamma = g*DICE_PI/L;
  //phi = -1.0/gamma*(std::cos(gamma*x)*std::cos(gamma*(x-L)) + std::cos(gamma*y)*std::cos(y-L));
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void calc_mms_phi_rich(const precision_t & x,
  const precision_t & y,
  const precision_t & L,
  const precision_t & g,
  precision_t & phi);
#endif
template DICE_LIB_DLL_EXPORT void calc_mms_phi_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  const scalar_t & g,
  scalar_t & phi);

template <typename S>
DICE_LIB_DLL_EXPORT
void calc_mms_phi_terms_rich(const S & x,
  const S & y,
  const S & m,
  const S & L,
  const S & g,
  S & d_phi_dt,
  S & grad_phi_x,
  S & grad_phi_y){
  assert(L!=0.0);
  S b_x = 0.0;
  S b_y = 0.0;
  calc_mms_vel_rich(x,y,L,m,b_x,b_y);
  S mod_x = x - b_x;
  S mod_y = y - b_y;

  S phi_0 = 0.0;
  calc_mms_phi_rich(x,y,L,g,phi_0);
  S phi = 0.0;
  calc_mms_phi_rich(mod_x,mod_y,L,g,phi);
  d_phi_dt = phi - phi_0;

  const S gamma = g*DICE_PI/L;
  grad_phi_x = gamma*cos(gamma*x)*cos(DICE_PI/2.0 + gamma*y);
  grad_phi_y = -gamma*sin(gamma*x)*sin(DICE_PI/2.0 + gamma*y);
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void calc_mms_phi_terms_rich(const precision_t & x,
  const precision_t & y,
  const precision_t & m,
  const precision_t & L,
  const precision_t & g,
  precision_t & d_phi_dt,
  precision_t & grad_phi_x,
  precision_t & grad_phi_y);
#endif
template DICE_LIB_DLL_EXPORT void calc_mms_phi_terms_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & m,
  const scalar_t & L,
  const scalar_t & g,
  scalar_t & d_phi_dt,
  scalar_t & grad_phi_x,
  scalar_t & grad_phi_y);

template <typename S>
DICE_LIB_DLL_EXPORT
void calc_mms_bc_simple(const S & x,
  const S & y,
  S & b_x,
  S & b_y){
  //b_x = x + y;
  //b_y = x - y;
  b_x = x + y*y;
  b_y = x*x - y;
  //b_x = 0.0001;
  //b_y = 0.0001;
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void calc_mms_bc_simple(const precision_t & x,
  const precision_t & y,
  precision_t & b_x,
  precision_t & b_y);
#endif
template DICE_LIB_DLL_EXPORT void calc_mms_bc_simple(const scalar_t & x,
  const scalar_t & y,
  scalar_t & b_x,
  scalar_t & b_y);

template <typename S>
DICE_LIB_DLL_EXPORT
void calc_mms_bc_2(const S & x,
  const S & y,
  const S & L,
  S & b_x,
  S & b_y){
  assert(L!=0.0);
  b_x = std::cos(x*DICE_PI/L);
  b_y = 0.0;
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void calc_mms_bc_2(const precision_t & x,
  const precision_t & y,
  const precision_t & L,
  precision_t & b_x,
  precision_t & b_y);
#endif
template DICE_LIB_DLL_EXPORT void calc_mms_bc_2(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  scalar_t & b_x,
  scalar_t & b_y);

template <typename S>
DICE_LIB_DLL_EXPORT
void calc_mms_force_simple(const S & alpha,
  S & force_x,
  S & force_y){
  //force_x = 0.0;
  //force_y = 0.0;
  force_x = -2.0*alpha;
  force_y = -2.0*alpha;
}
#ifndef PRECISION_SCALAR_SAME_TYPE
template DICE_LIB_DLL_EXPORT void calc_mms_force_simple(const precision_t & alpha,
  precision_t & force_x,
  precision_t & force_y);
#endif
template DICE_LIB_DLL_EXPORT void calc_mms_force_simple(const scalar_t & alpha,
  scalar_t & force_x,
  scalar_t & force_y);

}// end global namespace


}// End DICe Namespace
