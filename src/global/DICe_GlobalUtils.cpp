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
#include <DICe_MatrixService.h>
#include <DICe_Schema.h>

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
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_force){
  TEUCHOS_TEST_FOR_EXCEPTION(alg==NULL,std::runtime_error,
    "Error, the pointer to the algorithm must be valid");

  // compute the image force terms
  const intensity_t phi_0 = alg->ref_img()->interpolate_bicubic(x,y);
  const intensity_t phi = alg->def_img()->interpolate_bicubic(x,y);
  const scalar_t d_phi_dt = phi - phi_0;
  const scalar_t grad_phi_x = alg->grad_x()->interpolate_bicubic(x,y);
  const scalar_t grad_phi_y = alg->grad_y()->interpolate_bicubic(x,y);
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
