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
#ifndef DICE_GLOBALUTILS_H
#define DICE_GLOBALUTILS_H

#include <DICe.h>
#include <Teuchos_ParameterList.hpp>

namespace DICe {

namespace global{

// forward declaration
class Global_Algorithm;

/// \class MMS_Problem
/// \brief method of manufactured solutions problem generator
/// provides forces and an analytic solution to evaluate convergence of
/// global method
class MMS_Problem
{
public:
  /// Constrtuctor
  MMS_Problem(const scalar_t & dim_x,
    const scalar_t & dim_y):
      dim_x_(dim_x),
      dim_y_(dim_y){};

  /// Destructor
  virtual ~MMS_Problem(){};

  /// returns the problem dimension in x
  scalar_t dim_x()const{
    return dim_x_;
  }

  /// returns the problem dimension in x
  scalar_t dim_y()const{
    return dim_y_;
  }

  /// Evaluation of the velocity field
  /// \param x x coordinate at which to evaluate the velocity
  /// \param y y coordinate at which to evaluate the velocity
  /// \param b_x output x velocity
  /// \param b_y output y velocity
  virtual void velocity(const scalar_t & x,
    const scalar_t & y,
    scalar_t & b_x,
    scalar_t & b_y)=0;

  /// Evaluation of the lagrange multiplier field
  /// \param x x coordinate at which to evaluate the velocity
  /// \param y y coordinate at which to evaluate the velocity
  /// \param l_out output x lagrange multplier
  virtual void lagrange(const scalar_t & x,
    const scalar_t & y,
    scalar_t & l_out)=0;

  /// Evaluation of the lagrangian derivatives
  /// \param x x coordinate at which to evaluate the velocity
  /// \param y y coordinate at which to evaluate the velocity
  /// \param dl_dx output x deriv
  /// \param dl_dy output y deriv
  virtual void grad_lagrange(const scalar_t & x,
    const scalar_t & y,
    scalar_t & l_x,
    scalar_t & l_y)=0;

  /// Evaluation of the grad of the velocity
  /// \param x x coordinate at which to evaluate the velocity
  /// \param y y coordinate at which to evaluate the velocity
  /// \param b_x_x output x derivative of x velocity
  /// \param b_x_y output y derivative of x velocity
  /// \param b_y_x output x derivative of y velocity
  /// \param b_y_y output y derivative of y velocity
  virtual void grad_velocity(const scalar_t & x,
    const scalar_t & y,
    scalar_t & b_x_x,
    scalar_t & b_x_y,
    scalar_t & b_y_x,
    scalar_t & b_y_y)=0;

  /// Evaluation of the laplacian of the velocity field
  /// \param x x coordinate at which to evaluate the velocity
  /// \param y y coordinate at which to evaluate the velocity
  /// \param lap_b_x output x velocity laplacian
  /// \param lap_b_y output y velocity laplacian
  virtual void velocity_laplacian(const scalar_t & x,
    const scalar_t & y,
    scalar_t & lap_b_x,
    scalar_t & lap_b_y)=0;

  /// Evaluation of the image phi (intensity) field
  /// \param x x coordinate at which to evaluate the intensity
  /// \param y y coordinate at which to evaluate the intensity
  /// \param phi output x velocity laplacian
  virtual void phi(const scalar_t & x,
    const scalar_t & y,
    scalar_t & phi)=0;

  /// Evaluation of the image phi derivatives
  /// \param x x coordinate at which to evaluate the derivatives
  /// \param y y coordinate at which to evaluate the derivatives
  /// \param dphi_dt output time derivative
  /// \param grad_phi_x output x derivative
  /// \param grad_phi_y output y derivative
  virtual void phi_derivatives(const scalar_t & x,
    const scalar_t & y,
    scalar_t & dphi_dt,
    scalar_t & grad_phi_x,
    scalar_t & grad_phi_y)=0;

  /// Evaluation of the force terms
  /// \param x x coordinate at which to evaluate the derivatives
  /// \param y y coordinate at which to evaluate the derivatives
  /// \param f_x output force x
  /// \param f_y output force y
  virtual void force(const scalar_t & x,
    const scalar_t & y,
    const scalar_t & coeff_1,
    std::set<Global_EQ_Term> * eq_terms,
    scalar_t & f_x,
    scalar_t & f_y)=0;


protected:
  /// Protect the default constructor
  MMS_Problem(const MMS_Problem&);
  /// Comparison operator
  MMS_Problem& operator=(const MMS_Problem&);
  /// size of the domain in x
  const scalar_t dim_x_;
  /// size of the domain in y
  const scalar_t dim_y_;
};


/// \class Div_Curl_Modulator
/// \brief an MMS problem where the user can modulate the amount of the div or curl
/// free component in the velocity solution
class Div_Curl_Modulator : public MMS_Problem
{
public:
  /// Constructor
  /// force the size to be 500 x 500
  Div_Curl_Modulator(const Teuchos::RCP<Teuchos::ParameterList> & params):
    MMS_Problem(500,500),
    phi_coeff_(10.0),
    b_coeff_(2.0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("phi_coeff"),std::runtime_error,
      "Error, Div_Curl_Modulator mms problem requires the parameter phi_coeff");
    phi_coeff_ = params->get<double>("phi_coeff");
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("b_coeff"),std::runtime_error,
      "Error, Div_Curl_Modulator mms problem requires the parameter b_coeff");
    b_coeff_ = params->get<double>("b_coeff");
  };

  /// Destructor
  virtual ~Div_Curl_Modulator(){};

  /// See base class definition
  virtual void velocity(const scalar_t & x,
    const scalar_t & y,
    scalar_t & b_x,
    scalar_t & b_y){
    static scalar_t beta = b_coeff_*DICE_PI/dim_x_;
    b_x = sin(beta*x)*cos(beta*y);
    b_y = -cos(beta*x)*sin(beta*y);
  }

  /// See base class definition
  virtual void velocity_laplacian(const scalar_t & x,
    const scalar_t & y,
    scalar_t & lap_b_x,
    scalar_t & lap_b_y){
    static scalar_t beta = b_coeff_*DICE_PI/dim_x_;
    lap_b_x = -beta*beta*cos(beta*y)*sin(beta*x);
    lap_b_y = beta*beta*cos(beta*x)*sin(beta*y);
  }

  /// See base class definition
  virtual void grad_velocity(const scalar_t & x,
    const scalar_t & y,
    scalar_t & b_x_x,
    scalar_t & b_x_y,
    scalar_t & b_y_x,
    scalar_t & b_y_y){
    static scalar_t beta = b_coeff_*DICE_PI/dim_x_;
    b_x_x = beta*cos(beta*x)*cos(beta*y);
    b_x_y = -beta*sin(beta*x)*sin(beta*y);
    b_y_x = beta*sin(beta*x)*sin(beta*y);
    b_y_y = -beta*cos(beta*x)*cos(beta*y);
  }

  /// See base class definition
  virtual void lagrange(const scalar_t & x,
    const scalar_t & y,
    scalar_t & l_out){
    static scalar_t beta = b_coeff_*DICE_PI/dim_x_;
    l_out = sin(beta*x)*cos(beta*y + 0.5*DICE_PI);
  }

  /// See base class definition
  virtual void grad_lagrange(const scalar_t & x,
    const scalar_t & y,
    scalar_t & l_x,
    scalar_t & l_y){
    static scalar_t beta = b_coeff_*DICE_PI/dim_x_;
    l_x = beta*cos(beta*x)*cos(beta*y+0.5*DICE_PI);
    l_y = -beta*sin(beta*x)*sin(beta*y+0.5*DICE_PI);
  }

  /// See base class definition
  virtual void phi(const scalar_t & x,
    const scalar_t & y,
    scalar_t & phi){
    static scalar_t gamma = phi_coeff_*DICE_PI/dim_x_;
    phi = sin(gamma*x)*cos(gamma*y+DICE_PI/2.0);
  }

  /// See base class definition
  virtual void phi_derivatives(const scalar_t & x,
    const scalar_t & y,
    scalar_t & dphi_dt,
    scalar_t & grad_phi_x,
    scalar_t & grad_phi_y) {
    scalar_t b_x = 0.0;
    scalar_t b_y = 0.0;
    velocity(x,y,b_x,b_y);
    scalar_t mod_x = x - b_x;
    scalar_t mod_y = y - b_y;
    scalar_t phi_0 = 0.0;
    phi(x,y,phi_0);
    scalar_t phi_cur = 0.0;
    phi(mod_x,mod_y,phi_cur);
    dphi_dt = phi_cur - phi_0;
    static scalar_t gamma = phi_coeff_*DICE_PI/dim_x_;
    grad_phi_x = gamma*cos(gamma*x)*cos(DICE_PI/2.0 + gamma*y);
    grad_phi_y = -gamma*sin(gamma*x)*sin(DICE_PI/2.0 + gamma*y);
  }

  /// See base class definition
  virtual void force(const scalar_t & x,
    const scalar_t & y,
    const scalar_t & coeff_1,
    std::set<Global_EQ_Term> * eq_terms,
    scalar_t & f_x,
    scalar_t & f_y){

    // TODO check which terms are active

    f_x = 0.0;
    f_y = 0.0;
    scalar_t d_phi_dt = 0.0, grad_phi_x = 0.0, grad_phi_y = 0.0;
    phi_derivatives(x,y,d_phi_dt,grad_phi_x,grad_phi_y);
    if(eq_terms->find(MMS_IMAGE_TIME_FORCE)!=eq_terms->end()){
      f_x += grad_phi_x*d_phi_dt;
      f_y += grad_phi_y*d_phi_dt;
    }
    if(eq_terms->find(MMS_IMAGE_GRAD_TENSOR)!=eq_terms->end()){
      scalar_t b_x=0.0,b_y=0.0;
      velocity(x,y,b_x,b_y);
      f_x += (grad_phi_x*grad_phi_x*b_x + grad_phi_x*grad_phi_y*b_y);
      f_y += (grad_phi_y*grad_phi_x*b_x + grad_phi_y*grad_phi_y*b_y);
    }
    if(eq_terms->find(DIV_SYMMETRIC_STRAIN_REGULARIZATION)!=eq_terms->end()){
      scalar_t lap_b_x=0.0,lap_b_y=0.0;
      velocity_laplacian(x,y,lap_b_x,lap_b_y);
      f_x -= coeff_1*lap_b_x;
      f_y -= coeff_1*lap_b_y;
    }
    if(eq_terms->find(TIKHONOV_REGULARIZATION)!=eq_terms->end()){
      scalar_t b_x=0.0,b_y=0.0;
      velocity(x,y,b_x,b_y);
      f_x += coeff_1*b_x;
      f_y += coeff_1*b_y;
    }
    if(eq_terms->find(GRAD_LAGRANGE_MULTIPLIER)!=eq_terms->end()){
      scalar_t dl_dx=0.0,dl_dy=0.0;
      grad_lagrange(x,y,dl_dx,dl_dy);
      f_x += dl_dx;
      f_y += dl_dy;
    }
  }

protected:
  /// coefficient for image intensity values
  scalar_t phi_coeff_;
  /// coefficient for velocity values
  scalar_t b_coeff_;
};

/// \class MMS_Problem_Factory
/// \brief Factory class that creates method of manufactured solutions problems
class MMS_Problem_Factory
{
public:
  /// Constructor
  MMS_Problem_Factory(){};

  /// Destructor
  virtual ~MMS_Problem_Factory(){}

  /// Create an evaluator
  /// \param problem_name the name of the problem to create
  virtual Teuchos::RCP<MMS_Problem> create(const Teuchos::RCP<Teuchos::ParameterList> & params){
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("problem_name"),std::runtime_error,
      "Error, problem_name not defined for mms_spec");
    const std::string problem_name = params->get<std::string>("problem_name");
    if(problem_name=="div_curl_modulator")
      return Teuchos::rcp(new Div_Curl_Modulator(params));
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"MMS_Problem_Factory: invalid problem: " + problem_name);
    }
  };

private:
  /// Protect the base default constructor
  MMS_Problem_Factory(const MMS_Problem_Factory&);

  /// Comparison operator
  MMS_Problem_Factory& operator=(const MMS_Problem_Factory&);
};

/// adds the symmeterized strain regularizer to the elem stiffness matrix
/// \param spa_dim spatial dimension
/// \param num_funcs the number of shape functions
/// \param coeff leading constant coefficient
/// \param J determinant of the jacobian
/// \param gp_weight gauss weight
/// \param inv_jac inverse jacobian
/// \param DN derivatives of the shape functions
/// \param elem_stiffness output the element stiffness contributions
void div_symmetric_strain(const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & coeff,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * inv_jac,
  const scalar_t * DN,
  scalar_t * elem_stiffness);

/// adds the divergence free constraint stiffness terms
/// \param spa_dim spatial dimension
/// \param t3_num_funcs the number of shape functions for tri3
/// \param t6_num_funcs the number of shape functions for tri6
/// \param J determinant of the jacobian
/// \param gp_weight gauss weight
/// \param inv_jac inverse jacobian
/// \param DN6 derivatives of the shape functions for tri6 elem
/// \param N3 shape functions for tri3 elem
/// \param elem_div_stiffness output the element stiffness contributions
void div_velocity(const int_t spa_dim,
  const int_t t3_num_funcs,
  const int_t t6_num_funcs,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * inv_jac,
  const scalar_t * DN6,
  const scalar_t * N3,
  scalar_t * elem_div_stiffness);

/// adds a tikhonov type regularizer to the governing eqs
/// \param spa_dim spatial dimension
/// \param num_funcs number of shape functions
/// \param J determinant of the jacobian
/// \param gp_weight gauss point weight
/// \param N shape function vector
/// \param elem_stiffness element stiffness
void tikhonov_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness);

/// adds a tikhonov type regularizer to the governing eqs
/// \param spa_dim spatial dimension
/// \param num_funcs number of shape functions
/// \param J determinant of the jacobian
/// \param gp_weight gauss point weight
/// \param N shape function vector
/// \param elem_stiffness element stiffness
void lumped_tikhonov_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness);


/// adds the image gradients term to the stiffness matrx (from manufactured solutions problem)
/// \param mms_problem pointer to the method of manufactured solutions problem
/// \param spa_dim spatial dimension
/// \param num_funcs the number of shape functions
/// \param x x-coordinate of the point
/// \param y y-coordinate of the point
/// \param J determinant of the jacobian
/// \param gp_weight gauss weight
/// \param N shape functions
/// \param elem_stiffness output the element stiffness contributions
void mms_image_grad_tensor(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness);

/// adds the image gradients term to the stiffness matrx (from manufactured solutions problem)
/// \param alg pointer to the Global_Algorithm
/// \param spa_dim spatial dimension
/// \param num_funcs the number of shape functions
/// \param x x-coordinate of the point
/// \param y y-coordinate of the point
/// \param J determinant of the jacobian
/// \param gp_weight gauss weight
/// \param N shape functions
/// \param elem_stiffness output the element stiffness contributions
void image_grad_tensor(Global_Algorithm * alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_stiffness);

/// adds the image gradients term to the force vector (from manufactured solutions problem)
/// \param mms_problem pointer to the method of manufactured solutions problem
/// \param spa_dim spatial dimension
/// \param num_funcs the number of shape functions
/// \param x x-coordinate of the point
/// \param y y-coordinate of the point
/// \param J determinant of the jacobian
/// \param gp_weight gauss weight
/// \param N shape functions
/// \param elem_force output the element force contributions
void mms_image_time_force(Teuchos::RCP<MMS_Problem> mms_problem,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_force);

/// adds the mms force vector (from manufactured solutions problem)
/// \param mms_problem pointer to the method of manufactured solutions problem
/// \param spa_dim spatial dimension
/// \param num_funcs the number of shape functions
/// \param x x-coordinate of the point
/// \param y y-coordinate of the point
/// \param coeff coefficient for stress term
/// \param J determinant of the jacobian
/// \param gp_weight gauss weight
/// \param N shape functions
/// \param elem_force output the element force contributions
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
  scalar_t * elem_force);

/// adds the dphi_dt force vector to the residual
/// \param alg pointer to the calling Global_Algorithm
/// \param spa_dim spatial dimension
/// \param num_funcs the number of shape functions
/// \param x x-coordinate of the point
/// \param y y-coordinate of the point
/// \param J determinant of the jacobian
/// \param gp_weight gauss weight
/// \param N shape functions
/// \param elem_force output the element force contributions
void image_time_force(Global_Algorithm* alg,
  const int_t spa_dim,
  const int_t num_funcs,
  const scalar_t & x,
  const scalar_t & y,
  const scalar_t & J,
  const scalar_t & gp_weight,
  const scalar_t * N,
  scalar_t * elem_force);

/// computes the optical flow velocity about a point
/// \param alg pointer to the global algorithm
/// \param c_x closest pixel coord in x
/// \param c_y closest pixel coord in y
/// \param b_x output flow velocity in x
/// \param b_y output flow velocity in y
void optical_flow_velocity(Global_Algorithm * alg,
  const int_t & c_x, // closest pixel in x
  const int_t & c_y, // closest pixel in y
  scalar_t & b_x,
  scalar_t & b_y);

/// computes subset velocity about a point
/// \param alg pointer to the global algorithm
/// \param c_x closest pixel coord in x
/// \param c_y closest pixel coord in y
/// \param subset_size the size of the subset to use
/// \param b_x output flow velocity in x
/// \param b_y output flow velocity in y
void subset_velocity(Global_Algorithm * alg,
  const int_t & c_x, // closest pixel in x
  const int_t & c_y, // closest pixel in y
  const int_t & subset_size,
  scalar_t & b_x,
  scalar_t & b_y);

void calc_jacobian(const scalar_t * xcap,
  const scalar_t * DN,
  scalar_t * jacobian,
  scalar_t * inv_jacobian,
  scalar_t & J,
  int_t num_elem_nodes,
  int_t dim );

void calc_B(const scalar_t * DN,
  const scalar_t * inv_jacobian,
  const int_t num_elem_nodes,
  const int_t dim,
  scalar_t * solid_B);

void calc_mms_force_elasticity(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & alpha,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & force_x,
  scalar_t & force_y);

void calc_mms_vel_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & b_x,
  scalar_t & b_y);

void calc_mms_lap_vel_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & lap_b_x,
  scalar_t & lap_b_y);

void calc_mms_phi_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  const scalar_t & g,
  scalar_t & phi);

void calc_mms_phi_terms_rich(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & m,
  const scalar_t & L,
  const scalar_t & g,
  scalar_t & d_phi_dt,
  scalar_t & grad_phi_x,
  scalar_t & grad_phi_y);

void calc_mms_bc_simple(const scalar_t & x,
  const scalar_t & y,
  scalar_t & b_x,
  scalar_t & b_y);

void calc_mms_force_simple(const scalar_t & alpha,
  scalar_t & force_x,
  scalar_t & force_y);

void calc_mms_bc_2(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  scalar_t & b_x,
  scalar_t & b_y);

}// end global namespace

}// End DICe Namespace

#endif
