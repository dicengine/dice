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

#ifndef DICE_SIMPLEX_H
#define DICE_SIMPLEX_H

#include <DICe.h>
#include <DICe_Objective.h>
#include <DICe_Image.h>

namespace DICe {

/// forward dec for a Triangulation
class Triangulation;


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
  /// \param params Paramters that define the varaitions on the initial guess, convergence tolerance and max number of iterations
  Simplex(const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// destructor
  virtual ~Simplex(){};

  /// \brief Returns the status of the algorithm when complete
  /// \param variables [out] Taken as the initial guess for the first iteration and returned as the converged solution at successful completion.
  /// \param deltas The variations on the initial guess used to construct the other simplex points.
  /// \param num_iterations [out] Returns the number of iterations that took place
  /// \param threshold if the initial evaluation of gamma is below this value, the analysis will be skipped
  /// A return value of MAX_ITERATIONS_REACHED means that convergence did not occur in the allowed number of iterations.
  /// A return value of CORRELATION_SUCCESSFUL means that convergence was obtained for the solution stored in the variables vector
  virtual Status_Flag minimize(Teuchos::RCP<std::vector<scalar_t> > variables,
    Teuchos::RCP<std::vector<scalar_t> > deltas,
    int_t & num_iterations,
    const scalar_t & threshold = 1.0E-10);

  /// \brief the objective function that the simplex method is optimizing
  /// \param variables the current guess at which to evaluate the objective
  virtual scalar_t objective(Teuchos::RCP<std::vector<scalar_t> > variables)=0;

protected:
  /// Maximum allowed iterations for convergence
  int_t max_iterations_;
  /// Convergence tolerance
  double tolerance_;
  /// Numerically small value
  scalar_t tiny_;

};


/// a class that peforms a Nelde-Mead type optimization
class DICE_LIB_DLL_EXPORT
Subset_Simplex : public Simplex {

public:

  /// \brief Default constructor
  /// \param obj Pointer to a DICe::Objective, used to gain access to the gamma() method of the objective
  /// \param params Paramters that define the varaitions on the initial guess, convergence tolerance and max number of iterations
  Subset_Simplex(const DICe::Objective * const obj,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// destructor
  virtual ~Subset_Simplex(){};

  /// \brief the objective function that the simplex method is optimizing
  /// \param variables the current guess at which to evaluate the objective
  virtual scalar_t objective(Teuchos::RCP<std::vector<scalar_t> > variables);

  /// call the minimization routine
  /// \param shape_function pointer to a shape function
  /// \param num_iterations the number of iterations
  /// \param threshold the convergence threshold
  using Simplex::minimize;
  Status_Flag minimize(Teuchos::RCP<Local_Shape_Function> shape_function,
    int_t & num_iterations,
    const scalar_t & threshold = 1.0E-10);

protected:
  /// Pointer to a DICe::Objective, used to gain access to objective methods like gamma()
  const DICe::Objective * const obj_;
  /// Pointer to a shape function class
  Teuchos::RCP<Local_Shape_Function> shape_function_;
};

/// a derived optimization class specific for image homography between two cameras
class DICE_LIB_DLL_EXPORT
Homography_Simplex : public Simplex {

public:

  /// \brief Default constructor
  /// \param left_img the left image
  /// \param right_img the right image
  /// \param tri pointer to a triangulation class
  /// \param params Paramters that define the varaitions on the initial guess, convergence tolerance and max number of iterations
  Homography_Simplex(Teuchos::RCP<Image> left_img,
    Teuchos::RCP<Image> right_img,
    Triangulation * tri,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// destructor
  virtual ~Homography_Simplex(){};

  /// \brief the objective function that the simplex method is optimizing
  /// \param variables the current guess at which to evaluate the objective
  virtual scalar_t objective(Teuchos::RCP<std::vector<scalar_t> > variables);

protected:
  /// Pointer to the left image
  Teuchos::RCP<Image> left_img_;
  /// Pointer to the right image
  Teuchos::RCP<Image> right_img_;
  /// Pointer to a triangulation
  Triangulation * tri_;
};

/// a simplex that performs an affine homography transformation
class DICE_LIB_DLL_EXPORT
Affine_Homography_Simplex : public Simplex {

public:

  /// \brief Default constructor
  /// \param left_img the left image
  /// \param right_img the right image
  /// \param tri pointer to a triangulation class
  /// \param params Paramters that define the varaitions on the initial guess, convergence tolerance and max number of iterations
  Affine_Homography_Simplex(Teuchos::RCP<Image> left_img,
    Teuchos::RCP<Image> right_img,
    Triangulation * tri,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// destructor
  virtual ~Affine_Homography_Simplex(){};

  /// \brief the objective function that the simplex method is optimizing
  /// \param variables the current guess at which to evaluate the objective
  virtual scalar_t objective(Teuchos::RCP<std::vector<scalar_t> > variables);

protected:
  /// Pointer to the left image
  Teuchos::RCP<Image> left_img_;
  /// Pointer to the right image
  Teuchos::RCP<Image> right_img_;
  /// Pointer to a triangulation
  Triangulation * tri_;
};


/// a simplex that performs a quadratic homography transformation
class DICE_LIB_DLL_EXPORT
Quadratic_Homography_Simplex : public Simplex {

public:

  /// \brief Default constructor
  /// \param left_img the left image
  /// \param right_img the right image
  /// \param ulx x coordinate of the upper left corner
  /// \param uly y coordinate of the upper left corner
  /// \param lrx x coordinate of the lower right corner
  /// \param lry y coordinate of the lower right corner (note: in loops the extents are inclusive of the bounds)
  /// \param params Paramters that define the varaitions on the initial guess, convergence tolerance and max number of iterations
  Quadratic_Homography_Simplex(Teuchos::RCP<Image> left_img,
    Teuchos::RCP<Image> right_img,
    const int_t & ulx,
    const int_t & uly,
    const int_t & lrx,
    const int_t & lry,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// destructor
  virtual ~Quadratic_Homography_Simplex(){};

  /// \brief the objective function that the simplex method is optimizing
  /// \param variables the current guess at which to evaluate the objective
  virtual scalar_t objective(Teuchos::RCP<std::vector<scalar_t> > variables);

protected:
  /// Pointer to the left image
  Teuchos::RCP<Image> left_img_;
  /// Pointer to the right image
  Teuchos::RCP<Image> right_img_;
  /// coordinates of the window corners to optimize
  int_t ulx_;
  int_t uly_;
  int_t lrx_;
  int_t lry_;
};

/// a simplex that performs a nonlinear warp
class DICE_LIB_DLL_EXPORT
Warp_Simplex : public Simplex {

public:

  /// \brief Default constructor
  /// \param left_img the left image
  /// \param right_img the right image
  /// \param tri pointer to a triangulation class
  /// \param params Paramters that define the varaitions on the initial guess, convergence tolerance and max number of iterations
  Warp_Simplex(Teuchos::RCP<Image> left_img,
    Teuchos::RCP<Image> right_img,
    Triangulation * tri,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// destructor
  virtual ~Warp_Simplex(){};

  /// \brief the objective function that the simplex method is optimizing
  /// \param variables the current guess at which to evaluate the objective
  virtual scalar_t objective(Teuchos::RCP<std::vector<scalar_t> > variables);

protected:
  /// Pointer to the left image
  Teuchos::RCP<Image> left_img_;
  /// Pointer to the right image
  Teuchos::RCP<Image> right_img_;
  /// Pointer to a triangulation
  Triangulation * tri_;
};

}// End DICe Namespace

#endif
