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
#ifndef DICE_GLOBAL_H
#define DICE_GLOBAL_H

#include <DICe.h>
#include <DICe_Mesh.h>
#include <DICe_GlobalUtils.h>
#include <DICe_MatrixService.h>

#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#ifdef DICE_TPETRA
  #include <BelosTpetraAdapter.hpp>
#else
  #include <BelosEpetraAdapter.hpp>
#endif


namespace DICe {

// forward declaration of a schema
class Schema;

namespace global{

/// \class Global_Algorithm
/// \brief holds all the methods and data for global DIC
class Global_Algorithm
{
public:
  /// Constrtuctor with valid pointer to a schema
  /// \param schema pointer to the initializing schema
  /// \param params pointer to a set of parameters that define which terms are included, etc.
  Global_Algorithm(Schema * schema,
    const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// Constrtuctor with no schema, but a string name of
  /// manufactured solution problem to run (used for regression testing mostly)
  /// \param params pointer to a set of parameters that define which terms are included, etc.
  Global_Algorithm(const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// Destructor
  virtual ~Global_Algorithm(){};

  /// default constructor tasks
  /// \param params the global params passed through
  void default_constructor_tasks(Teuchos::ArrayRCP<scalar_t> & points_x,
    Teuchos::ArrayRCP<scalar_t> & points_y,
    const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// set up the solvers and allocate memory
  void pre_execution_tasks();

  /// execute the global alg
  Status_Flag execute();

  /// post execution tasks
  void post_execution_tasks(const scalar_t & time_stamp);

  /// evaluate the error norms for mms problems
  /// \param error_bx output error in x velocity field
  /// \param error_by output error in y velocity field
  /// \param error_lambda output error in lagrange multiplier (zero if not used)
  void evaluate_mms_error(scalar_t & error_bx,
    scalar_t & error_bt,
    scalar_t & error_lambda);

  /// Returns the mesh size (max area constraint)
  scalar_t mesh_size()const{
    return mesh_size_;
  }

  /// Returns a pointer to the mesh
  Teuchos::RCP<DICe::mesh::Mesh> mesh()const{
    return mesh_;
  }

protected:
  /// protect the default constructor
  Global_Algorithm(const Global_Algorithm&);
  /// comparison operator
  Global_Algorithm& operator=(const Global_Algorithm&);
  /// pointer to the calling schema
  Schema * schema_;
  /// mesh size constraint (largest size allowed in pixels^2)
  scalar_t mesh_size_;
  /// alpha coefficient squared
  scalar_t alpha2_;
  /// computational mesh
  Teuchos::RCP<DICe::mesh::Mesh> mesh_;
  /// optional pointer to a method of manufactured solutions problem
  Teuchos::RCP<MMS_Problem> mms_problem_;
  /// name of the output file
  std::string output_file_name_;
  /// linear problem
  Teuchos::RCP< Belos::LinearProblem<mv_scalar_type,vec_type,operator_type> > linear_problem_;
  /// belos solver
  Teuchos::RCP< Belos::SolverManager<mv_scalar_type,vec_type,operator_type> > belos_solver_;
  /// matrix service
  Teuchos::RCP<DICe::Matrix_Service> matrix_service_;
  /// true if the solver, etc been initialized
  bool is_initialized_;
};

}// end global namespace

}// End DICe Namespace

#endif
