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

#ifndef DICE_BCMANAGER_H_
#define DICE_BCMANAGER_H_

#include <DICe.h>
#include <DICe_Mesh.h>
#ifdef DICE_TPETRA
  #include "DICe_MultiFieldTpetra.h"
#else
  #include "DICe_MultiFieldEpetra.h"
#endif

#include <Teuchos_RCP.hpp>

namespace DICe{

namespace global{

// forward declaration of global algorithm
class Global_Algorithm;

/// \class Boundary_Condition
/// \brief boundary condition
class
DICE_LIB_DLL_EXPORT
Boundary_Condition
{
public:
  /// Constrtuctor
  Boundary_Condition(Teuchos::RCP<DICe::mesh::Mesh> mesh,
    const bool is_mixed):
  mesh_(mesh),
  is_mixed_(is_mixed){};

  /// Destructor
  virtual ~Boundary_Condition(){};

  /// Apply the boundary condition
  /// \param first_iteration true if this is the first iteration
  virtual void apply(const bool first_iteration)=0;

protected:
  /// Protect the default constructor
  Boundary_Condition(const Boundary_Condition &);
  /// Comparison operator
  Boundary_Condition & operator=(const Boundary_Condition &);
  /// pointer to the mesh
  Teuchos::RCP<DICe::mesh::Mesh> mesh_;
  /// true if this is a mixed formulation
  bool is_mixed_;
};

/// \class Dirichlet_BC
/// \brief prescribed values along the boundary
class
DICE_LIB_DLL_EXPORT
Dirichlet_BC : public Boundary_Condition
{
public:
  /// Constructor
  Dirichlet_BC(Teuchos::RCP<DICe::mesh::Mesh> mesh,
    const bool is_mixed):
  Boundary_Condition(mesh,is_mixed){
    DEBUG_MSG("Dirichlet_BC::Dirichlet_BC(): Creating a dirichlet BC");
  };

  /// Destructor
  virtual ~Dirichlet_BC(){};

  /// Apply the boundary condition
  virtual void apply(const bool first_iteration);

protected:
};

/// \class Lagrange_BC
/// \brief prescribed values along the boundary
class
DICE_LIB_DLL_EXPORT
Lagrange_BC : public Boundary_Condition
{
public:
  /// Constructor
  Lagrange_BC(Teuchos::RCP<DICe::mesh::Mesh> mesh,
    const bool is_mixed):
  Boundary_Condition(mesh,is_mixed){
    DEBUG_MSG("Lagrange_BC::Lagrange_BC(): Creating a Lagrange BC");
  };

  /// Destructor
  virtual ~Lagrange_BC(){};

  /// Apply the boundary condition
  virtual void apply(const bool first_iteration);

protected:
};

/// \class Corner_BC
/// \brief prescribed values at a corner (presumably the first node with GID 1)
class Corner_BC : public Boundary_Condition
{
public:
  /// Constructor
  Corner_BC(Teuchos::RCP<DICe::mesh::Mesh> mesh,
    const bool is_mixed):
  Boundary_Condition(mesh,is_mixed){
    DEBUG_MSG("Corner_BC::Corner_BC(): Creating a Corner BC");
  };

  /// Destructor
  virtual ~Corner_BC(){};

  /// Apply the boundary condition
  virtual void apply(const bool first_iteration);

protected:
};



/// \class Subset_BC
/// \brief prescribed values along the boundary using a subset solution
class
DICE_LIB_DLL_EXPORT
Subset_BC : public Boundary_Condition
{
public:
  /// Constructor
  Subset_BC(Global_Algorithm * alg,
    Teuchos::RCP<DICe::mesh::Mesh> mesh,
    const bool is_mixed):
  Boundary_Condition(mesh,is_mixed),
  alg_(alg){
    DEBUG_MSG("Subset_BC::Subset_BC(): Creating a subset BC");
  };

  /// Destructor
  virtual ~Subset_BC(){};

  /// Apply the boundary condition
  virtual void apply(const bool first_iteration);

protected:
  /// pointer to a global algorithm
  Global_Algorithm * alg_;
};

/// \class Constant_IC
/// \brief initialize the solution to a constant value for all nodes
class
DICE_LIB_DLL_EXPORT
Constant_IC : public Boundary_Condition
{
public:
  /// Constructor
  Constant_IC(Global_Algorithm * alg,
    Teuchos::RCP<DICe::mesh::Mesh> mesh,
    const bool is_mixed):
  Boundary_Condition(mesh,is_mixed),
  alg_(alg){
    DEBUG_MSG("Constant_IC::Constant_IC(): Creating a constant IC");
  };

  /// Destructor
  virtual ~Constant_IC(){};

  /// Apply the boundary condition
  virtual void apply(const bool first_iteration);

protected:
  /// pointer to a global algorithm
  Global_Algorithm * alg_;
};


/// \class MMS_BC
/// \brief prescribed values along the boundary using the method of manufactured solutions values
class
DICE_LIB_DLL_EXPORT
MMS_BC : public Boundary_Condition
{
public:
  /// Constructor
  MMS_BC(Global_Algorithm * alg,
    Teuchos::RCP<DICe::mesh::Mesh> mesh,
    const bool is_mixed):
  Boundary_Condition(mesh,is_mixed),
  alg_(alg){
    DEBUG_MSG("MMS_BC::Subset_BC(): Creating a MMS BC");
  };

  /// Destructor
  virtual ~MMS_BC(){};

  /// Apply the boundary condition
  virtual void apply(const bool first_iteration);

protected:
  /// pointer to a global algorithm
  Global_Algorithm * alg_;
};

/// \class MMS_Lagrange_BC
/// \brief prescribed values along the boundary using the method of manufactured solutions values
class
DICE_LIB_DLL_EXPORT
MMS_Lagrange_BC : public Boundary_Condition
{
public:
  /// Constructor
  MMS_Lagrange_BC(Global_Algorithm * alg,
    Teuchos::RCP<DICe::mesh::Mesh> mesh,
    const bool is_mixed):
  Boundary_Condition(mesh,is_mixed),
  alg_(alg){
    DEBUG_MSG("MMS_Lagrange_BC::Subset_BC(): Creating an MMS Lagrange BC");
  };

  /// Destructor
  virtual ~MMS_Lagrange_BC(){};

  /// Apply the boundary condition
  virtual void apply(const bool first_iteration);

protected:
  /// pointer to a global algorithm
  Global_Algorithm * alg_;
};


class
DICE_LIB_DLL_EXPORT
BC_Manager
{
public:
  /// constructor that takes two mesh pointers, the l_mesh can be null for single field formulations
  /// \param alg pointer to the parent algorithm
  BC_Manager(Global_Algorithm * alg);

  /// destructor
  virtual ~BC_Manager(){
    delete [] row_bc_register_;
    delete [] col_bc_register_;
    delete [] mixed_bc_register_;
  };

  /// returns true if the register has been initialized
  const bool bc_register_initialized()const{
    return bc_register_initialized_;
  }

  /// method to initialize the bc_register
  /// \param row_register_size the number of rows to track
  /// \param col_register_size the number of cols to track
  /// \param mixed_register_size the number mixed dofs to track
  void initialize_bc_register(const int_t row_register_size,
    const int_t col_register_size,
    const int_t mixed_register_size);

  /// resets all the bc registers
  void clear_bc_register();

  /// returns the size of the row register
  const int_t row_bc_register_size()const{
    return row_bc_register_size_;
  }

  /// returns the size of the col register
  const int_t col_bc_register_size()const{
    return col_bc_register_size_;
  }

  /// returns the size of the mixed bc register
  const int_t mixed_bc_register_size()const{
    return mixed_bc_register_size_;
  }

  /// returns a pointer to teh row register
  bool * row_bc_register(){
    TEUCHOS_TEST_FOR_EXCEPTION(!bc_register_initialized_,std::logic_error,
      "  ERROR: Matrix Service BC register is not yet initialized.");
    return row_bc_register_;
  }

  /// returns a pointer to the col register
  bool * col_bc_register(){
    TEUCHOS_TEST_FOR_EXCEPTION(!bc_register_initialized_,std::logic_error,
      "  ERROR: Matrix Service BC register is not yet initialized.");
    return col_bc_register_;
  }

  /// returns a pointer to the mixed register
  bool * mixed_bc_register(){
    TEUCHOS_TEST_FOR_EXCEPTION(!bc_register_initialized_,std::logic_error,
      "  ERROR: Matrix Service BC register is not yet initialized.");
    return mixed_bc_register_;
  }

  /// registers a row bc
  void register_row_bc(const int_t row_local_id){
    TEUCHOS_TEST_FOR_EXCEPTION(row_local_id>row_bc_register_size_||row_local_id<0,std::logic_error,
      "  ERROR: BC registration requested for invalid row_local_id: " << row_local_id);
    row_bc_register_[row_local_id] = true;
  }
  void register_col_bc(const int_t col_local_id){
    TEUCHOS_TEST_FOR_EXCEPTION(col_local_id>col_bc_register_size_||col_local_id<0,std::logic_error,
      "  ERROR: BC registration requested for invalid col_local_id: " << col_local_id);
    col_bc_register_[col_local_id] = true;
  }
  void register_mixed_bc(const int_t mixed_local_id){
    TEUCHOS_TEST_FOR_EXCEPTION(mixed_local_id>mixed_bc_register_size_||mixed_local_id<0,std::logic_error,
      "  ERROR: BC registration requested for invalid col_local_id: " << mixed_local_id);
    mixed_bc_register_[mixed_local_id] = true;
  }
  void unregister_row_bc(const int_t row_local_id){
    TEUCHOS_TEST_FOR_EXCEPTION(row_local_id>row_bc_register_size_||row_local_id<0,std::logic_error,
      "  ERROR: BC registration removal requested for invalid row_local_id: " << row_local_id);
    row_bc_register_[row_local_id] = false;
  }
  void unregister_col_bc(const int_t col_local_id){
    TEUCHOS_TEST_FOR_EXCEPTION(col_local_id>col_bc_register_size_||col_local_id<0,std::logic_error,
      "  ERROR: BC registration removal requested for invalid col_local_id: " << col_local_id);
    col_bc_register_[col_local_id] = false;
  }
  void unregister_mixed_bc(const int_t mixed_local_id){
    TEUCHOS_TEST_FOR_EXCEPTION(mixed_local_id>mixed_bc_register_size_||mixed_local_id<0,std::logic_error,
      "  ERROR: BC registration removal requested for invalid col_local_id: " << mixed_local_id);
    mixed_bc_register_[mixed_local_id] = false;
  }
  void initialize_matrix_storage(Teuchos::RCP<matrix_type> matrix, const int_t storage_size);
  const bool is_col_bc(const int_t col_local_id, const unsigned col_dim)const{return col_bc_register_[col_local_id*spa_dim_ + col_dim];}
  const bool is_row_bc(const int_t row_local_id, const unsigned row_dim)const{return row_bc_register_[row_local_id*spa_dim_ + row_dim];}
  const bool is_col_bc(const unsigned col_index)const{return col_bc_register_[col_index];}
  const bool is_row_bc(const unsigned row_index)const{return row_bc_register_[row_index];}
  const bool is_mixed_bc(const unsigned mixed_index)const{return mixed_bc_register_[mixed_index];}

  /// Create a boundary condition by name:
  /// \param eq_term type of boundary condition to create
  void create_bc(Global_EQ_Term eq_term,
    const bool is_mixed);

  /// apply the boundary conditions
  /// \param first_iteration true if this is the first iteration
  void apply_bcs(const bool first_iteration);

  /// apply the initial conditions
  /// \param first_iteration true if this is the first iteration
  void apply_ics(const bool first_iteration);

protected:
  BC_Manager(const BC_Manager&);
  BC_Manager& operator=(const BC_Manager&);
  /// spatial dimension
  const int_t spa_dim_;
  // A simple array of bools that identifies which ids are boundary conditions.
  // This is used to prevent assembly in the matrix for these ids.
  // Rows are stored separately from column bcs since the rows will only be local elems
  // but the columns may have off-processor elems
  bool * row_bc_register_;
  bool * col_bc_register_;
  bool * mixed_bc_register_;
  int_t row_bc_register_size_;
  int_t col_bc_register_size_;
  int_t mixed_bc_register_size_;
  bool bc_register_initialized_;
  /// quadratic parent mesh
  Teuchos::RCP<DICe::mesh::Mesh> mesh_;
  /// true if this is a mixed formulation
  bool is_mixed_;
  /// a vector of boundary conditions to enforce
  std::vector<Teuchos::RCP<Boundary_Condition> > bcs_;
  /// a vector of initial conditions to enforce
  std::vector<Teuchos::RCP<Boundary_Condition> > ics_;
  /// pointer to a global algorithm
  Global_Algorithm * alg_;
};

} // namespace DICe

} // namespace DICe

#endif /* DICE_BCMANAGER_H_ */
