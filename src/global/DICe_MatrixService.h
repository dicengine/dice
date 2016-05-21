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

#ifndef DICE_MATRIXSERVICE_H_
#define DICE_MATRIXSERVICE_H_

#include <DICe.h>
#ifdef DICE_TPETRA
  #include "DICe_MultiFieldTpetra.h"
#else
  #include "DICe_MultiFieldEpetra.h"
#endif

#include <Teuchos_RCP.hpp>

namespace DICe{


class Matrix_Service
{
public:
  Matrix_Service(const int_t spatial_dimension);
  virtual ~Matrix_Service(){};
  const bool bc_register_initialized()const{return bc_register_initialized_;}
  void initialize_bc_register(const int_t row_register_size, const int_t col_register_size, const int_t mixed_register_size);
  void clear_bc_register();
  const int_t row_bc_register_size()const{return row_bc_register_size_;}
  const int_t col_bc_register_size()const{return col_bc_register_size_;}
  const int_t mixed_bc_register_size()const{return mixed_bc_register_size_;}
  bool * row_bc_register(){
    TEUCHOS_TEST_FOR_EXCEPTION(!bc_register_initialized_,std::logic_error,
      "  ERROR: Matrix Service BC register is not yet initialized.");
    return row_bc_register_;}
  bool * col_bc_register(){
    TEUCHOS_TEST_FOR_EXCEPTION(!bc_register_initialized_,std::logic_error,
      "  ERROR: Matrix Service BC register is not yet initialized.");
    return col_bc_register_;}
  bool * mixed_bc_register(){
    TEUCHOS_TEST_FOR_EXCEPTION(!bc_register_initialized_,std::logic_error,
      "  ERROR: Matrix Service BC register is not yet initialized.");
    return mixed_bc_register_;}
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
  const bool is_col_bc(const int_t col_local_id, const unsigned col_dim)const{return col_bc_register_[col_local_id*spatial_dimension_ + col_dim];}
  const bool is_row_bc(const int_t row_local_id, const unsigned row_dim)const{return row_bc_register_[row_local_id*spatial_dimension_ + row_dim];}
  const bool is_col_bc(const unsigned col_index)const{return col_bc_register_[col_index];}
  const bool is_row_bc(const unsigned row_index)const{return row_bc_register_[row_index];}
  const bool is_mixed_bc(const unsigned mixed_index)const{return mixed_bc_register_[mixed_index];}
  void apply_bc_diagonals(const mv_scalar_type & diagonal_value);
protected:
  Matrix_Service(const Matrix_Service&);
  Matrix_Service& operator=(const Matrix_Service&);
  const int_t spatial_dimension_;
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
  Teuchos::RCP<MultiField_Matrix> matrix_;
  Teuchos::Array<int_t> diag_column_;
  Teuchos::Array<mv_scalar_type> diag_value_;
};
} // namespace DICe

#endif /* DICE_MATRIXSERVICE_H_ */
