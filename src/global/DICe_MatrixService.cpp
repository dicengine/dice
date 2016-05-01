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

#include <DICe_MatrixService.h>

namespace DICe {

Matrix_Service::Matrix_Service(const int_t spatial_dimension) :
    spatial_dimension_(spatial_dimension),
    row_bc_register_(0),
    col_bc_register_(0),
    row_bc_register_size_(0),
    col_bc_register_size_(0),
    bc_register_initialized_(false)
{
  diag_value_.push_back(0);
  diag_column_.push_back(0);
}

void
Matrix_Service::initialize_bc_register(const int_t row_register_size, const int_t col_register_size)
{
  row_bc_register_size_ = row_register_size;
  row_bc_register_ = new bool[row_bc_register_size_];
  for(int_t i=0;i<row_bc_register_size_;++i)
    row_bc_register_[i] = false;

  col_bc_register_size_ = col_register_size;
  col_bc_register_ = new bool[col_bc_register_size_];
  for(int_t i=0;i<col_bc_register_size_;++i)
    col_bc_register_[i] = false;
  bc_register_initialized_ = true;
}

void
Matrix_Service::clear_bc_register()
{
  for(int_t i=0;i<row_bc_register_size_;++i)
    row_bc_register_[i] = false;
  for(int_t i=0;i<col_bc_register_size_;++i)
    col_bc_register_[i] = false;
}

void
Matrix_Service::apply_bc_diagonals(const mv_scalar_type & diagonal_value)
{
  diag_value_[0] = diagonal_value;
  for(int_t row=0;row<row_bc_register_size_;++row)
  {
    if(row_bc_register_[row])
    {
      diag_column_[0] = row;
      matrix_->replace_local_values(row,diag_column_,diag_value_);
    }
  }
}

}// End DICe Namespace
