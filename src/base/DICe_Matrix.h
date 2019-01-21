// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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

#ifndef DICE_MATRIX_H
#define DICE_MATRIX_H

#include <DICe.h>

#include <Teuchos_LAPACK.hpp>
#include <Teuchos_RCP.hpp>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <array>

namespace DICe {

/// default size to use when the matrix dimension is not known at compile time
//  also, since the Matrix class is allocated always on the stack, this
//  limit is used to ensure that a matrix is not constructed that is too large
#define MAX_MATRIX_DIM 16

/// helper function to turn a type into a string using type_to_string<type>();
template <typename Type>
std::string type_to_string(){
  if (std::is_same<Type, int>::value)
    return "int";
  else if (std::is_same<Type,double>::value)
    return "double";
  else if (std::is_same<Type,bool>::value)
    return "bool";
  else if (std::is_same<Type,float>::value)
    return "float";
  else
    return "unknown";
}

/// \class DICe::Matrix
/// \brief Generic class for matrix storage and operations
/// The idea behind this matrix class is to have a simple matrix (and vector) class
/// that has all it's storage on the stack and several of it's methods optimized.
/// This class is intended for small size matrices that are used in things like objective calculations
/// of other heavily repeated operations. It is not intended for use with dynamic sizes. In most
/// cases the size should be known at compile time. In a few instances the size is not known (for example
/// when there are different sets of shape functions activated by the user, this leads to runtime sizing
/// of the tangent matrix in the subset-based optimization loop). In cases like these, functionality for
/// dynamic sizing has been emulated by making a Matrix of MAX_MATRIX_DIM size and only uses a portion
/// of that allocation.
template <typename Type, size_t Rows=MAX_MATRIX_DIM, size_t Cols = Rows>
class DICE_LIB_DLL_EXPORT
Matrix {
public:
  /// convenience alias for vectors
  template <size_t VRows=MAX_MATRIX_DIM>
  using Vector = Matrix<Type,VRows,1>;

  /// default constructor take two args which are the dimensions of the matrix
  /// \param rows number of rows in use (the Rows template parameter will determine how many rows are allocated on the stack memory)
  /// \param cols number of columns in use (the Cols template parameter will determine how many cols are allocated on the stack memory)
  Matrix(const size_t rows=Rows, const size_t cols=Cols):
  rows_(rows),
  cols_(cols){
    static_assert(Rows>=1&&Rows<=MAX_MATRIX_DIM,"invalid matrix row size");
    static_assert(Cols>=1&&Cols<=MAX_MATRIX_DIM,"invalid matrix col size");
    assert(rows_>0&&rows_<=Rows);
    assert(cols_>0&&cols_<=Cols);
  };

  /// copy constructor
  Matrix(Matrix const & in):
    Matrix<Type,Rows,Cols>(){
    validate_dimensions(in);
    for(size_t i=0;i<rows_*cols_;++i)
      data_[i] = in[i];
  };

  /// initializer list constructor
  Matrix(std::initializer_list<std::initializer_list<Type> > list):
    Matrix<Type,Rows,Cols>(){
    // check if the default constructor was called with no dimensions and fix
    assert(list.size()<=MAX_MATRIX_DIM);
    assert(list.begin()->size()<=MAX_MATRIX_DIM);
    if(rows_==MAX_MATRIX_DIM&&cols_==MAX_MATRIX_DIM){
      rows_ = list.size();
      cols_ = list.begin()->size();
    }
    assert(list.size()==rows_);
    assert(list.begin()->size()!=0);
    assert(list.begin()->size()==cols_);
    std::vector<std::initializer_list<Type> > set_vec = list;
    std::vector<std::vector<Type> > v;
    for (auto i = set_vec.begin(); i != set_vec.end(); i++)
      v.push_back(std::vector<Type>(*i));
    for (size_t i = 0; i < rows_; i++)
      for(size_t j=0;j<cols_;++j)
        (*this)(i,j) = v[i][j];
  };

  /// default destructor
  virtual ~Matrix(){};

  /// return the number of rows
  size_t rows() const {return rows_;};

  /// return the stack storage number of rows
  size_t storage_rows() const {return Rows;};

  /// return the number of columns
  size_t cols() const {return cols_;};

  /// return the stack storage number of columns
  size_t storage_cols() const {return Cols;};

  /// return the number of columns
  size_t size() const {return rows_*cols_;};

  /// return the stack storage number of columns
  size_t storage_size() const {return Rows*Cols;};

  /// validate the matrix dimensions of the rhs for copy, etc.
  template <typename RType, size_t RRows, size_t RCols>
  void validate_dimensions( const Matrix <RType,RRows,RCols> & rhs)const{
    TEUCHOS_TEST_FOR_EXCEPTION(rhs.rows()!=rows_,std::runtime_error,"number of rows must match");
    TEUCHOS_TEST_FOR_EXCEPTION(rhs.cols()!=cols_,std::runtime_error,"number of cols must match");
  }

  /// access operator
  Type & operator()(size_t row, size_t col){
    assert(row>=0&&row<rows_); // assert used only in debug, the release build will skip this for opt
    assert(col>=0&&col<cols_);
    return data_[row*cols_+col];
  };

  /// access operator
  Type & operator()(size_t index){
    assert(index>=0&&index<rows_*cols_);
    return data_[index];
  };

  /// const access operator
  Type operator[](size_t index) const {
    assert(index>=0&&index<rows_*cols_);
    return data_[index];
  };

  /// return a pointer to the storage of the matrix (for LAPACK calls)
  Type * data() {return &data_[0];};

  /// set all values of a matrix to given value
  void put_value(const Type & value){
    std::fill(data_.begin(), data_.end(), value);
  };

  /// copy another matrix of a different type
  template <typename RType, size_t RRows, size_t RCols>
  void copy(const Matrix<RType,RRows,RCols> & rhs){
    validate_dimensions(rhs);
    for(size_t i=0;i<rows_*cols_;++i){
      data_[i] = static_cast<Type>(rhs[i]);
    }
  }

  /// assignment operator
  template <size_t RRows, size_t RCols>
  Matrix<Type,Rows,Cols> & operator=(const Matrix<Type,RRows,RCols> & rhs){
    validate_dimensions(rhs);
    for(size_t i=0;i<rows_*cols_;++i)
      data_[i] = rhs[i];
    return *this;
  };

  /// add operator
  template <size_t RRows, size_t RCols>
  Matrix<Type,Rows,Cols> operator+(const Matrix<Type,RRows,RCols> & rhs){
    validate_dimensions(rhs);
    Matrix<Type,Rows,Cols> ans;
    for(size_t i=0;i<rows_;++i)
      for(size_t j=0;j<cols_;++j)
        ans(i,j) = (*this)(i,j) + rhs[i*cols_+j];
    return ans;
  };

  /// subtract operator
  template <size_t RRows, size_t RCols>
  Matrix<Type,Rows,Cols> operator-(const Matrix<Type,RRows,RCols> & rhs){
    validate_dimensions(rhs);
    Matrix<Type,Rows,Cols> ans;
    for(size_t i=0;i<rows_;++i)
      for(size_t j=0;j<cols_;++j)
        ans(i,j) = (*this)(i,j) - rhs[i*cols_+j];
    return ans;
  };

  /// matrix times a matrix
  template <typename RType,size_t RRows,size_t RCols>
  Matrix<Type,Rows,RCols> operator*(const Matrix<RType,RRows,RCols> & rhs){
    static_assert(std::is_same<Type,RType>::value,"scalar type must be the same to multiply matrices");
    TEUCHOS_TEST_FOR_EXCEPTION(cols_!=rhs.rows(),std::runtime_error,"matrix dimensions are not compatible");
    Matrix<Type,Rows,RCols> ans(rows_,rhs.cols());
    const size_t rhs_cols = rhs.cols();
    const size_t rhs_rows = rhs.rows();
    for(size_t i=0;i<rows_;++i){
      for(size_t j=0;j<rhs_cols;++j){
        for(size_t k=0;k<rhs_rows;++k){
          ans(i,j) += (*this)(i,k)*rhs[k*rhs_cols+j];
        }
      }
    }
    return ans;
  };

  /// transpose of a matrix
  Matrix<Type,Cols,Rows> transpose(){
    Matrix<Type,Cols,Rows> trans(cols_,rows_);
    for(size_t i=0;i<rows_;++i){
      for(size_t j=0;j<cols_;++j){
        trans(j,i) = (*this)(i,j);
      }
    }
    return trans;
  };

  /// 2 norm of a matrix
  scalar_t norm(){
    return norm(*this);
  };

  /// compute the inverse of a matrix and return as a new matrix
  // always has to be scalar type because of lapack
  Matrix<scalar_t,Rows,Cols> inv(){
    TEUCHOS_TEST_FOR_EXCEPTION(rows_!=cols_,std::runtime_error,"matrix must be square");
    Matrix<scalar_t,Rows,Cols> inverse;
    inverse.copy(*this); // lapack only works on scalar type not integers, copy needed to cast values
    int_t info = 0;
    std::fill(int_work_.begin(), int_work_.end(), 0);
    lapack_.GETRF(rows_,rows_,inverse.data(),
      rows_,&int_work_[0],&info);
    std::fill(scalar_work_.begin(), scalar_work_.end(), 0);
    lapack_.GETRI(rows_,inverse.data(),rows_,&int_work_[0],
      &scalar_work_[0],rows_*rows_,&info);
    return inverse;
  }

  scalar_t condition_number(){
    TEUCHOS_TEST_FOR_EXCEPTION(rows_!=cols_,std::runtime_error,"matrix must be square");
    Matrix<scalar_t,Rows,Cols> scalar_mat;
    scalar_mat.copy(*this); // lapack only works on scalar type not integers, copy needed to cast values
    int_t info = 0;
    std::fill(int_work_.begin(), int_work_.end(), 0);
    lapack_.GETRF(rows_,rows_,scalar_mat.data(),
      rows_,&int_work_[0],&info);
    scalar_t rcond = 0.0;
    scalar_t norm = one_norm(scalar_mat);
    lapack_.GECON('1',rows_,scalar_mat.data(),rows_,norm,&rcond,
      &scalar_work_[0],&int_work_[0],&info);
    return rcond==0.0?0.0:1.0/rcond;
  }

  /// make a diagonal matrix
  static Matrix<Type,Rows,Cols> diag(const Type & value, const size_t size=Rows){
    static_assert(Rows==Cols,"diag() can only be called for square matrix");
    Matrix<Type,Rows,Cols> ans(size,size);
    for(size_t i=0;i<Rows;++i)
      ans(i,i) = value;
    return ans;
  }

  // create an identity matrix
  static Matrix<Type,Rows,Cols> identity(const size_t size=Rows){
    static_assert(Rows==Cols,"diag() can only be called for square matrix");
    return diag(1,size);
  }

  /// overload the ostream operator to enable std::cout << matrix << std::endl;, etc.
  friend std::ostream & operator<<(std::ostream & os, Matrix<Type,Rows,Cols> & matrix){
    std::ios_base::fmtflags f( std::cout.flags() );
    os.precision(5);
    os << std::scientific;
    if(matrix.cols()==1||matrix.rows()==1){
      os << std::endl << "--- Vector  ---" << std::endl << std::endl;
    }else{
      os << std::endl << "--- Matrix  ---" << std::endl << std::endl;
    }
    os << "    value type:   " << type_to_string<Type>() << std::endl;
    os << "    active dims:  ("<<matrix.rows()<<"x"<<matrix.cols()<<")" << std::endl;
    os << "    storage dims: ("<<Rows<<"x"<<Cols<<")" << std::endl << std::endl;
    for(size_t i=0;i<matrix.rows();++i){
      std::stringstream row_id;
      row_id << i << ":";
      os << std::setw(4) << std::right << row_id.str();
      for(size_t j=0;j<matrix.cols();++j){
        os << std::setw(14) << std::right << matrix(i,j);
      }
      //if(i<matrix.rows()-1) os << std::endl;
      os << std::endl;
    }
    std::cout.flags( f );
    return os;
  };

  /// 2 norm of a matrix as a static method
  static scalar_t norm(Matrix<Type,Rows,Cols> matrix){
    scalar_t norm = 0.0;
    for(size_t i=0;i<matrix.rows()*matrix.cols();++i)
      norm += matrix(i)*matrix(i);
    return std::sqrt(norm);
  };

private:
  /// number of rows (may not be equal to Rows if the dimensions are runtime defined)
  size_t rows_;
  /// number of columns (may not be equal to Rows if the dimensions are runtime defined)
  size_t cols_;
  /// value storage (using array to have the storage be on the stack not the heap like a vector)
  std::array<Type,Rows*Cols> data_{}; // the {} are for zero-initialization
  /// work vectors for lapack calls
  std::array<int_t,Rows*Rows> int_work_{};
  std::array<scalar_t,(Rows>10)?Rows*Rows:10*Rows> scalar_work_{};
  /// member lapack object
  Teuchos::LAPACK<int_t,scalar_t> lapack_; // lapack requires scalar_t (not int or other)
};

/// convenience alias for vectors
template <typename Type, size_t Rows=MAX_MATRIX_DIM>
using Vector = Matrix<Type,Rows,1>;

/// function to set all the values of a vector to one value
template <typename Type>
void put_value(Vector<Type> & vec,const Type & value){
  static_assert(vec.cols()==1,"invalid column dimension for a vector");
  assert(vec.size()!=0);
  for(size_t i=0;i<vec.size();++i)
    vec(i) = value;
}

/// function to reset the values of a vector to zero
template <typename Type>
void zero(Vector<Type> & vec){
  put_value(vec,0.0);
}

/// 2 norm of a matrix as a static method
template <typename Type, size_t Rows, size_t Cols>
scalar_t norm(Matrix<Type,Rows,Cols> matrix){
  return Matrix<Type,Rows,Cols>::norm(matrix);
};

/// function to compute the 1-norm of a matrix (max of the column totals)
template <typename Type, size_t Rows, size_t Cols>
scalar_t one_norm(Matrix<Type,Rows,Cols> matrix){
  scalar_t norm = 0.0;
  Vector<Type,Cols> col_totals;
  for(size_t i=0;i<matrix.rows();++i)
    for(size_t j=0;j<matrix.cols();++j)
      col_totals(j) += std::abs(matrix(i,j));
  for(size_t j=0;j<matrix.cols();++j)
    if(col_totals[j] > norm) norm = col_totals[j];
  return norm;
};

}// End DICe Namespace

#endif
