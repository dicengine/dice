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

/// \class DICe::Matrix
/// \brief Generic class for matrix storage and operations
template <typename Type, size_t Rows, size_t Cols = Rows>
class DICE_LIB_DLL_EXPORT
Matrix {
public:
  /// default constructor take two args which are the dimensions of the matrix
  /// \param rows number of rows
  /// \param cols number of columns
  Matrix():
    data_(Rows*Cols,0),
    int_work_(Rows*Rows,0),
    scalar_work_(Rows>10?Rows*Rows:10*Rows,0){};

  /// copy constructor
  Matrix(Matrix const & in):
    Matrix<Type,Rows,Cols>(){
    for(size_t i=0;i<Rows*Cols;++i)
      data_[i] = in(i);
  };

  /// initializer list constructor
  Matrix(std::initializer_list<std::initializer_list<Type> > list):
    Matrix<Type,Rows,Cols>(){
    assert(list.size()==Rows);
    assert(list.begin()->size()!=0);
    assert(list.begin()->size()==Cols);
    std::vector<std::initializer_list<Type> > set_vec = list;
    std::vector<std::vector<Type> > v;
    for (auto i = set_vec.begin(); i != set_vec.end(); i++)
      v.push_back(std::vector<Type>(*i));
    for (size_t i = 0; i < Rows; i++)
      for(size_t j=0;j<Cols;++j)
        (*this)(i,j) = v[i][j];
  };

  /// default destructor
  virtual ~Matrix(){};

  /// return the number of rows
  size_t rows() {return Rows;};

  /// return the number of columns
  size_t cols() {return Cols;};

  /// access operator
  Type & operator()(size_t row, size_t col){
    assert(row>=0&&row<Rows);
    assert(col>=0&&col<Cols);
    return data_[row*Cols+col];
  };

  /// access operator
  Type operator()(size_t index)const{
    assert(index>=0&&index<Rows*Cols);
    return data_[index];
  };

  /// return a pointer to the storage of the matrix (for LAPACK calls)
  Type * data() {return &data_[0];};

  /// set all values of a matrix to given value
  void put_value(const Type & value){
    std::fill(data_.begin(), data_.end(), value);
  };

  /// copy another matrix of a different type
  template <typename RType>
  void copy(const Matrix<RType,Rows,Cols> & rhs){
    for(size_t i=0;i<Rows*Cols;++i)
      data_[i] = static_cast<Type>(rhs(i));
  }

  /// assignment operator
  Matrix<Type,Rows,Cols> & operator=(const Matrix<Type,Rows,Cols> & rhs){
    for(size_t i=0;i<Rows*Cols;++i)
      data_[i] = rhs(i);
    return *this;
  };

  /// add operator
  Matrix<Type,Rows,Cols> operator+(Matrix<Type,Rows,Cols> & rhs){
    Matrix<Type,Rows,Cols> ans;
    for(size_t i=0;i<Rows;++i)
      for(size_t j=0;j<Cols;++j)
        ans(i,j) = (*this)(i,j) + rhs(i,j);
    return ans;
  };

  /// subtract operator
  Matrix<Type,Rows,Cols> operator-(Matrix<Type,Rows,Cols> & rhs){
    Matrix<Type,Rows,Cols> ans;
    for(size_t i=0;i<Rows;++i)
      for(size_t j=0;j<Cols;++j)
        ans(i,j) = (*this)(i,j) - rhs(i,j);
    return ans;
  };

  /// matrix times a vector
  std::vector<Type> operator*(std::vector<Type> & rhs){
    TEUCHOS_TEST_FOR_EXCEPTION(rhs.size()!=Cols,std::runtime_error,"vector dim != matrix cols");
    std::vector<Type> ans(rhs.size(),0);
    for(size_t i=0;i<Rows;++i){
      for(size_t j=0;j<Cols;++j){
        ans[i] += (*this)(i,j)*rhs[j];
      }
    }
    return ans;
  };

  /// matrix times a matrix
  template <typename RType,size_t RRows,size_t RCols>
  Matrix<Type,Rows,RCols> operator*(Matrix<RType,RRows,RCols> & rhs){
    static_assert(std::is_same<Type,RType>::value,"scalar type must be the same");
    static_assert(Cols==RRows,"matrix dimensions must be compatible");
    Matrix<Type,Rows,RCols> ans;
    for(size_t i=0;i<Rows;++i){
      for(size_t j=0;j<RCols;++j){
        for(size_t k=0;k<RRows;++k){
          ans(i,j) += (*this)(i,k)*rhs(k,j);
        }
      }
    }
    return ans;
  };

  /// transpose of a matrix
  Matrix<Type,Rows,Cols> transpose(){
    auto trans(*this);
    for(size_t i=0;i<Rows;++i){
      for(size_t j=0;j<Cols;++j){
        trans(i,j) = (*this)(j,i);
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
    TEUCHOS_TEST_FOR_EXCEPTION(Rows!=Cols,std::runtime_error,"matrix must be square");
    Matrix<scalar_t,Rows,Cols> inverse;
    inverse.copy(*this); // lapack only works on scalar type not integers, copy needed to cast values
    int_t info = 0;
    std::fill(int_work_.begin(), int_work_.end(), 0);
    lapack_.GETRF(Rows,Rows,inverse.data(),
      Rows,&int_work_[0],&info);
    std::fill(scalar_work_.begin(), scalar_work_.end(), 0);
    lapack_.GETRI(Rows,inverse.data(),Rows,&int_work_[0],
      &scalar_work_[0],Rows*Rows,&info);
    return inverse;
  }

  scalar_t condition_number(){
    TEUCHOS_TEST_FOR_EXCEPTION(Rows!=Cols,std::runtime_error,"matrix must be square");
    Matrix<scalar_t,Rows,Cols> scalar_mat;
    scalar_mat.copy(*this); // lapack only works on scalar type not integers, copy needed to cast values
    int_t info = 0;
    std::fill(int_work_.begin(), int_work_.end(), 0);
    lapack_.GETRF(Rows,Rows,scalar_mat.data(),
      Rows,&int_work_[0],&info);
    scalar_t rcond = 0.0;
    scalar_t norm = one_norm(scalar_mat);
    lapack_.GECON('1',Rows,scalar_mat.data(),Rows,norm,&rcond,
      &scalar_work_[0],&int_work_[0],&info);
    return rcond==0.0?0.0:1.0/rcond;
  }

  /// make a diagonal matrix
  static Matrix<Type,Rows,Cols> diag(const Type & value){
    static_assert(Rows==Cols,"diag() can only be called for square matrix");
    Matrix<Type,Rows,Cols> ans;
    for(size_t i=0;i<Rows;++i)
      ans(i,i) = value;
    return ans;
  }

  // create an identity matrix
  static Matrix<Type,Rows,Cols> identity(){
    static_assert(Rows==Cols,"diag() can only be called for square matrix");
    return diag(1);
  }

  /// overload the ostream operator to enable std::cout << matrix << std::endl;, etc.
  friend std::ostream & operator<<(std::ostream & os, Matrix<Type,Rows,Cols> & matrix){
    std::ios_base::fmtflags f( std::cout.flags() );
    os.precision(5);
    os << std::scientific;
    os << "--- Matrix ("<<Rows<<"x"<<Cols<<") ---" << std::endl;
    for(size_t i=0;i<Rows;++i){
      std::stringstream row_id;
      row_id << i << ":";
      os << std::setw(4) << std::right << row_id.str();
      for(size_t j=0;j<Cols;++j){
        os << std::setw(14) << std::right << matrix(i,j);
      }
      if(i<Rows-1) os << std::endl;
    }
    std::cout.flags( f );
    return os;
  };

private:
  /// value storage
  std::vector<Type> data_;
  /// work vectors for lapack calls
  std::vector<int_t> int_work_;
  std::vector<scalar_t> scalar_work_;
  /// member lapack object
  Teuchos::LAPACK<int_t,scalar_t> lapack_; // lapack requires scalar_t (not int or other)
};

/// 2 norm of a matrix as a static method
template <typename Type, size_t Rows, size_t Cols>
scalar_t norm(Matrix<Type,Rows,Cols> matrix){
  scalar_t norm = 0.0;
  for(size_t i=0;i<Rows*Cols;++i)
    norm += matrix(i)*matrix(i);
  return std::sqrt(norm);
};

/// function to set all the values of a vector to one value
template <typename Type>
void put_scalar(std::vector<Type> & vec,const Type & value){
  assert(vec.size()!=0);
  for(size_t i=0;i<vec.size();++i)
    vec[i] = value;
}

/// function to reset the values of a vector to zero
template <typename Type>
void zero(std::vector<Type> & vec){
  put_scalar(vec,0.0);
}

/// function to compute the 1-norm of a matrix (max of the column totals)
template <typename Type, size_t Rows, size_t Cols>
scalar_t one_norm(Matrix<Type,Rows,Cols> matrix){
  scalar_t norm = 0.0;
  std::vector<Type> col_totals(Cols,0);
  for(size_t i=0;i<Rows;++i)
    for(size_t j=0;j<Cols;++j)
      col_totals[j] += std::abs(matrix(i,j));
  for(size_t j=0;j<Cols;++j)
    if(col_totals[j] > norm) norm = col_totals[j];
  return norm;
};

/// overload the ostream operator to enable std::cout << vector << std::endl;, etc.
template <typename Type>
std::ostream & operator<<(std::ostream & os, std::vector<Type> & vec){
  std::ios_base::fmtflags f( std::cout.flags() );
  os.precision(5);
  os << std::scientific;
  os << "--- Vector (size: "<<vec.size()<<") ---" << std::endl;
  for(size_t i=0;i<vec.size();++i){
    std::stringstream row_id;
    row_id << i << ":";
    os << std::setw(4) << std::right << row_id.str();
    os << std::setw(14) << std::right << vec[i];
    if(i<vec.size()-1) os << std::endl;
  }
  std::cout.flags( f );
  return os;
};

}// End DICe Namespace

#endif
