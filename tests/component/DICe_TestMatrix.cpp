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

#include <DICe.h>
#include <DICe_Matrix.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t error_flag  = 0;
  scalar_t error_tol = 1.0E-3;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // create an image and take the polar transform:
  *outStream << "testing constructor" << std::endl;
  Matrix<scalar_t,3,3> mat;
  *outStream << "testing access operator" << std::endl;
  mat(0,0) = 34.89;
  mat(1,0) = 438.9;
  mat(2,0) = 89.190;
  mat(0,1) = 356.89;
  mat(1,1) = -12;
  mat(2,1) = 34.67;
  mat(0,2) = 390.4;
  mat(1,2) = 45.987;
  mat(2,2) = 127.99;
  *outStream << mat << std::endl;
  *outStream << "testing condition number" << std::endl;
  scalar_t cond_num = mat.condition_number();
  *outStream << "condition number: " << cond_num << std::endl;
  if(std::abs(cond_num - 11.538)>error_tol){
    *outStream << "***error: condition number incorrect " << std::endl;
    error_flag++;
  }
  *outStream << "testing inverse method" << std::endl;
  auto inverse = mat.inv();
  *outStream << inverse << std::endl;
  Matrix<scalar_t,3> gold_inv; // testing default template arg
  gold_inv(0,0) = 2.53765e-04;
  gold_inv(1,0) = 4.22151e-03;
  gold_inv(2,0) = -1.32036e-03;
  gold_inv(0,1) = 2.60581e-03;
  gold_inv(1,1) = 2.46078e-03;
  gold_inv(2,1) = -2.48244e-03;
  gold_inv(0,2) = -1.71032e-03;
  gold_inv(1,2) = -1.37608e-02;
  gold_inv(2,2) = 1.27325e-02;
  scalar_t mat_norm = norm(inverse - gold_inv);
  *outStream << "inv error norm: " << mat_norm << std::endl;
  if(mat_norm>error_tol){
    *outStream << "***error: inverse error norm too high" << std::endl;
    error_flag++;
  }
  *outStream << "testing inverse for integers and array constructor" << std::endl;
  Matrix<int_t,4> int_mat
  ({{1,2,3,4},
    {5,6,7,8},
    {9,10,11,12},
    {13,14,15,16}});
  *outStream << int_mat << std::endl;
  cond_num = int_mat.condition_number();
  *outStream << "condition number: " << cond_num << std::endl;
  if(std::abs(cond_num)>error_tol){
    *outStream << "***error: condition number incorrect" << std::endl;
    error_flag++;
  }
  auto int_inverse = int_mat.inv();
  *outStream << int_inverse << std::endl;

  *outStream << "testing copy constructor, put_value(), add and subtract" << std::endl;
  Matrix<int_t,4,9> A;
  A.put_value(4);
  auto B(A);
  B.put_value(12);
  auto C = A + B;
  auto D(A);
  D.put_value(16);
  mat_norm = norm(D-C);
  *outStream << "add subtract error norm: " << mat_norm << std::endl;
  if(mat_norm>error_tol){
    error_flag++;
    *outStream << "***error: add subtract error norm too high" << std::endl;
  }

  *outStream << "testing transpose" << std::endl;
  Matrix<int_t,4> int_mat_trans_gold
  ({{1,5,9,13},
    {2,6,10,14},
    {3,7,11,15},
    {4,8,12,16}});
  auto int_mat_trans = int_mat.transpose();
  mat_norm = norm(int_mat_trans-int_mat_trans_gold);
  *outStream << "transpose error norm: " << mat_norm << std::endl;
  if(mat_norm>error_tol){
    error_flag++;
    *outStream << "***error: transpose error norm too high" << std::endl;
  }

  *outStream << "testing transpose for non-square matrix" << std::endl;
  Vector<scalar_t> sample_vec(4);
  sample_vec.put_value(23.45);
  *outStream << sample_vec << std::endl;
  auto sample_vec_trans = sample_vec.transpose();
  *outStream << sample_vec_trans << std::endl;

  if(sample_vec_trans.size()!=4){
    *outStream << "***error: invalid size dimensions" << std::endl;
    error_flag++;
  }
  if(sample_vec_trans.rows()!=1){
    *outStream << "***error: invalid row dimensions" << std::endl;
    error_flag++;
  }
  if(sample_vec_trans.cols()!=4){
    *outStream << "***error: invalid col dimensions" << std::endl;
    error_flag++;
  }
  if(sample_vec_trans.storage_size()!=16){
    *outStream << "***error: invalid storage dimensions" << std::endl;
    error_flag++;
  }

  *outStream << "testing matrix vector multiply" << std::endl;
  Vector<int_t,4> rhs;
  rhs.put_value(1);
  auto multiply = int_mat * rhs;
  *outStream << multiply << std::endl;
  std::vector<int_t> multiply_gold = {10,26,42,58};
  if(multiply.size()!=multiply_gold.size()){
    *outStream << "***error: multiply vector is the wrong size" << std::endl;
    error_flag++;
  }
  scalar_t vec_norm = 0.0;
  for(size_t i=0;i<multiply_gold.size();++i)
    vec_norm += std::abs(multiply_gold[i] - multiply[i]);
  *outStream << "vec multiply error: " << vec_norm << std::endl;
  if(vec_norm > error_tol){
    *outStream << "***error: matrix multiply error too large" << std::endl;
    error_flag++;
  }

  *outStream << "testing multiply fails for wrong vector dims" << std::endl;
  Vector<int_t,5> rhs_wrong_size;
  bool exception_thrown = false;
  try
  {
    int_mat * rhs_wrong_size;
  }
  catch(std::exception &e){
    std::cout << e.what() << '\n';
    exception_thrown = true;
  }
  if(!exception_thrown){
    error_flag++;
    *outStream << "***error: calling matrix multiple on a vector of the wrong size should have thrown an exception" << std::endl;
  }

  *outStream << "testing that calling inv() on non-square throws an exception" << std::endl;
  Matrix<scalar_t,4,8> non_square;
  exception_thrown = false;
  try
  {
    non_square.inv();
  }
  catch(std::exception &e){
    std::cout << e.what() << '\n';
    exception_thrown = true;
  }
  if(!exception_thrown){
    error_flag++;
    *outStream << "***error: calling inv() on a non-square matrix should have thrown an exception" << std::endl;
  }

  // test matrix matrix multiply
  Matrix<scalar_t,1,4> M = {{1,2,3.9,9.0}};
  Matrix<scalar_t,4,1> P = {{1},{2},{3.9},{9}};
  auto Q = M*P;
  *outStream << Q << std::endl;
  mat_norm = norm(Q)-101.21;
  if(std::abs(mat_norm)>error_tol){
    *outStream << "***error: matrix matrix multiply incorrect" << std::endl;
    error_flag++;
  }
  auto mat_times_mat_inv = mat*inverse;
  *outStream << mat_times_mat_inv << std::endl;
  auto identity = Matrix<scalar_t,3>::identity();
  *outStream << identity << std::endl;
  mat_norm = norm(mat_times_mat_inv - identity);
  *outStream << "matrix times inverse error norm: " << mat_norm << std::endl;
  if(std::abs(mat_norm)>error_tol){
    *outStream << "***error: matrix times inverse incorrect" << std::endl;
    error_flag++;
  }

  *outStream << "testing multiplying fixed size matrix with run-time sized matrix" << std::endl;

  Matrix<int_t> runtime_mat;
  runtime_mat = {{1,5,9},
    {2,6,10},
    {3,7,11},
    {4,8,12}};
  auto runtime_trans = runtime_mat.transpose();
  *outStream << runtime_mat << std::endl;
  auto runtime_times_fixed = int_mat * runtime_trans.transpose();
  *outStream << runtime_times_fixed << std::endl;
  Matrix<int_t,4,3> fixed_times_runtime_gold = {{30,70,110},{70,174,278},{110,278,446},{150,382,614}};
  scalar_t fixed_runtime_norm = norm(fixed_times_runtime_gold - runtime_times_fixed);
  *outStream << "fixed times runtime sized matrices error norm: " << fixed_runtime_norm << std::endl;
  if(std::abs(fixed_runtime_norm)>error_tol){
    *outStream << "***error: multiplying fixed times a runtime sized matrix failed" << std::endl;
    error_flag++;
  }

  *outStream << "testing vector add operation" << std::endl;

  Vector<int_t,4> int_vec = {{1},{2},{3},{4}};
  Vector<int_t,4> int_vec_rhs;
  int_vec_rhs(3) = 10;
  int_vec_rhs(1) = 12;
  auto result_int_add = int_vec + int_vec_rhs;
  Matrix<int_t,4,1> result_int_add_gold = {{1},{14},{3},{14}};
  *outStream << result_int_add << std::endl;
  scalar_t int_vec_add_norm = norm(result_int_add_gold - result_int_add);
  *outStream << "int add vector norm: " << int_vec_add_norm << std::endl;
  if(std::abs(int_vec_add_norm)>error_tol){
    *outStream << "***error: int vector add error" << std::endl;
    error_flag++;
  }

  *outStream << "testing all zero method" << std::endl;
  Matrix<scalar_t,3,6> no_values;
  if(!no_values.all_values_are_zero()){
    *outStream << "***error: all_values_are_zero() error" << std::endl;
    error_flag++;
  }
  Matrix<bool,3,6> some_bool_values;
  some_bool_values(1,4) = true;
  if(some_bool_values.all_values_are_zero()){
    *outStream << "***error: all_values_are_zero() error" << std::endl;
    error_flag++;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (error_flag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

