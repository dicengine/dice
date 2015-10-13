// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact:
//              Dan Turner   (danielzturner@gmail.com)
//
// ************************************************************************
// @HEADER

#ifndef DICE_UTILITIES_H
#define DICE_UTILITIES_H

#include <DICe.h>

#include <Teuchos_ArrayRCP.hpp>

#include <utility>
#include <cassert>

namespace DICe {

/// \class DICe::Tensor_2D
/// \brief Small helper class for tensor operations (used mostly in computing strains)

template<class Scalar>
struct DICE_LIB_DLL_EXPORT
Tensor_2D {
public:
  // All member data is public
  /// XX component of the tensor
  Scalar xx;
  /// XY component of the tensor
  Scalar xy;
  /// YX component of the tensor
  Scalar yx;
  /// YY component of the tensor
  Scalar yy;

  /// Default constructor initializes all values to zero
  Tensor_2D(){
    xx=0.0;
    xy=0.0;
    yx=0.0;
    yy=0.0;}

  /// \brief Constructor that takes as arguments the values of all tensor components
  /// \param xx_ Value of xx component
  /// \param xy_ Value of xy component
  /// \param yx_ Value of yx component
  /// \param yy_ Value of yy component
  Tensor_2D(const Scalar & xx_,
    const Scalar & xy_,
    const Scalar & yx_,
    const Scalar & yy_){
    xx=xx_;
    xy=xy_;
    yx=yx_;
    yy=yy_;}

  /// \brief Constructor that creates an identity tensor scaled by alpha
  /// \param alpha The value to place on the diagonal
  Tensor_2D(const Scalar & alpha){
    xx=alpha;
    xy=0.0;
    yx=0.0;
    yy=alpha;}

  /// \brief Adds two tensors
  /// \param other The other tensor to add to this one
  const Tensor_2D operator+(const Tensor_2D & other)const{
    Tensor_2D result;
    result.xx = xx + other.xx;
    result.xy = xy + other.xy;
    result.yx = yx + other.yx;
    result.yy = yy + other.yy;
    return result;}

  /// \brief Subtracts two tensors
  /// \param other The other tensor to subtract from this one
  const Tensor_2D operator-(const Tensor_2D & other)const{
    Tensor_2D result;
    result.xx = xx - other.xx;
    result.xy = xy - other.xy;
    result.yx = yx - other.yx;
    result.yy = yy - other.yy;
    return result;}

  /// \brief Multiplies two tensors
  /// \param other The other tensor to multiply with this one
const Tensor_2D operator*(const Tensor_2D & other)const{
    Tensor_2D result;
    result.xx=xx*other.xx+xy*other.yx;
    result.xy=xx*other.xy+xy*other.yy;
    result.yx=yx*other.xx+yy*other.yx;
    result.yy=yx*other.xy+yy*other.yy;
    return result;}

  /// \brief scales this tensor by alpha
  /// \param alpha The value to use for scaling
  const Tensor_2D operator*(const Scalar & alpha)const{
    Tensor_2D result;
    result.xx = xx*alpha;
    result.xy = xy*alpha;
    result.yx = yx*alpha;
    result.yy = yy*alpha;
    return result;}

  /// Returns the transpose of this tensor
  const Tensor_2D transpose() const{
    Tensor_2D transpose;
    transpose.xx=xx;
    transpose.yx=xy;
    transpose.xy=yx;
    transpose.yy=yy;
    return transpose;}

  /// Returns the inverse of this tensor
  const Tensor_2D inverse() const{
    Tensor_2D inv;
    const Scalar denom = xx*yy - xy*yx;
    assert(denom != 0.0 && "  DICe ERROR: The denominator cannot be zero.");
    inv.xx =  1.0/denom * yy;
    inv.xy = -1.0/denom * xy;
    inv.yx = -1.0/denom * yx;
    inv.yy =  1.0/denom * xx;
    return inv;
  }

  /// Returns the trace of a tensor \f$ \sum_i T_{ii} \f$
  const Scalar trace()const{ return xx + yy;}
};

}// End DICe Namespace

#endif
