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

#ifndef DICE_TRIANGULATION_H
#define DICE_TRIANGULATION_H

#include <DICe.h>

#include <Teuchos_SerialDenseMatrix.hpp>

#include <cassert>
#include <vector>

namespace DICe {

/// \class DICe::Triangulation
/// \brief A class for computing the triangulation of 3d points from two correlation and a calibration file
///
class DICE_LIB_DLL_EXPORT
Triangulation{
public:
  /// \brief Default constructor
  Triangulation():
    has_cal_params_(false){};

  /// Pure virtual destructor
  virtual ~Triangulation(){}

  /// \brief load the calibration parameters
  /// \param param_file_name File name of the cal parameters file
  void load_calibration_parameters(const std::string & param_file_name);

  /// returns true if the cal parameters have been loaded
  bool has_cal_params()const{
    return has_cal_params_;
  }

  /// returns a pointer to the calibration intrinsics
  std::vector<std::vector<scalar_t> > * cal_intrinsics(){
    return & cal_intrinsics_;
  }

  /// returns a pointer to the calibration extrinsics
  std::vector<std::vector<scalar_t> > * cal_extrinsics(){
    return & cal_extrinsics_;
  }

  /// returns a pointer to the camera 0 to world extrinsics
  std::vector<std::vector<scalar_t> > * trans_extrinsics(){
    return & trans_extrinsics_;
  }

  /// triangulate the optimal point in 3D.
  /// global coordinates are always defined with camera 0 as the origin
  /// unless another transformation is requested by specifying a transformation file
  /// \param x0 sensor x coordinate of the point in camera 0
  /// \param y0 sensor y coordinate of the point in camera 0
  /// \param x1 sensor x coordinate of the point in camera 1
  /// \param y1 sensor y coordinate of the point in camera 1
  /// \param x_out global x position in world coords
  /// \param y_out global y position in world coords
  /// \param z_out global z position in world coords
  void triangulate(const scalar_t & x0,
    const scalar_t & y0,
    const scalar_t & x1,
    const scalar_t & y1,
    scalar_t & x_out,
    scalar_t & y_out,
    scalar_t & z_out);

  /// convert Cardan-Bryant angles and offsets to T matrix format
  /// \param alpha euler angle in degrees
  /// \param beta euler angle in degrees
  /// \param gamma euler angle in degrees
  /// \param tx offset x
  /// \param ty offset y
  /// \param tz offset z
  void convert_CB_angles_to_T(const scalar_t & alpha,
    const scalar_t & beta,
    const scalar_t & gamma,
    const scalar_t & tx,
    const scalar_t & ty,
    const scalar_t & tz,
    Teuchos::SerialDenseMatrix<int_t,double> & T_out);

  /// computes and returns in place the inverse of a transform
  /// \param T_out the matrix to invert
  void invert_transform(Teuchos::SerialDenseMatrix<int_t,double> & T_out);

protected:
  /// true if the calibration parameters have been loaded
  bool has_cal_params_;
  /// 2 x 8 matrix of camera intrinsics
  /// first index is the camera id, second index is cx cy fx fy fs k1 k2 k3
  std::vector<std::vector<scalar_t> > cal_intrinsics_;
  /// 4 x 4 matrix of calibration extrinsics
  /// R11 R12 R13 tx
  /// R21 R22 R23 ty
  /// R31 R32 R33 tz
  /// 0   0   0   1
  std::vector<std::vector<scalar_t> > cal_extrinsics_;
  /// 4 x 4 matrix to convert camera 0 coordinates to world coordinate system
  /// with input values defined in terms of transforming from world to camera 0 coordinates
  /// (the matrix gets inverted when stored so the transform is camera 0 to world when applied)
  /// R11 R12 R13 tx
  /// R21 R22 R23 ty
  /// R31 R32 R33 tz
  /// 0   0   0   1
  std::vector<std::vector<scalar_t> > trans_extrinsics_;
};

}// End DICe Namespace

#endif
