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

#ifndef DICE_TRIANGULATION_H
#define DICE_TRIANGULATION_H

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Matrix.h>
#include <DICe_Camera.h>
#include <DICe_CameraSystem.h>
#ifdef DICE_TPETRA
  #include "DICe_MultiFieldTpetra.h"
#else
  #include "DICe_MultiFieldEpetra.h"
#endif

#include <Teuchos_SerialDenseMatrix.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <cassert>
#include <vector>

namespace DICe {

/// free function to estimate a 9 parameter affine projection
/// \param proj_xl x coordinates of the points in the left image or coordinate system
/// \param proj_yl y coordinates of the points in the left image or coordinate system
/// \param proj_xr x coordinates of the points in the right image or coordinate system
/// \param proj_yr y coordinates of the points in the right image or coordinate system
/// return value affine_matrix storage for the affine parameters (must be 3x3)
DICE_LIB_DLL_EXPORT
Teuchos::SerialDenseMatrix<int_t,double>
compute_affine_matrix(const std::vector<scalar_t> proj_xl,
  const std::vector<scalar_t> proj_yl,
  const std::vector<scalar_t> proj_xr,
  const std::vector<scalar_t> proj_yr);

/// \class DICe::Triangulation
/// \brief A class for computing the triangulation of 3d points from two correlation and a calibration file
///
class DICE_LIB_DLL_EXPORT
Triangulation{
public:
  /// \brief Default constructor
  /// \param param_file_name the name of the file to parse the calibration parameters from
  Triangulation(const std::string & param_file_name):
  Triangulation(){
    load_calibration_parameters(param_file_name);
  };

  /// \brief constructor with no args
  Triangulation(){
    warp_params_ = Teuchos::rcp(new std::vector<scalar_t>(12,0.0)); /// at max there are 12 parameters that must be set (for the quadratic)
    (*warp_params_)[1] = 1.0;
    (*warp_params_)[8] = 1.0;
    projective_params_ = Teuchos::rcp(new std::vector<scalar_t>(9,0.0));
    (*projective_params_)[0] = 1.0;
    (*projective_params_)[4] = 1.0;
    (*projective_params_)[8] = 1.0;
    cam_0_to_cam_1_ = Matrix<scalar_t,4>::identity();
    cam_0_to_world_ = Matrix<scalar_t,4>::identity();
    cal_intrinsics_.clear();
    for(int_t i=0;i<2;++i) // one vec for each camera
      cal_intrinsics_.push_back(std::vector<scalar_t>(Camera::MAX_CAM_INTRINSIC_PARAM,0.0));
  };

  /// Pure virtual destructor
  virtual ~Triangulation(){}

  /// returns a pointer to the calibration intrinsics
  std::vector<std::vector<scalar_t> > * cal_intrinsics(){
    return & cal_intrinsics_;
  }

  /// returns a pointer to the camera_system
  Teuchos::RCP<Camera_System> camera_system(){
    return camera_system_;
  }

  /// provides access to image width from a triangulation
  const int_t image_width()const{
    if(camera_system_==Teuchos::null) return -1;
    if(camera_system_->num_cameras()<1) return -1;
    return camera_system_->camera(0)->image_width();
  }

  /// provides access to image height from a triangulation
  const int_t image_height()const{
    if(camera_system_==Teuchos::null) return -1;
    if(camera_system_->num_cameras()<1) return -1;
    return camera_system_->camera(0)->image_height();
  }

  /// utility function to undistort point locations
  void undistort_points(std::vector<scalar_t> & points_x,
    std::vector<scalar_t> & points_y,
    const int_t camera_id) const;

  /// provide access to epipolar error in camera calibration
  const scalar_t avg_epipolar_error()const{return camera_system_->avg_epipolar_error();}

  /// returns a pointer to the transform from camera 0 to camera 1
  const Matrix<scalar_t,4> * cam_0_to_cam_1() const {
    return & cam_0_to_cam_1_;
  }

  /// returns a pointer to the camera 0 to world extrinsics
  const Matrix<scalar_t,4> * cam_0_to_world() const {
    return & cam_0_to_world_;
  }

  /// triangulate the optimal point in 3D.
  /// returns the the max value of the psuedo matrix
  /// global coordinates are always defined with camera 0 as the origin
  /// unless another transformation is requested by specifying a transformation file
  /// \param x0 sensor x coordinate of the point in camera 0
  /// \param y0 sensor y coordinate of the point in camera 0
  /// \param x1 sensor x coordinate of the point in camera 1
  /// \param y1 sensor y coordinate of the point in camera 1
  /// \param xc_out global x position in camera 0 coords
  /// \param yc_out global y position in camera 0 coords
  /// \param zc_out global z position in camera 0 coords
  /// \param xw_out global x position in world coords
  /// \param yw_out global y position in world coords
  /// \param zw_out global z position in world coords
  /// \param correct_lens_distortion correct for lens distortion
  scalar_t triangulate(const scalar_t & x0,
    const scalar_t & y0,
    const scalar_t & x1,
    const scalar_t & y1,
    scalar_t & xc_out,
    scalar_t & yc_out,
    scalar_t & zc_out,
    scalar_t & xw_out,
    scalar_t & yw_out,
    scalar_t & zw_out,
    const bool correct_lens_distortion = false) const;

  /// triangulate the optimal point in 3D (from 2d data with calibration).
  /// global coordinates are always defined with camera 0 as the origin
  /// unless another transformation is requested by specifying a transformation file
  /// \param image_x vector of image x coordinates
  /// \param image_y vector of image y coordinates
  /// \param xw_out vector of global x positions in world coords
  /// \param yw_out vector of global y positions in world coords
  /// \param zw_out vector of global z positions in world coords
  void triangulate(const std::vector<scalar_t> & image_x,
    const std::vector<scalar_t> & image_y,
    std::vector<scalar_t> & xw_out,
    std::vector<scalar_t> & yw_out,
    std::vector<scalar_t> & zw_out) const;

  /// compute the fundamental matrix and return it as an opencv mat
  cv::Mat fundamental_matrix() const{
    DICe::Matrix<DICe::scalar_t,3> F = camera_system_->fundamental_matrix();
    cv::Mat matF(3, 3, CV_32F);
    for(int_t i=0;i<matF.rows;++i){
      for(int_t j=0;j<matF.cols;++j){
        matF.at<float>(i,j) = F(i,j);
      }
    }
    return matF;
  }

  /// correct the lens distortion with a radial model
  /// \param x_s x sensor coordinate to correct, modified in place
  /// \param y_s y sensor coordinate to correct, modified in place
  /// \param camera_id either 0 or 1
  void correct_lens_distortion_radial(scalar_t & x_s,
    scalar_t & y_s,
    const int_t camera_id) const;

  /// estimate the projective transform from the left to right image
  /// \param left_img pointer to the left image
  /// \param right_img pointer to the right image
  /// \param output_projected_image true if an image should be output using the solution projection parameters
  /// \param use_nonlinear_projection true if an additional 12 nonlinear parameter projection model should be used
  /// \param processor_id the id of the process calling this routine
  /// returns 0 if successful -1 if the linear projection failed -2 if the nonlinear projection fails, -3 for other error
  int_t estimate_projective_transform(Teuchos::RCP<Image> left_img,
    Teuchos::RCP<Image> right_img,
    const bool output_projected_image = false,
    const bool use_nonlinear_projection = false,
    const int_t processor_id = 0);

  /// determine the corresponding right sensor coordinates given the left
  /// using the projective transform
  /// \param xl left x sensor coord
  /// \param yl left y sensor coord
  /// \param xr [out] right x sensor coord
  /// \param yr [out] right y sensor coord
  void project_left_to_right_sensor_coords(const scalar_t & xl,
    const scalar_t & yl,
    scalar_t & xr,
    scalar_t & yr);

  /// set the warp parameter vector of the triangulation
  /// \param params the projective parameters
  void set_warp_params(Teuchos::RCP<std::vector<scalar_t> > & params){
    TEUCHOS_TEST_FOR_EXCEPTION(params->size()!=12,std::runtime_error,"Error, params vector is the wrong size");
    warp_params_ = params;
  }

  /// return a pointer to the warp parameters
  Teuchos::RCP<std::vector<scalar_t> > warp_params()const{
    return warp_params_;
  }

  /// set the projective parameter vector of the triangulation
  /// \param params the projective parameters
  void set_projective_params(Teuchos::RCP<std::vector<scalar_t> > & params){
    TEUCHOS_TEST_FOR_EXCEPTION(params->size()!=9,std::runtime_error,"Error, params vector is the wrong size");
    projective_params_ = params;
  }

  /// return a pointer to the projective parameters
  Teuchos::RCP<std::vector<scalar_t> > projective_params()const{
    return projective_params_;
  }

  /// determine the best fit plane to the complete set of X Y Z coordinates (excluding any failed points)
  /// \param cx pointer to the x coordinates from the initial triangulation
  /// \param cy pointer to the y coordinates from the initial triangulation
  /// \param cz pointer to the z coordinates from the intiial triangulation
  /// \param sigma pointer to the sigma field
  void best_fit_plane(Teuchos::RCP<MultiField> & cx,
    Teuchos::RCP<MultiField> & cy,
    Teuchos::RCP<MultiField> & cz,
    Teuchos::RCP<MultiField> & sigma);


  /// returns the cosine of the angle between two vectors
  /// \param a vector a, must have three components
  /// \param b vector b, must have three components
  scalar_t cosine_of_two_vectors(const std::vector<scalar_t> & a,
    const std::vector<scalar_t> & b);

  /// set the transform to the identity matrix
  void reset_cam_0_to_world(){
    cam_0_to_world_ = Matrix<scalar_t,4>::identity();
  }

private:
  /// \brief load the calibration parameters
  /// \param param_file_name File name of the cal parameters file
  void load_calibration_parameters(const std::string & param_file_name);

  /// vector camera intrinsics vectors, one for camera 0 and one for camera 1
  /// See Camera::Cam_Intrinsic_Param for the ordering of the parameters in the vector
  std::vector<std::vector<scalar_t> > cal_intrinsics_;

  /// transformation from camera 0 to camera 1 coordinates
  Matrix<scalar_t,4> cam_0_to_cam_1_;

  /// transformation from camera 0 to world model/physical coordinates
  Matrix<scalar_t,4> cam_0_to_world_;

  // both the warp_params and projective_params are for doing a global warp of the
  // image from one camera to another to help initialize the cross-correlation, they are not
  // associated with the calibration

  /// 12 parameters that define a warping (independent from intrinsic and extrinsic parameters)
  Teuchos::RCP<std::vector<scalar_t> > warp_params_;
  /// 8 parameters that define a projective transform (independent from intrinsic and extrinsic parameters)
  Teuchos::RCP<std::vector<scalar_t> > projective_params_;

  /// save off a pointer to the camera system for use in the triangulation
  Teuchos::RCP<Camera_System> camera_system_;

};

DICE_LIB_DLL_EXPORT
void update_legacy_txt_cal_input(const Teuchos::RCP<Teuchos::ParameterList> & input_params);

DICE_LIB_DLL_EXPORT
void update_vic3d_cal_input(const Teuchos::RCP<Teuchos::ParameterList> & input_params,
		const int_t im_width = -1,
		const int_t im_height = -1);


}// End DICe Namespace

#endif
