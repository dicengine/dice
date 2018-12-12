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
#include <DICe_LocalShapeFunction.h>
#include <DICe_Camera.h>
#include <DICe_Parser.h>
#include <fstream>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <DICe_XMLUtils.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Array.hpp>
#include <math.h>




namespace DICe {
  namespace CAM = DICe_Camera;
  namespace LSFunc = DICe_LocalShapeFunction;
  DICE_LIB_DLL_EXPORT
    void Camera::clear_camera() {

    //clear the intrinsic and extrinsic values
    intrinsics_.assign(CAM::MAX_CAM_INTRINSIC_PARAMS, 0.0);
    extrinsics_.assign(CAM::MAX_CAM_EXTRINSIC_PARAMS, 0.0);

    //clear the camera to world coordinate transform values
    cam_world_trans_.clear();
    cam_world_trans_.resize(4);
    cam_world_trans_[0].assign(4, 0);
    cam_world_trans_[1].assign(4, 0);
    cam_world_trans_[2].assign(4, 0);
    cam_world_trans_[3].assign(4, 0);

    //clear the 3x3 rotation matrix
    rotation_3x3_matrix_.clear();
    rotation_3x3_matrix_.resize(3);
    rotation_3x3_matrix_[0].assign(3, 0.0);
    rotation_3x3_matrix_[1].assign(3, 0.0);
    rotation_3x3_matrix_[2].assign(3, 0.0);

    //clear the camera to world coordinate transform values
    cam_world_trans_.clear();
    cam_world_trans_.resize(4);
    cam_world_trans_[0].assign(4, 0);
    cam_world_trans_[1].assign(4, 0);
    cam_world_trans_[2].assign(4, 0);
    cam_world_trans_[3].assign(4, 0);

    //clear the camera to world coordinate transform values
    world_cam_trans_.clear();
    world_cam_trans_.resize(4);
    world_cam_trans_[0].assign(4, 0);
    world_cam_trans_[1].assign(4, 0);
    world_cam_trans_[2].assign(4, 0);
    world_cam_trans_[3].assign(4, 0);

    //clear the other values
    camera_id_.clear();
    image_height_ = 0;
    image_width_ = 0;
    pixel_depth_ = 0;
    camera_lens_.clear();
    camera_comments_.clear();
    camera_filled_ = false;
    rot_3x3_matrix_filled_ = false;
  }

  bool Camera::prep_camera() {
    //run the pre-run functions
    //need to add more error handling into the functions
    bool return_val = false;
    return_val = prep_lens_distortion_();
    return_val = prep_transforms_();
    return return_val;
  }

  bool Camera::prep_transforms_() {
    //if the rot_3x3_matrix is filled use those values
    //if the matrix is all zeros consider it unfilled
    bool nonzero_matrix = false;
    for (int_t i = 0; i < 3; i++) {
      for (int_t j = 0; j < 3; j++) {
        nonzero_matrix = nonzero_matrix || rotation_3x3_matrix_[i][j] != 0;
      }
    }
    //create the matrix from the alpha, beta, gamma values if not filled or all zeros
    if (!nonzero_matrix || !rot_3x3_matrix_filled_) {
      scalar_t cx, cy, cz, sx, sy, sz;
      cx = cos(extrinsics_[CAM::ALPHA] * DICE_PI / 180.0);
      cy = cos(extrinsics_[CAM::BETA] * DICE_PI / 180.0);
      cz = cos(extrinsics_[CAM::GAMMA] * DICE_PI / 180.0);
      sx = sin(extrinsics_[CAM::ALPHA] * DICE_PI / 180.0);
      sy = sin(extrinsics_[CAM::BETA] * DICE_PI / 180.0);
      sz = sin(extrinsics_[CAM::GAMMA] * DICE_PI / 180.0);

      rotation_3x3_matrix_[0][0] = cy * cz;
      rotation_3x3_matrix_[0][1] = sx * sy*cz - sz * cx;
      rotation_3x3_matrix_[0][2] = cx * sy*cz + sx * sz;
      rotation_3x3_matrix_[1][0] = cy * sz;
      rotation_3x3_matrix_[1][1] = sx * sy*sz + cx * cz;
      rotation_3x3_matrix_[1][2] = -sx * cz + cx * sy*sz;
      rotation_3x3_matrix_[2][0] = -sy;
      rotation_3x3_matrix_[2][1] = sx * cy;
      rotation_3x3_matrix_[2][2] = cx * cy;
    }

    //clear the world to camera coordinate transform values
    world_cam_trans_.clear();
    world_cam_trans_[0].assign(4, 0);
    world_cam_trans_[1].assign(4, 0);
    world_cam_trans_[2].assign(4, 0);
    world_cam_trans_[3].assign(4, 0);
    //create the 4x4 transformation matrix
    for (int_t i = 0; i < 3; i++) {
      for (int_t j = 0; j < 3; j++) {
        world_cam_trans_[i][j] = rotation_3x3_matrix_[i][j];
      }
    }
    world_cam_trans_[0][3] = extrinsics_[CAM::TX];
    world_cam_trans_[1][3] = extrinsics_[CAM::TY];
    world_cam_trans_[2][3] = extrinsics_[CAM::TZ];
    world_cam_trans_[3][3] = 1.0;

    //the cam to world transform will be the invers of the matrix
    cv::Mat temp_mat(4, 4, CV_64FC1);
    for (int_t i = 0; i < 4; i++) {
      for (int_t j = 0; j < 4; j++) {
        temp_mat.at<double>(i, j) = world_cam_trans_[i][j];
      }
    }
    //invert the matrix and save into cam_world_trans_
    cv::Mat inv_mat = temp_mat.inv();
    for (int_t i = 0; i < 4; i++) {
      for (int_t j = 0; j < 4; j++) {
        cam_world_trans_[i][j] = inv_mat.at<double>(i, j);
      }
    }
    return true;
  }


  bool Camera::prep_lens_distortion_() {
    //pre-run lens distortion function
    int_t image_size;
    scalar_t del_img_x;
    scalar_t del_img_y;
    scalar_t end_crit = 0.0001;
    std::stringstream msg_output;
    bool end_loop;

    //get the needed intrinsic values
    const scalar_t fx = intrinsics_[CAM::FX];
    const scalar_t fy = intrinsics_[CAM::FY];
    const scalar_t fs = intrinsics_[CAM::FS];
    const scalar_t cx = intrinsics_[CAM::CX];
    const scalar_t cy = intrinsics_[CAM::CY];

    //set the image size
    image_size = image_height_ * image_width_;

    //initialize the arrays
    inv_lens_dis_x_.assign(image_size, 0.0);
    inv_lens_dis_y_.assign(image_size, 0.0);
    std::vector<scalar_t> image_x(image_size, 0.0);
    std::vector<scalar_t> image_y(image_size, 0.0);
    std::vector<scalar_t> targ_x(image_size, 0.0);
    std::vector<scalar_t> targ_y(image_size, 0.0);
    std::vector<scalar_t> params(1, 0.0);

    for (int_t i = 0; i < image_size; i++) {
      //set the target value for x,y
      targ_x[i] = (scalar_t)(i % image_width_);
      targ_y[i] = (scalar_t)(i / image_width_);
      //generate the initial guss for the inverted sensor position
      inv_lens_dis_x_[i] = (targ_x[i] - cx) / fx;
      inv_lens_dis_y_[i] = (targ_y[i] - cy) / fy;
    }

    //iterate until the inverted point is near the target location
    for (int_t j = 0; j < 60; j++) {
      end_loop = true;
      DEBUG_MSG(" ");
      DEBUG_MSG("Invers distortion prep iteration  " << j);
      //do the projection
      sensor_to_image(inv_lens_dis_x_, inv_lens_dis_y_, image_x, image_y);
      //apply the correction
      for (int_t i = 0; i < image_size; i++) {
        del_img_x = targ_x[i] - image_x[i];
        del_img_y = targ_y[i] - image_y[i];
        if ((abs(del_img_x) > end_crit) || (abs(del_img_y) > end_crit)) end_loop = false;
        inv_lens_dis_x_[i] += (del_img_x) / fx;
        inv_lens_dis_y_[i] += (del_img_y) / fy;
      }
      if (end_loop) break;
    }
    return false;
  }


  DICE_LIB_DLL_EXPORT
    void Camera::image_to_sensor(
      std::vector<scalar_t> & image_x,
      std::vector<scalar_t> & image_y,
      std::vector<scalar_t> & sen_x,
      std::vector<scalar_t> & sen_y,
      bool integer_locs) {
    //transformation from distorted image locations to undistorted sensor locations
    int_t vect_size = 0;
    int_t index;
    int_t index00;
    int_t index10;
    int_t index01;
    int_t index11;
    int_t x_base, y_base;
    scalar_t dx, dy, x, y;
    vect_size = (int_t)(sen_x.size());
    //if we are at an integer pixel location it is a simple lookup of the pre-calculated values
    if (integer_locs) {
      for (int_t i = 0; i < vect_size; i++) {
        index = image_y[i] * image_width_ + image_x[i];
        sen_x[i] = inv_lens_dis_x_[index];
        sen_y[i] = inv_lens_dis_y_[index];
      }
    }
    else
    {
      //if not at an interger pixel location use linear interpolation to get the value
      for (int_t i = 0; i < vect_size; i++) {
        x = image_x[i];
        y = image_y[i];
        x_base = floor(x);
        y_base = floor(y);
        index00 = y_base * image_width_ + x_base;
        index10 = y_base * image_width_ + x_base + 1;
        index01 = (y_base + 1) * image_width_ + x_base;
        index11 = (y_base + 1) * image_width_ + x_base + 1;
        dx = x - x_base;
        dy = y - y_base;
        //quick linear interpolation to get the sensor location
        sen_x[i] = inv_lens_dis_x_[index00] * (1 - dx)*(1 - dy) + inv_lens_dis_x_[index10] * dx*(1 - dy) + inv_lens_dis_x_[index01] * (1 - dx)*dy + inv_lens_dis_x_[index11] * dx*dy;
        sen_y[i] = inv_lens_dis_y_[index00] * (1 - dx)*(1 - dy) + inv_lens_dis_y_[index10] * dx*(1 - dy) + inv_lens_dis_y_[index01] * (1 - dx)*dy + inv_lens_dis_y_[index11] * dx*dy;
      }
    }
  }


  DICE_LIB_DLL_EXPORT
    void Camera::sensor_to_image(
      std::vector<scalar_t> & sen_x,
      std::vector<scalar_t> & sen_y,
      std::vector<scalar_t> & image_x,
      std::vector<scalar_t> & image_y) {
    //converts sensor locations to image locations by applying lens distortions
    //scaling with fx and fy and shifting by cx, cy.
    const scalar_t fx = intrinsics_[CAM::FX];
    const scalar_t fy = intrinsics_[CAM::FY];
    const scalar_t fs = intrinsics_[CAM::FS];
    const scalar_t cx = intrinsics_[CAM::CX];
    const scalar_t cy = intrinsics_[CAM::CY];
    const scalar_t k1 = intrinsics_[CAM::K1];
    const scalar_t k2 = intrinsics_[CAM::K2];
    const scalar_t k3 = intrinsics_[CAM::K3];
    const scalar_t k4 = intrinsics_[CAM::K4];
    const scalar_t k5 = intrinsics_[CAM::K5];
    const scalar_t k6 = intrinsics_[CAM::K6];
    const scalar_t p1 = intrinsics_[CAM::P1];
    const scalar_t p2 = intrinsics_[CAM::P2];
    const scalar_t s1 = intrinsics_[CAM::S1];
    const scalar_t s2 = intrinsics_[CAM::S2];
    const scalar_t s3 = intrinsics_[CAM::S3];
    const scalar_t s4 = intrinsics_[CAM::S4];
    const scalar_t t1 = intrinsics_[CAM::T1];
    const scalar_t t2 = intrinsics_[CAM::T2];
    const int_t lens_dis_type = (int_t)intrinsics_[CAM::LD_MODEL];
    scalar_t x_sen, y_sen, rad, dis_coef, rad_sqr;
    scalar_t x_temp, y_temp;
    int_t vect_size = 0;

    vect_size = (int_t)(sen_x.size());

    //use the appropriate lens distortion model (only 8 parameter openCV has been tested)
    switch (lens_dis_type) {

    case CAM::NO_DIS_MODEL:
      for (int_t i = 0; i < vect_size; i++) {
        image_x[i] = sen_x[i] * fx + cx;
        image_y[i] = sen_y[i] * fy + cy;
      }
      break;

    case CAM::K1R1_K2R2_K3R3:
      for (int_t i = 0; i < vect_size; i++) {
        x_sen = sen_x[i];
        y_sen = sen_y[i];
        rad = sqrt(x_sen * x_sen + y_sen * y_sen);
        dis_coef = k1 * rad + k2 * pow(rad, 2) + k3 * pow(rad, 3);
        x_sen = (rad + dis_coef)*x_sen / rad;
        y_sen = (rad + dis_coef)*y_sen / rad;
        image_x[i] = x_sen * fx + cx;
        image_y[i] = y_sen * fy + cy;
      }
      break;

    case CAM::K1R2_K2R4_K3R6:
      for (int_t i = 0; i < vect_size; i++) {
        x_sen = sen_x[i];
        y_sen = sen_y[i];
        rad = sqrt(x_sen * x_sen + y_sen * y_sen);
        dis_coef = k1 * pow(rad, 2) + k2 * pow(rad, 4) + k3 * pow(rad, 6);
        x_sen = (rad + dis_coef)*x_sen / rad;
        y_sen = (rad + dis_coef)*y_sen / rad;
        image_x[i] = x_sen * fx + cx;
        image_y[i] = y_sen * fy + cy;
      }
      break;

    case CAM::K1R3_K2R5_K3R7:
      for (int_t i = 0; i < vect_size; i++) {
        x_sen = sen_x[i];
        y_sen = sen_y[i];
        rad = sqrt(x_sen * x_sen + y_sen * y_sen);
        dis_coef = k1 * pow(rad, 3) + k2 * pow(rad, 5) + k3 * pow(rad, 7);
        x_sen = (rad + dis_coef)*x_sen / rad;
        y_sen = (rad + dis_coef)*y_sen / rad;
        image_x[i] = x_sen * fx + cx;
        image_y[i] = y_sen * fy + cy;
      }
      break;

    case CAM::VIC3D_DIS:  //I believe it is K1R1_K2R2_K3R3 but need to confirm
      for (int_t i = 0; i < vect_size; i++) {
        x_sen = sen_x[i];
        y_sen = sen_y[i];
        rad_sqr = x_sen * x_sen + y_sen * y_sen;
        rad = sqrt((double)rad_sqr);
        dis_coef = k1 * rad + k2 * pow(rad, 2) + k3 * pow(rad, 3);
        x_sen = (rad + dis_coef)*x_sen / rad;
        y_sen = (rad + dis_coef)*y_sen / rad;
        image_x[i] = x_sen * fx + y_sen * fs + cx;
        image_y[i] = y_sen * fy + cy;
      }
      break;

    case CAM::OPENCV_DIS:
      //equations from: https://docs.opencv.org/3.4.1/d9/d0c/group__calib3d.html
    {
      const bool has_denom = (k4 != 0 || k5 != 0 || k6 != 0);
      const bool has_tangential = (p1 != 0 || p2 != 0);
      const bool has_prism = (s1 != 0 || s2 != 0 || s3 != 0 || s4 != 0);
      const bool has_Scheimpfug = (t1 != 0 || t2 != 0);

      for (int_t i = 0; i < vect_size; i++) {
        x_sen = sen_x[i];
        y_sen = sen_y[i];
        rad_sqr = x_sen * x_sen + y_sen * y_sen;
        dis_coef = 1 + k1 * rad_sqr + k2 * pow(rad_sqr, 2) + k3 * pow(rad_sqr, 3);
        image_x[i] = x_sen * dis_coef;
        image_y[i] = y_sen * dis_coef;
      }
      if (has_denom) {
        for (int_t i = 0; i < vect_size; i++) {
          x_sen = sen_x[i];
          y_sen = sen_y[i];
          rad_sqr = x_sen * x_sen + y_sen * y_sen;
          dis_coef = 1 / (1 + k4 * rad_sqr + k5 * pow(rad_sqr, 2) + k6 * pow(rad_sqr, 3));
          image_x[i] = image_x[i] * dis_coef;
          image_y[i] = image_y[i] * dis_coef;
        }
      }
      if (has_tangential) {
        for (int_t i = 0; i < vect_size; i++) {
          x_sen = sen_x[i];
          y_sen = sen_y[i];
          rad_sqr = x_sen * x_sen + y_sen * y_sen;
          dis_coef = 2 * p1*x_sen*y_sen + p2 * (rad_sqr + 2 * x_sen*x_sen);
          image_x[i] = image_x[i] + dis_coef;
          dis_coef = p1 * (rad_sqr + 2 * y_sen*y_sen) + 2 * p2*x_sen*y_sen;
          image_y[i] = image_y[i] + dis_coef;
        }
      }
      if (has_prism) {
        for (int_t i = 0; i < vect_size; i++) {
          x_sen = sen_x[i];
          y_sen = sen_y[i];
          rad_sqr = x_sen * x_sen + y_sen * y_sen;
          dis_coef = s1 * rad_sqr + s2 * rad_sqr*rad_sqr;
          image_x[i] = image_x[i] + dis_coef;
          dis_coef = s3 * rad_sqr + s4 * rad_sqr*rad_sqr;
          image_y[i] = image_y[i] + dis_coef;
        }
      }
      if (has_Scheimpfug) {
        scalar_t R11, R12, R13, R21, R22, R23, R31, R32, R33;
        scalar_t S11, S12, S13, S21, S22, S23, S31, S32, S33;
        scalar_t norm;

        //calculate the Scheimpfug factors
        //assuming t1, t2 in radians need to find out
        const scalar_t c_t1 = cos(t1);
        const scalar_t s_t1 = sin(t1);
        const scalar_t c_t2 = cos(t2);
        const scalar_t s_t2 = sin(t2);
        R11 = c_t2;
        R12 = s_t1 * s_t2;
        R13 = -s_t2 * c_t1;
        R22 = c_t1;
        R23 = s_t1;
        R31 = s_t2;
        R32 = -c_t2 * s_t1;
        R33 = c_t2 * c_t1;
        S11 = R11 * R33 - R13 * R31;
        S12 = R12 * R33 - R13 * R32;
        S13 = R33 * R13 - R13 * R33;
        S21 = -R23 * R31;
        S22 = R33 * R22 - R23 * R32;
        S23 = R33 * R23 - R23 * R33;
        S31 = R31;
        S32 = R32;
        S33 = R33;

        for (int_t i = 0; i < vect_size; i++) {
          x_temp = image_x[i];
          y_temp = image_y[i];
          norm = 1 / (S31 * x_temp + S32 * y_temp + S33);
          image_x[i] = (x_temp * S11 + y_temp * S12 + S13)*norm;
          image_y[i] = (x_temp * S21 + y_temp * S22 + S23)*norm;
        }
      }
      for (int_t i = 0; i < vect_size; i++) {
        image_x[i] = image_x[i] * fx + cx;
        image_y[i] = image_y[i] * fy + cy;
      }
    }
    break;
    default:
      //raise exception if it gets here?
      for (int_t i = 0; i < vect_size; i++) {
        image_x[i] = sen_x[i] * fx + cx;
        image_y[i] = sen_y[i] * fy + cy;
      }
    }
  }

  DICE_LIB_DLL_EXPORT
    void Camera::sensor_to_image(
      std::vector<scalar_t> & sen_x,
      std::vector<scalar_t> & sen_y,
      std::vector<scalar_t> & image_x,
      std::vector<scalar_t> & image_y,
      std::vector<std::vector<scalar_t>> & sen_dx,
      std::vector<std::vector<scalar_t>> & sen_dy,
      std::vector<std::vector<scalar_t>> & image_dx,
      std::vector<std::vector<scalar_t>> & image_dy) {
    //overload for derivitives
    //assume the lens distortion is mostly a translation of the subset
    //and does not effect the derivities. The scaling factors and skew will.
    sensor_to_image(sen_x, sen_y, image_x, image_y);

    const scalar_t fx = intrinsics_[CAM::FX];
    const scalar_t fy = intrinsics_[CAM::FY];
    const scalar_t fs = intrinsics_[CAM::FS];
    int_t vect_size = 0;

    vect_size = (int_t)(sen_x.size());

    for (int_t i = 0; i < vect_size; i++) {
      for (int_t j = 0; j < 3; i++) {
        image_dx[j][i] = sen_dx[j][i] * fx + sen_dy[j][i] * fs;
        image_dy[j][i] = sen_dy[j][i] * fy;
      }
    }

  }


  DICE_LIB_DLL_EXPORT
    void Camera::sensor_to_cam(
      std::vector<scalar_t> & sen_x,
      std::vector<scalar_t> & sen_y,
      std::vector<scalar_t> & cam_x,
      std::vector<scalar_t> & cam_y,
      std::vector<scalar_t> & cam_z,
      std::vector<scalar_t> & params)
  {
    //project the sensor locations onto a plane in space defined by zp, theta, phi 
    scalar_t zp = params[LSFunc::ZP];
    scalar_t theta = params[LSFunc::THETA];
    scalar_t phi = params[LSFunc::PHI];
    scalar_t cos_theta, cos_phi, cos_xi;
    scalar_t x_sen, y_sen, denom;
    int_t vect_size;
    cos_theta = cos(theta);
    cos_phi = cos(phi);
    cos_xi = sqrt(1 - cos_theta * cos_theta - cos_phi * cos_phi);

    vect_size = (int_t)(sen_x.size());

    for (int_t i = 0; i < vect_size; i++) {
      x_sen = sen_x[i];
      y_sen = sen_y[i];
      denom = 1 / (cos_xi + y_sen * cos_phi + x_sen * cos_theta);
      cam_x[i] = x_sen * zp * cos_xi * denom;
      cam_y[i] = y_sen * zp * cos_xi * denom;
      cam_z[i] = zp * cos_xi * denom;
    }
  }

  DICE_LIB_DLL_EXPORT
    void Camera::sensor_to_cam(
      std::vector<scalar_t> & sen_x,
      std::vector<scalar_t> & sen_y,
      std::vector<scalar_t> & cam_x,
      std::vector<scalar_t> & cam_y,
      std::vector<scalar_t> & cam_z,
      std::vector<scalar_t> & params,
      std::vector<std::vector<scalar_t>> & cam_dx,
      std::vector<std::vector<scalar_t>> & cam_dy,
      std::vector<std::vector<scalar_t>> & cam_dz)
  {
    //overloaded for first derivitives
    scalar_t zp = params[LSFunc::ZP];
    scalar_t theta = params[LSFunc::THETA];
    scalar_t phi = params[LSFunc::PHI];
    scalar_t cos_theta, cos_phi, cos_xi, sin_theta, sin_phi;
    scalar_t cos_xi_dzp, cos_xi_dtheta, cos_xi_dphi;
    scalar_t x_sen, y_sen, denom, denom2;
    scalar_t denom_dzp, denom_dtheta, denom_dphi;
    scalar_t uxcam, uycam, uzcam;
    scalar_t uxcam_dzp, uxcam_dtheta, uxcam_dphi;
    scalar_t uycam_dzp, uycam_dtheta, uycam_dphi;
    scalar_t uzcam_dzp, uzcam_dtheta, uzcam_dphi;
    int_t vect_size;


    cos_theta = cos(theta);
    cos_phi = cos(phi);
    sin_theta = sin(theta);
    sin_phi = sin(phi);
    cos_xi = sqrt(1 - cos_theta * cos_theta - cos_phi * cos_phi);
    cos_xi_dzp = 0;
    cos_xi_dtheta = cos_theta * sin_theta / cos_xi;
    cos_xi_dphi = cos_phi * sin_phi / cos_xi;

    vect_size = (int_t)(sen_x.size());

    for (int_t i = 0; i < vect_size; i++) {
      x_sen = sen_x[i];
      y_sen = sen_y[i];

      denom = (cos_xi + y_sen * cos_phi + x_sen * cos_theta);
      denom2 = denom * denom;

      uxcam = x_sen * zp * cos_xi;
      uycam = y_sen * zp * cos_xi;
      uzcam = zp * cos_xi;

      //factors for the derivitives
      denom_dzp = 0;
      denom_dtheta = cos_xi_dtheta - x_sen * sin_theta;
      denom_dphi = cos_xi_dphi - y_sen * sin_phi;

      uxcam_dzp = x_sen * cos_xi;
      uxcam_dtheta = x_sen * zp * cos_xi_dtheta;
      uxcam_dphi = x_sen * zp * cos_xi_dphi;

      uycam_dzp = y_sen * cos_xi;
      uycam_dtheta = y_sen * zp * cos_xi_dtheta;
      uycam_dphi = y_sen * zp * cos_xi_dphi;

      uzcam_dzp = cos_xi;
      uzcam_dtheta = zp * cos_xi_dtheta;
      uzcam_dphi = zp * cos_xi_dphi;

      //calculate the positions
      cam_x[i] = uxcam / denom;
      cam_y[i] = uycam / denom;
      cam_z[i] = uzcam / denom;

      //first derivities
      cam_dx[LSFunc::ZP][i] = (denom * uxcam_dzp - uxcam * denom_dzp) / denom2;
      cam_dx[LSFunc::THETA][i] = (denom * uxcam_dtheta - uxcam * denom_dtheta) / denom2;
      cam_dx[LSFunc::PHI][i] = (denom * uxcam_dphi - uxcam * denom_dphi) / denom2;

      cam_dy[LSFunc::ZP][i] = (denom * uycam_dzp - uycam * denom_dzp) / denom2;
      cam_dy[LSFunc::THETA][i] = (denom * uycam_dtheta - uycam * denom_dtheta) / denom2;
      cam_dy[LSFunc::PHI][i] = (denom * uycam_dphi - uycam * denom_dphi) / denom2;

      cam_dz[LSFunc::ZP][i] = (denom * uzcam_dzp - uzcam * denom_dzp) / denom2;
      cam_dz[LSFunc::THETA][i] = (denom * uzcam_dtheta - uzcam * denom_dtheta) / denom2;
      cam_dz[LSFunc::PHI][i] = (denom * uzcam_dphi - uzcam * denom_dphi) / denom2;

    }
  }

  DICE_LIB_DLL_EXPORT
    void Camera::cam_to_sensor(
      std::vector<scalar_t> & cam_x,
      std::vector<scalar_t> & cam_y,
      std::vector<scalar_t> & cam_z,
      std::vector<scalar_t> & sen_x,
      std::vector<scalar_t> & sen_y)
  {
    //project from camera x,y,z to sensor x,y
    int_t vect_size;
    vect_size = (int_t)(sen_x.size());

    for (int_t i = 0; i < vect_size; i++) {
      sen_x[i] = cam_x[i] / cam_z[i];
      sen_y[i] = cam_y[i] / cam_z[i];
    }
  }


  DICE_LIB_DLL_EXPORT
    void Camera::cam_to_sensor(
      std::vector<scalar_t> & cam_x,
      std::vector<scalar_t> & cam_y,
      std::vector<scalar_t> & cam_z,
      std::vector<scalar_t> & sen_x,
      std::vector<scalar_t> & sen_y,
      std::vector<std::vector<scalar_t>> & cam_dx,
      std::vector<std::vector<scalar_t>> & cam_dy,
      std::vector<std::vector<scalar_t>> & cam_dz,
      std::vector<std::vector<scalar_t>> & sen_dx,
      std::vector<std::vector<scalar_t>> & sen_dy)
  {
    //overloaded for first derivitives
    int_t vect_size;
    vect_size = (int_t)(sen_x.size());

    for (int_t i = 0; i < vect_size; i++) {
      sen_x[i] = cam_x[i] / cam_z[i];
      sen_y[i] = cam_y[i] / cam_z[i];

      for (int_t j = 0; j < 3; j++) {
        sen_dx[j][i] = (cam_dx[j][i] * cam_z[i] - cam_x[i] * cam_dz[j][i]) / (cam_z[j] * cam_z[j]);
        sen_dy[j][i] = (cam_dy[j][i] * cam_z[i] - cam_y[i] * cam_dz[j][i]) / (cam_z[j] * cam_z[j]);
      }
    }
  }


  DICE_LIB_DLL_EXPORT
    void Camera::rot_trans_transform_(
      std::vector<std::vector<scalar_t>> & RT_matrix,
      std::vector<scalar_t> & in_x,
      std::vector<scalar_t> & in_y,
      std::vector<scalar_t> & in_z,
      std::vector<scalar_t> & out_x,
      std::vector<scalar_t> & out_y,
      std::vector<scalar_t> & out_z)
  {
    //generic rotation/translation transformation
    int_t vect_size;
    vect_size = (int_t)(out_x.size());
    const scalar_t RT00 = RT_matrix[0][0];
    const scalar_t RT01 = RT_matrix[0][1];
    const scalar_t RT02 = RT_matrix[0][2];
    const scalar_t RT03 = RT_matrix[0][3];
    const scalar_t RT10 = RT_matrix[1][0];
    const scalar_t RT11 = RT_matrix[1][1];
    const scalar_t RT12 = RT_matrix[1][2];
    const scalar_t RT13 = RT_matrix[1][3];
    const scalar_t RT20 = RT_matrix[2][0];
    const scalar_t RT21 = RT_matrix[2][1];
    const scalar_t RT22 = RT_matrix[2][2];
    const scalar_t RT23 = RT_matrix[2][3];


    for (int_t i = 0; i < vect_size; i++) {
      out_x[i] = RT00 * in_x[i] + RT01 * in_y[i] + RT02 * in_z[i] + RT03;
      out_y[i] = RT10 * in_x[i] + RT11 * in_y[i] + RT12 * in_z[i] + RT13;
      out_z[i] = RT20 * in_x[i] + RT21 * in_y[i] + RT22 * in_z[i] + RT23;
    }
  }

  DICE_LIB_DLL_EXPORT
    void Camera::rot_trans_transform_(
      std::vector<std::vector<scalar_t>> & RT_matrix,
      std::vector<scalar_t> & in_x,
      std::vector<scalar_t> & in_y,
      std::vector<scalar_t> & in_z,
      std::vector<scalar_t> & out_x,
      std::vector<scalar_t> & out_y,
      std::vector<scalar_t> & out_z,
      std::vector<std::vector<scalar_t>> & in_dx,
      std::vector<std::vector<scalar_t>> & in_dy,
      std::vector<std::vector<scalar_t>> & in_dz,
      std::vector<std::vector<scalar_t>> & out_dx,
      std::vector<std::vector<scalar_t>> & out_dy,
      std::vector<std::vector<scalar_t>> & out_dz)
  {
    //overloaded for first derivitives

    int_t vect_size;
    vect_size = (int_t)(out_x.size());
    const scalar_t RT00 = RT_matrix[0][0];
    const scalar_t RT01 = RT_matrix[0][1];
    const scalar_t RT02 = RT_matrix[0][2];
    const scalar_t RT03 = RT_matrix[0][3];
    const scalar_t RT10 = RT_matrix[1][0];
    const scalar_t RT11 = RT_matrix[1][1];
    const scalar_t RT12 = RT_matrix[1][2];
    const scalar_t RT13 = RT_matrix[1][3];
    const scalar_t RT20 = RT_matrix[2][0];
    const scalar_t RT21 = RT_matrix[2][1];
    const scalar_t RT22 = RT_matrix[2][2];
    const scalar_t RT23 = RT_matrix[2][3];


    for (int_t i = 0; i < vect_size; i++) {
      out_x[i] = RT00 * in_x[i] + RT01 * in_y[i] + RT02 * in_z[i] + RT03;
      out_y[i] = RT10 * in_x[i] + RT11 * in_y[i] + RT12 * in_z[i] + RT13;
      out_z[i] = RT20 * in_x[i] + RT21 * in_y[i] + RT22 * in_z[i] + RT23;

      out_dx[0][i] = RT00 * in_dx[0][i] + RT01 * in_dy[0][i] + RT02 * in_dz[0][i];
      out_dx[1][i] = RT00 * in_dx[1][i] + RT01 * in_dy[1][i] + RT02 * in_dz[1][i];
      out_dx[2][i] = RT00 * in_dx[2][i] + RT01 * in_dy[2][i] + RT02 * in_dz[2][i];

      out_dy[0][i] = RT10 * in_dx[0][i] + RT11 * in_dy[0][i] + RT12 * in_dz[0][i];
      out_dy[1][i] = RT10 * in_dx[1][i] + RT11 * in_dy[1][i] + RT12 * in_dz[1][i];
      out_dy[2][i] = RT10 * in_dx[2][i] + RT11 * in_dy[2][i] + RT12 * in_dz[2][i];

      out_dz[0][i] = RT20 * in_dx[0][i] + RT21 * in_dy[0][i] + RT22 * in_dz[0][i];
      out_dz[1][i] = RT20 * in_dx[1][i] + RT21 * in_dy[1][i] + RT22 * in_dz[1][i];
      out_dz[2][i] = RT20 * in_dx[2][i] + RT21 * in_dy[2][i] + RT22 * in_dz[2][i];
    }
  }




  DICE_LIB_DLL_EXPORT
    bool Camera::check_valid_(std::string & msg) {
    //quick check to see if the minimal required information is present
    bool is_valid = true;
    std::stringstream message;
    message = std::stringstream();
    if (intrinsics_[CAM::FX] <= 0) {
      message << "fx must be greater than 0" << "\n";
      is_valid = false;
    }
    if (intrinsics_[CAM::FY] <= 0) {
      message << "fy must be greater than 0" << "\n";
      is_valid = false;
    }
    if (image_height_ <= 0) {
      message << "image height must be greater than 0" << "\n";
      is_valid = false;
    }
    if (image_width_ <= 0) {
      message << "image width must be greater than 0" << "\n";
      is_valid = false;
    }
    msg = message.str();
    return is_valid;
  }
}// end DICe namespace
