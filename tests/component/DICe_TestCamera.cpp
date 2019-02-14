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

#include <DICe_LocalShapeFunction.h>
#include <DICe_Camera.h>
#include <cstdlib>
#include <ctime>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <Teuchos_oblackholestream.hpp>
#include <iostream>
#include <fstream>

using namespace DICe;

int main(int argc, char *argv[]) {

  bool all_passed = true;
  DICe::initialize(argc, argv);
  // only print output if args are given (for testing the output is quiet)
  int_t iprint = argc - 1;
  scalar_t strong_match = 1.0e-4;
  scalar_t soft_match = 0.005;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Starting test of the camera class ---" << std::endl;

  bool passed = true;
  DICe::Camera::Camera_Info camera_info;

  //projection parameter arrays
  std::vector<scalar_t> proj_params(3, 0.0);
  std::vector<scalar_t> proj_params2(3, 0.0);

  //image, cam, sensor, world and return arrays
  std::vector<scalar_t> img_x(1, 0.0);
  std::vector<scalar_t> img_y(1, 0.0);
  std::vector<scalar_t> sen_x(1, 0.0);
  std::vector<scalar_t> sen_y(1, 0.0);
  std::vector<scalar_t> cam_x(1, 0.0);
  std::vector<scalar_t> cam_y(1, 0.0);
  std::vector<scalar_t> cam_z(1, 0.0);
  std::vector<scalar_t> wld_x(1, 0.0);
  std::vector<scalar_t> wld_y(1, 0.0);
  std::vector<scalar_t> wld_z(1, 0.0);
  std::vector<scalar_t> ret_x(1, 0.0);
  std::vector<scalar_t> ret_y(1, 0.0);
  std::vector<scalar_t> ret_z(1, 0.0);
  std::vector<scalar_t> ret1_x(1, 0.0);
  std::vector<scalar_t> ret1_y(1, 0.0);
  std::vector<scalar_t> ret1_z(1, 0.0);
  std::vector<scalar_t> ret2_x(1, 0.0);
  std::vector<scalar_t> ret2_y(1, 0.0);
  std::vector<scalar_t> ret2_z(1, 0.0);
  //first partials
  std::vector<std::vector<scalar_t> > d1_dx;
  std::vector<std::vector<scalar_t> > d1_dy;
  std::vector<std::vector<scalar_t> > d1_dz;
  std::vector<std::vector<scalar_t> > d2_dx;
  std::vector<std::vector<scalar_t> > d2_dy;
  std::vector<std::vector<scalar_t> > d2_dz;
  std::vector<std::vector<scalar_t> > dn_dx;
  std::vector<std::vector<scalar_t> > dn_dy;
  std::vector<std::vector<scalar_t> > dn_dz;
  std::vector<std::vector<scalar_t> > der_dels;
  std::vector<std::vector<scalar_t> > der_aves;
  // dummy values used for testing set and get methods:
  std::string test_str = "big";
  //int_t test_val = 1024;
  int_t image_height=-1;
  int_t image_width=-1;
  scalar_t params_delta = 0.001;

  //*********************Test the basic get/set functions************************************************

  //fill one pair of arrays with random numbers
  for (size_t i = 0; i < camera_info.intrinsics_.size(); i++) camera_info.intrinsics_[i] = (rand() % 10000) / 1000.0;
  camera_info.tx_ = (rand() % 10000) / 1000.0;
  camera_info.ty_ = (rand() % 10000) / 1000.0;
  camera_info.tz_ = (rand() % 10000) / 1000.0;
  Matrix<scalar_t,3> gold_rotation;
  for(size_t i=0;i<gold_rotation.rows();++i){
    for(size_t j=0;j<gold_rotation.cols();++j){
      gold_rotation(i,j) = (rand() % 10000) / 1000.0;
    }
  }
  camera_info.rotation_matrix_ = gold_rotation;
  camera_info.id_ = "camera_1";
  camera_info.image_height_ = 10;
  camera_info.image_width_ = 10;
  camera_info.pixel_depth_ = 10;

  DICe::Camera test_cam(camera_info);

  //Test setting getting and clearing the camera parameters
  passed = true;
  for (size_t i = 0; i < test_cam.intrinsics()->size(); i++) {
    if ((*test_cam.intrinsics())[i] != camera_info.intrinsics_[i]){
      *outStream << "error: intrinsics are not correct" << std::endl;
      passed = false;
    }
  }
  if((*test_cam.rotation_matrix())!=gold_rotation){
    *outStream << "error: extrinsic rotation matrix is not correct" << std::endl;
    *outStream << "extrinsic rotation matrix" << std::endl;
    *outStream << test_cam.rotation_matrix() << std::endl;
    *outStream << "gold rotation matrix" << std::endl;
    *outStream << gold_rotation << std::endl;
    passed = false;
  }
  if (test_cam.tx() != camera_info.tx_){
    *outStream << "error: extrinsic translation x is not correct" << std::endl;
    passed = false;
  }
  if (test_cam.ty() != camera_info.ty_){
    *outStream << "error: extrinsic translation y is not correct" << std::endl;
    passed = false;
  }
  if (test_cam.tz() != camera_info.tz_){
    *outStream << "error: extrinsic translation z is not correct" << std::endl;
    passed = false;
  }
  test_cam.set_comments(test_str);
  if (test_cam.comments() != test_str){
    *outStream << "error: camera comments are not correct" << std::endl;
    passed = false;
  }
  test_cam.set_lens(test_str);
  if (test_cam.lens() != test_str){
    *outStream << "camera lens is not correct" << std::endl;
    passed = false;
  }
  if(test_cam.id()!="camera_1"){
    *outStream << "error: identifier not correct" << std::endl;
    passed = false;
  }
  test_cam.set_id(test_str);
  if (test_cam.id() != test_str){
    *outStream << "error: set identifier method not working" << std::endl;
    passed = false;
  }
  if(test_cam.image_height()!=10){
    *outStream << "error: image height not correct" << std::endl;
    passed = false;
  }
  if(test_cam.image_width()!=10){
    *outStream << "error: image width not correct" << std::endl;
    passed = false;
  }
  if(test_cam.pixel_depth()!=10){
    *outStream << "error: pixel depth not correct" << std::endl;
    passed = false;
  }

  all_passed = all_passed && passed;

  //*********************Test rotation matrix creation************************************************

  passed = true;
  Matrix<scalar_t,3> rotate_3x3;
  for (int_t dataset = 0; dataset < 2; dataset++) {
    if (dataset == 0) {
      //values from the checkerboard calibration test
      //Fill parameters with reasonable values for testing
      //From checkerboard calibration in TestStereoCalib
      camera_info.intrinsics_[Camera::CX] = 3.138500331437E+02;
      camera_info.intrinsics_[Camera::CY] = 2.423001711052E+02;
      camera_info.intrinsics_[Camera::FX] = 5.341694772442E+02;
      camera_info.intrinsics_[Camera::FY] = 5.288189029375E+02;
      camera_info.intrinsics_[Camera::FS] = 0.0;
      camera_info.intrinsics_[Camera::K1] = -2.632408663590E-01;
      camera_info.intrinsics_[Camera::K2] = -1.196920656310E-02;
      camera_info.intrinsics_[Camera::K3] = 0;
      camera_info.intrinsics_[Camera::K4] = 0;
      camera_info.intrinsics_[Camera::K5] = 0;
      camera_info.intrinsics_[Camera::K6] = -1.959487138239E-01;
      camera_info.intrinsics_[Camera::S1] = 0;
      camera_info.intrinsics_[Camera::S2] = 0;
      camera_info.intrinsics_[Camera::S3] = 0;
      camera_info.intrinsics_[Camera::S4] = 0;
      camera_info.intrinsics_[Camera::P1] = 0;
      camera_info.intrinsics_[Camera::P2] = 0;
      camera_info.intrinsics_[Camera::T1] = 0;
      camera_info.intrinsics_[Camera::T2] = 0;
      camera_info.lens_distortion_model_ = Camera::OPENCV_LENS_DISTORTION;

      camera_info.tx_ = -3.342839793459E+00;
      camera_info.ty_ = 4.670078367815E-02;
      camera_info.tz_ = 3.252131958135E-03;

      rotate_3x3(0,0) = 9.999741331604E-01;
      rotate_3x3(0,1) = 4.700768261799E-03;
      rotate_3x3(0,2) = 5.443876178856E-03;
      rotate_3x3(1,0) = -4.764218549160E-03;
      rotate_3x3(1,1) = 9.999201838000E-01;
      rotate_3x3(1,2) = 1.170163454318E-02;
      rotate_3x3(2,0) = -5.388434997075E-03;
      rotate_3x3(2,1) = -1.172726767475E-02;
      rotate_3x3(2,2) = 9.999167145123E-01;
      camera_info.rotation_matrix_ = rotate_3x3;

      camera_info.image_height_ = 480;
      camera_info.image_width_ = 640;

    }
    else {
      //values from the dot calibration test
      //Fill parameters with reasonable values for testing
      //From checkerboard calibration in TestStereoCalib
      camera_info.intrinsics_[Camera::CX] = 1.224066475835E+03;
      camera_info.intrinsics_[Camera::CY] = 1.024007702260E+03;
      camera_info.intrinsics_[Camera::FX] = 1.275326315913E+04;
      camera_info.intrinsics_[Camera::FY] = 1.271582348356E+04;
      camera_info.intrinsics_[Camera::FS] = 0.0;
      camera_info.intrinsics_[Camera::K1] = -7.006520231053E-02;
      camera_info.intrinsics_[Camera::K2] = 8.573035432280E+01;
      camera_info.intrinsics_[Camera::K3] = 0;
      camera_info.intrinsics_[Camera::K4] = 0;
      camera_info.intrinsics_[Camera::K5] = 0;
      camera_info.intrinsics_[Camera::K6] = -5.423416904733E-01;
      camera_info.intrinsics_[Camera::S1] = 0;
      camera_info.intrinsics_[Camera::S2] = 0;
      camera_info.intrinsics_[Camera::S3] = 0;
      camera_info.intrinsics_[Camera::S4] = 0;
      camera_info.intrinsics_[Camera::P1] = 0;
      camera_info.intrinsics_[Camera::P2] = 0;
      camera_info.intrinsics_[Camera::T1] = 0;
      camera_info.intrinsics_[Camera::T2] = 0;
      camera_info.lens_distortion_model_ = Camera::OPENCV_LENS_DISTORTION;

      camera_info.tx_ = -8.823158862228E+01;
      camera_info.ty_ = 5.771721469879E-01;
      camera_info.tz_ = 2.396269011734E+01;

      rotate_3x3(0,0) = 8.838522041011E-01;
      rotate_3x3(0,1) = 1.380068199293E-02;
      rotate_3x3(0,2) = 4.675626401693E-01;
      rotate_3x3(1,0) = -1.330392252901E-02;
      rotate_3x3(1,1) = 9.999019740475E-01;
      rotate_3x3(1,2) = -4.364394718824E-03;
      rotate_3x3(2,0) = -4.675770385198E-01;
      rotate_3x3(2,1) = -2.362937250472E-03;
      rotate_3x3(2,2) = 8.839491668510E-01;
      camera_info.rotation_matrix_ = rotate_3x3;

      camera_info.image_height_ = 2000;
      camera_info.image_width_ = 2400;
    }

    //creat a compatible Mat for the decomposition of the projection matrix
    cv::Mat opencv_projection = cv::Mat(3, 4, cv::DataType<double>::type);

    //create the projection matrix
    opencv_projection.at<double>(0, 0) = camera_info.intrinsics_[Camera::FX] * rotate_3x3(0,0) + camera_info.intrinsics_[Camera::CX] * rotate_3x3(2,0);
    opencv_projection.at<double>(1, 0) = camera_info.intrinsics_[Camera::FY] * rotate_3x3(1,0) + camera_info.intrinsics_[Camera::CY] * rotate_3x3(2,0);
    opencv_projection.at<double>(2, 0) = rotate_3x3(2,0);

    opencv_projection.at<double>(0, 1) = camera_info.intrinsics_[Camera::FX] * rotate_3x3(0,1) + camera_info.intrinsics_[Camera::CX] * rotate_3x3(2,1);
    opencv_projection.at<double>(1, 1) = camera_info.intrinsics_[Camera::FY] * rotate_3x3(1,1) + camera_info.intrinsics_[Camera::CY] * rotate_3x3(2,1);
    opencv_projection.at<double>(2, 1) = rotate_3x3(2,1);

    opencv_projection.at<double>(0, 2) = camera_info.intrinsics_[Camera::FX] * rotate_3x3(0,2) + camera_info.intrinsics_[Camera::CX] * rotate_3x3(2,2);
    opencv_projection.at<double>(1, 2) = camera_info.intrinsics_[Camera::FY] * rotate_3x3(1,2) + camera_info.intrinsics_[Camera::CY] * rotate_3x3(2,2);
    opencv_projection.at<double>(2, 2) = rotate_3x3(2,2);

    opencv_projection.at<double>(0, 3) = camera_info.intrinsics_[Camera::FX] * camera_info.tx_
        + camera_info.intrinsics_[Camera::CX] * camera_info.tz_;
    opencv_projection.at<double>(1, 3) = camera_info.intrinsics_[Camera::FY] * camera_info.ty_
        + camera_info.intrinsics_[Camera::CY] * camera_info.tz_;
    opencv_projection.at<double>(2, 3) = camera_info.tz_;

    //create the return matrices
    cv::Mat cam, rot, trans, rotx, roty, rotz, euler;
    //call the decomposer
    cv::decomposeProjectionMatrix(opencv_projection, cam, rot, trans, rotx, roty, rotz, euler);

    DEBUG_MSG("Euler: " << euler);
    const scalar_t alpha = euler.at<double>(0);
    const scalar_t beta = euler.at<double>(1);
    const scalar_t gamma = euler.at<double>(2);

    //fill the euler angles
    camera_info.set_rotation_matrix(alpha,beta,gamma);

    // check that the rotation matrix as computed from the euler angles is the same as the inital rotation matrix
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        if (std::abs(camera_info.rotation_matrix_(i,j) - rotate_3x3(i, j)) > strong_match){
          passed = false;
          *outStream << "error: camera info rotation matrix not correct from Eulers" << std::endl;
        }
      }
    }
    DICe::Camera camera(camera_info);
    // test that the camera rotation matrix as initialized is the same as the opencv rotation matrix
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        if (std::abs((*camera.rotation_matrix())(i,j) - rot.at<double>(i, j)) > strong_match){
          passed = false;
          *outStream << "error: camera rotation matrix not correct from Eulers" << std::endl;
        }
      }
    }
  }
  if (passed) *outStream << "passed: create rotation matrix from Eulers" << std::endl;
  else *outStream << "failed: create rotation matrix from Eulers" << std::endl;

  //*********************Test the lens distortion functions************************************************
  std::string setup[4];
  setup[0] = "Checkerboard 8 parm lens dist";
  setup[1] = "Checkerboard 3 parm lens dist";
  setup[2] = "Dots 8 parm lens dist";
  setup[3] = "Dots 3 parm lens dist";

  std::vector<scalar_t> intrinsic1(Camera::MAX_CAM_INTRINSIC_PARAM,0);
  all_passed = all_passed && passed;
  for (int_t m = 0; m < 4; m++) {
    passed = true;
    if (m == 0) {
      // Checkerboard 8 parm lens dist
      intrinsic1[Camera::CX] = 3.138500331437E+02;
      intrinsic1[Camera::CY] = 2.423001711052E+02;
      intrinsic1[Camera::FX] = 5.341694772442E+02;
      intrinsic1[Camera::FY] = 5.288189029375E+02;
      intrinsic1[Camera::FS] = 0.0;
      intrinsic1[Camera::K1] = -23.96269259;
      intrinsic1[Camera::K2] = 144.3689661;
      intrinsic1[Camera::K3] = -9.005811529;
      intrinsic1[Camera::K4] = -23.68880258;
      intrinsic1[Camera::K5] = 137.8148457;
      intrinsic1[Camera::K6] = 30.21994582;
      intrinsic1[Camera::S1] = 0;
      intrinsic1[Camera::S2] = 0;
      intrinsic1[Camera::S3] = 0;
      intrinsic1[Camera::S4] = 0;
      intrinsic1[Camera::P1] = 0.001778201;
      intrinsic1[Camera::P2] = -0.000292407;
      intrinsic1[Camera::T1] = 0;
      intrinsic1[Camera::T2] = 0;

      image_height = 480;
      image_width = 640;
    }
    else if (m == 1) {
      // Checkerboard 3 parm lens dist
      intrinsic1[Camera::CX] = 3.138500331437E+02;
      intrinsic1[Camera::CY] = 2.423001711052E+02;
      intrinsic1[Camera::FX] = 5.341694772442E+02;
      intrinsic1[Camera::FY] = 5.288189029375E+02;
      intrinsic1[Camera::FS] = 0.0;
      intrinsic1[Camera::K1] = -0.268055154;
      intrinsic1[Camera::K2] = -0.020449625;
      intrinsic1[Camera::K3] = 0.201676357;
      intrinsic1[Camera::K4] = 0;
      intrinsic1[Camera::K5] = 0;
      intrinsic1[Camera::K6] = 0;
      intrinsic1[Camera::S1] = 0;
      intrinsic1[Camera::S2] = 0;
      intrinsic1[Camera::S3] = 0;
      intrinsic1[Camera::S4] = 0;
      intrinsic1[Camera::P1] = 0;
      intrinsic1[Camera::P2] = 0;
      intrinsic1[Camera::T1] = 0;
      intrinsic1[Camera::T2] = 0;

      image_height = 480;
      image_width = 640;
    }
    else if (m == 2) {
      //dot pattern 8 parm lens dist
      intrinsic1[Camera::CX] = 1.224066475835E+03;
      intrinsic1[Camera::CY] = 1.024007702260E+03;
      intrinsic1[Camera::FX] = 1.275326315913E+04;
      intrinsic1[Camera::FY] = 1.271582348356E+04;
      intrinsic1[Camera::FS] = 0.0;
      intrinsic1[Camera::K1] = 11.02145155;
      intrinsic1[Camera::K2] = 41.23525914;
      intrinsic1[Camera::K3] = 0.340681332;
      intrinsic1[Camera::K4] = 11.29201846;
      intrinsic1[Camera::K5] = -41.25264705;
      intrinsic1[Camera::K6] = -0.340766754;
      intrinsic1[Camera::S1] = 0;
      intrinsic1[Camera::S2] = 0;
      intrinsic1[Camera::S3] = 0;
      intrinsic1[Camera::S4] = 0;
      intrinsic1[Camera::P1] = 0.004164821;
      intrinsic1[Camera::P2] = -0.00707348;
      intrinsic1[Camera::T1] = 0;
      intrinsic1[Camera::T2] = 0;

      image_height = 2000;
      image_width = 2400;
    }
    else {
      //dot pattern 3 parm lens dist
      intrinsic1[Camera::CX] = 1.224066475835E+03;
      intrinsic1[Camera::CY] = 1.024007702260E+03;
      intrinsic1[Camera::FX] = 1.275326315913E+04;
      intrinsic1[Camera::FY] = 1.271582348356E+04;
      intrinsic1[Camera::FS] = 0.0;
      intrinsic1[Camera::K1] = -0.330765074;
      intrinsic1[Camera::K2] = 78.54478098;
      intrinsic1[Camera::K3] = 0.512417786;
      intrinsic1[Camera::K4] = 0;
      intrinsic1[Camera::K5] = 0;
      intrinsic1[Camera::K6] = 0;
      intrinsic1[Camera::S1] = 0;
      intrinsic1[Camera::S2] = 0;
      intrinsic1[Camera::S3] = 0;
      intrinsic1[Camera::S4] = 0;
      intrinsic1[Camera::P1] = 0;
      intrinsic1[Camera::P2] = 0;
      intrinsic1[Camera::T1] = 0;
      intrinsic1[Camera::T2] = 0;

      image_height = 2000;
      image_width = 2400;
    }

    camera_info.clear();
    camera_info.lens_distortion_model_ = Camera::OPENCV_LENS_DISTORTION;
    camera_info.image_height_ = image_height;
    camera_info.image_width_ = image_width;
    camera_info.set_intrinsics(intrinsic1);

    test_cam = DICe::Camera(camera_info);

    //assume the max distortion would be at point (0,0)
    img_x.assign(1, 0.0);
    img_y.assign(1, 0.0);
    sen_x.assign(1, 0.0);
    sen_y.assign(1, 0.0);
    ret_x.assign(1, 0.0);
    ret_y.assign(1, 0.0);
    //transform the points from the image to the sensor (inverse distortion)
    test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y);
    //convert that value back to image coordinates without applying distortoin
    img_x[0] = sen_x[0] * intrinsic1[Camera::FX] + intrinsic1[Camera::CX];
    img_y[0] = sen_y[0] * intrinsic1[Camera::FY] + intrinsic1[Camera::CY];
    DEBUG_MSG("max distortion movement: " << setup[m] << ":  " << img_x[0] << " " << img_y[0]);

    //testing the distortion/inverted distortion routines
    scalar_t max_deviation = -1;
    //seed the random number generator
    srand((unsigned)time(0));
    //create arrays of 10000 random x,y image points
    int_t num_points = 10000;
    img_x.assign(num_points, 0.0);
    img_y.assign(num_points, 0.0);
    sen_x.assign(num_points, 0.0);
    sen_y.assign(num_points, 0.0);
    cam_x.assign(num_points, 0.0);
    cam_y.assign(num_points, 0.0);
    cam_z.assign(num_points, 0.0);
    wld_x.assign(num_points, 0.0);
    wld_y.assign(num_points, 0.0);
    wld_z.assign(num_points, 0.0);
    ret_x.assign(num_points, 0.0);
    ret_y.assign(num_points, 0.0);
    ret_z.assign(num_points, 0.0);
    for (int_t i = 0; i < num_points; i++) {
      img_x[i] = rand() % image_width;
      img_y[i] = rand() % image_height;
    }
    //transform the points from the image to the sensor (inverse distortion)
    test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y);
    //make sure the projections are doing something
    bool all_match = true;
    for (int_t i = 0; i < num_points; i++) {
      if (0.01 < abs(img_x[i] - sen_x[i])) all_match = false;
    }
    if (all_match) {
      *outStream << "failed: no change in image coordinates" << setup[m] << ":  " << std::endl;
      all_passed = false;
    }

    //convert the sensor locations to image locations
    test_cam.sensor_to_image(sen_x, sen_y, ret_x, ret_y);
    for (int_t i = 0; i < num_points; i++) {
      if (max_deviation < abs(ret_x[i] - img_x[i])) max_deviation = abs(ret_x[i] - img_x[i]);
      if (max_deviation < abs(ret_y[i] - img_y[i])) max_deviation = abs(ret_y[i] - img_y[i]);
    }
    if (max_deviation > soft_match) {
      *outStream << "failed: max integer distortion deviation(10000 rand points) : " << setup[m] << ":  " << max_deviation << std::endl;
      all_passed = false;
    }
    else *outStream << "passed: max integer distortion deviation(10000 rand points) : " << setup[m] << ":  " << max_deviation << std::endl;

    //do the same with non-integer locations
    for (int_t i = 0; i < num_points; i++) {
      img_x[i] = rand() % (image_width);
      img_x[i] = img_x[i] / 100.0;
      img_y[i] = rand() % (image_height * 100);
      img_y[i] = img_y[i] / 100.0;
    }
    test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y, false);
    test_cam.sensor_to_image(sen_x, sen_y, ret_x, ret_y);
    for (int_t i = 0; i < num_points; i++) {
      if (max_deviation < abs(ret_x[i] - img_x[i])) max_deviation = abs(ret_x[i] - img_x[i]);
      if (max_deviation < abs(ret_y[i] - img_y[i])) max_deviation = abs(ret_y[i] - img_y[i]);
    }
    if (max_deviation > soft_match) {
      *outStream << "failed: max non-integer distortion deviation(10000 rand points) : " << setup[m] << ":  " << max_deviation << std::endl;
      all_passed = false;
    }
    else *outStream << "passed: max non-integer distortion deviation(10000 rand points) : " << setup[m] << ":  " << max_deviation << std::endl;
  } // end 4 case loop

  //*****************************test the other transformation functions****************
  //make sure we are playing with the dot pattern parameters
  //dot pattern 3 parm lens dist
  intrinsic1[Camera::CX] = 1.224066475835E+03;
  intrinsic1[Camera::CY] = 1.024007702260E+03;
  intrinsic1[Camera::FX] = 1.275326315913E+04;
  intrinsic1[Camera::FY] = 1.271582348356E+04;
  intrinsic1[Camera::FS] = 0.0;
  intrinsic1[Camera::K1] = -0.330765074;
  intrinsic1[Camera::K2] = 78.54478098;
  intrinsic1[Camera::K3] = 0.512417786;
  intrinsic1[Camera::K4] = 0;
  intrinsic1[Camera::K5] = 0;
  intrinsic1[Camera::K6] = 0;
  intrinsic1[Camera::S1] = 0;
  intrinsic1[Camera::S2] = 0;
  intrinsic1[Camera::S3] = 0;
  intrinsic1[Camera::S4] = 0;
  intrinsic1[Camera::P1] = 0;
  intrinsic1[Camera::P2] = 0;
  intrinsic1[Camera::T1] = 0;
  intrinsic1[Camera::T2] = 0;

  scalar_t tx = -8.823158862228E+01;
  scalar_t ty = 5.771721469879E-01;
  scalar_t tz = 2.396269011734E+01;

  rotate_3x3(0,0) = 8.838522041011E-01;
  rotate_3x3(0,1) = 1.380068199293E-02;
  rotate_3x3(0,2) = 4.675626401693E-01;
  rotate_3x3(1,0) = -1.330392252901E-02;
  rotate_3x3(1,1) = 9.999019740475E-01;
  rotate_3x3(1,2) = -4.364394718824E-03;
  rotate_3x3(2,0) = -4.675770385198E-01;
  rotate_3x3(2,1) = -2.362937250472E-03;
  rotate_3x3(2,2) = 8.839491668510E-01;

  image_height = 2000;
  image_width = 2400;

  camera_info.clear();
  camera_info.lens_distortion_model_ = Camera::OPENCV_LENS_DISTORTION;
  camera_info.image_height_ = image_height;
  camera_info.image_width_ = image_width;
  camera_info.set_intrinsics(intrinsic1);
  camera_info.rotation_matrix_ = rotate_3x3;
  camera_info.tx_ = tx;
  camera_info.set_extrinsic_translations(tx,ty,tz);

  test_cam = Camera(camera_info);

  //use reasonalable projection parameters for the setup
  proj_params[Projection_Shape_Function::ZP] = 188;
  proj_params[Projection_Shape_Function::THETA] = 1.32;
  proj_params[Projection_Shape_Function::PHI] = 1.5;

  proj_params2[Projection_Shape_Function::ZP] = proj_params[Projection_Shape_Function::ZP];
  proj_params2[Projection_Shape_Function::THETA] = proj_params[Projection_Shape_Function::THETA];
  proj_params2[Projection_Shape_Function::PHI] = proj_params[Projection_Shape_Function::PHI];

  //testing the distortion/inverted distortion routines
  scalar_t max_deviation = -1;

  //test projection - transformation functions
  srand((unsigned)time(0));
  //create arrays of 10000 random x,y image points
  int_t num_points = 10;

  //initialize the arrays of position values
  img_x.assign(num_points, 0.0);
  img_y.assign(num_points, 0.0);
  sen_x.assign(num_points, 0.0);
  sen_y.assign(num_points, 0.0);
  cam_x.assign(num_points, 0.0);
  cam_y.assign(num_points, 0.0);
  cam_z.assign(num_points, 0.0);
  wld_x.assign(num_points, 0.0);
  wld_y.assign(num_points, 0.0);
  wld_z.assign(num_points, 0.0);
  ret_x.assign(num_points, 0.0);
  ret_y.assign(num_points, 0.0);
  ret_z.assign(num_points, 0.0);
  ret1_x.assign(num_points, 0.0);
  ret1_y.assign(num_points, 0.0);
  ret1_z.assign(num_points, 0.0);
  ret2_x.assign(num_points, 0.0);
  ret2_y.assign(num_points, 0.0);
  ret2_z.assign(num_points, 0.0);
  //fill the image locations with random numbers
  for (int_t i = 0; i < num_points; i++) {
    img_x[i] = rand() % image_width;
    img_y[i] = rand() % image_height;
  }
  //convert to the camera coordiante and then back to image coordiates
  test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y);
  test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params);
  test_cam.cam_to_sensor(cam_x, cam_y, cam_z, sen_x, sen_y);
  test_cam.sensor_to_image(sen_x, sen_y, ret_x, ret_y);
  //the miage coordinates should be the same
  for (int_t i = 0; i < num_points; i++) {
    if (max_deviation < abs(ret_x[i] - img_x[i])) max_deviation = abs(ret_x[i] - img_x[i]);
    if (max_deviation < abs(ret_y[i] - img_y[i])) max_deviation = abs(ret_y[i] - img_y[i]);
  }
  if (max_deviation > soft_match) {
    *outStream << "failed: sensor->cam, cam->sensor maximum deviation: " << max_deviation << " pixels on image" << std::endl;
    all_passed = false;
  }
  else *outStream << "passed: sensor->cam, cam->sensor maximum deviation: " << max_deviation << " pixels on image" << std::endl;

  //repeat with the world coordinate transform
  test_cam.cam_to_world(cam_x, cam_y, cam_z, wld_x, wld_y, wld_z);
  test_cam.world_to_cam(wld_x, wld_y, wld_z, cam_x, cam_y, cam_z);
  test_cam.cam_to_sensor(cam_x, cam_y, cam_z, sen_x, sen_y);
  test_cam.sensor_to_image(sen_x, sen_y, ret_x, ret_y);

  for (int_t i = 0; i < num_points; i++) {
    if (max_deviation < abs(ret_x[i] - img_x[i])) max_deviation = abs(ret_x[i] - img_x[i]);
    if (max_deviation < abs(ret_y[i] - img_y[i])) max_deviation = abs(ret_y[i] - img_y[i]);
  }
  if (max_deviation > soft_match) {
    *outStream << "failed: cam->world, world->cam maximum deviation: " << max_deviation << " pixels on image" << std::endl;
    all_passed = false;
  }
  else *outStream << "passed: cam->world, world->cam maximum deviation: " << max_deviation << " pixels on image" << std::endl;

  //***********************Test the derivitives****************************************
  d1_dx.clear();
  d1_dx.resize(3);
  d1_dy.clear();
  d1_dy.resize(3);
  d1_dz.clear();
  d1_dz.resize(3);
  d2_dx.clear();
  d2_dx.resize(3);
  d2_dy.clear();
  d2_dy.resize(3);
  d2_dz.clear();
  d2_dz.resize(3);
  dn_dx.clear();
  dn_dx.resize(3);
  dn_dy.clear();
  dn_dy.resize(3);
  dn_dz.clear();
  dn_dz.resize(3);
  der_dels.clear();
  der_dels.resize(3);
  der_aves.clear();
  der_aves.resize(3);


  for (int_t i = 0; i < 3; i++) {
    d1_dx[i].assign(num_points, 0.0);
    d1_dy[i].assign(num_points, 0.0);
    d1_dz[i].assign(num_points, 0.0);
    d2_dx[i].assign(num_points, 0.0);
    d2_dy[i].assign(num_points, 0.0);
    d2_dz[i].assign(num_points, 0.0);
    dn_dx[i].assign(num_points, 0.0);
    dn_dy[i].assign(num_points, 0.0);
    dn_dz[i].assign(num_points, 0.0);
    der_dels[i].assign(3, 0.0);
    der_aves[i].assign(3, 0.0);
  }

  DEBUG_MSG(" *************Sensor to cam derivitives *****************************************");
  //first derivitives of the projection function alone
  //calculate numberical first derivitives
  test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y);
  test_cam.sensor_to_cam(sen_x, sen_y, ret_x, ret_y, ret_z, proj_params);
  proj_params2[Projection_Shape_Function::ZP] = proj_params[Projection_Shape_Function::ZP] + params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, ret2_x, ret2_y, ret2_z, proj_params2);
  proj_params2[Projection_Shape_Function::ZP] = proj_params[Projection_Shape_Function::ZP] - params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, ret1_x, ret1_y, ret1_z, proj_params2);
  for (int_t i = 0; i < num_points; i++) {
    dn_dx[Projection_Shape_Function::ZP][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
    dn_dy[Projection_Shape_Function::ZP][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
    dn_dz[Projection_Shape_Function::ZP][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
  }

  proj_params2[Projection_Shape_Function::ZP] = proj_params[Projection_Shape_Function::ZP];
  proj_params2[Projection_Shape_Function::THETA] = proj_params[Projection_Shape_Function::THETA] + params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, ret2_x, ret2_y, ret2_z, proj_params2);
  proj_params2[Projection_Shape_Function::THETA] = proj_params[Projection_Shape_Function::THETA] - params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, ret1_x, ret1_y, ret1_z, proj_params2);
  for (int_t i = 0; i < num_points; i++) {
    dn_dx[Projection_Shape_Function::THETA][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
    dn_dy[Projection_Shape_Function::THETA][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
    dn_dz[Projection_Shape_Function::THETA][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
  }

  proj_params2[Projection_Shape_Function::THETA] = proj_params[Projection_Shape_Function::THETA];
  proj_params2[Projection_Shape_Function::PHI] = proj_params[Projection_Shape_Function::PHI] + params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, ret2_x, ret2_y, ret2_z, proj_params2);
  proj_params2[Projection_Shape_Function::PHI] = proj_params[Projection_Shape_Function::PHI] - params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, ret1_x, ret1_y, ret1_z, proj_params2);
  for (int_t i = 0; i < num_points; i++) {
    dn_dx[Projection_Shape_Function::PHI][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
    dn_dy[Projection_Shape_Function::PHI][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
    dn_dz[Projection_Shape_Function::PHI][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
  }
  //compair the numberical derivitives to the calculated ones
  //print out additional information for the first ten points
  int_t disp_pnts = 10;
  if (disp_pnts > num_points) disp_pnts = num_points;
  test_cam.sensor_to_cam(sen_x, sen_y, ret2_x, ret2_y, ret2_z, proj_params, d1_dx, d1_dy, d1_dz);
  for (int_t i = 0; i < disp_pnts; i++) {
    DEBUG_MSG(" ");
    DEBUG_MSG("   dZP(x y z): X: " << dn_dx[Projection_Shape_Function::ZP][i] << " " << d1_dx[Projection_Shape_Function::ZP][i] <<
      " Y: " << dn_dy[Projection_Shape_Function::ZP][i] << " " << d1_dy[Projection_Shape_Function::ZP][i] << " Z: " <<
      dn_dz[Projection_Shape_Function::ZP][i] << " " << d1_dz[Projection_Shape_Function::ZP][i]);
    DEBUG_MSG("dTHETA(x y z): X: " << dn_dx[Projection_Shape_Function::THETA][i] << " " << d1_dx[Projection_Shape_Function::THETA][i] <<
      " Y: " << dn_dy[Projection_Shape_Function::THETA][i] << " " << d1_dy[Projection_Shape_Function::THETA][i] << " Z: " <<
      dn_dz[Projection_Shape_Function::THETA][i] << " " << d1_dz[Projection_Shape_Function::THETA][i]);
    DEBUG_MSG("  dPHI(x y z): X: " << dn_dx[Projection_Shape_Function::PHI][i] << " " << d1_dx[Projection_Shape_Function::PHI][i] <<
      " Y: " << dn_dy[Projection_Shape_Function::PHI][i] << " " << d1_dy[Projection_Shape_Function::PHI][i] << " Z: " <<
      dn_dz[Projection_Shape_Function::PHI][i] << " " << d1_dz[Projection_Shape_Function::PHI][i]);
  }

  //calculate the average derivitive value and the maximum deviation between the numerical and analyitical derivitives
  for (int_t i = 0; i < 3; i++) {
    der_dels[i].assign(3, 0.0);
    der_aves[i].assign(3, 0.0);
  }

  for (int_t j = 0; j < 3; j++) {
    for (int_t i = 0; i < num_points; i++) {
      der_aves[j][0] += abs(d1_dx[j][i]);
      der_aves[j][1] += abs(d1_dy[j][i]);
      der_aves[j][2] += abs(d1_dz[j][i]);

      if (der_dels[j][0] < abs(dn_dx[j][i] - d1_dx[j][i])) der_dels[j][0] = abs(dn_dx[j][i] - d1_dx[j][i]);
      if (der_dels[j][1] < abs(dn_dy[j][i] - d1_dy[j][i])) der_dels[j][1] = abs(dn_dy[j][i] - d1_dy[j][i]);
      if (der_dels[j][2] < abs(dn_dz[j][i] - d1_dz[j][i])) der_dels[j][2] = abs(dn_dz[j][i] - d1_dz[j][i]);
    }
    der_aves[j][0] = der_aves[j][0] / num_points;
    der_aves[j][1] = der_aves[j][1] / num_points;
    der_aves[j][2] = der_aves[j][2] / num_points;
  }
  DEBUG_MSG(" ");
  DEBUG_MSG("   dZP(x y z) (ave,max dev): X: (" << der_aves[Projection_Shape_Function::ZP][0] <<
    ", " << der_dels[Projection_Shape_Function::ZP][0] << ") Y: (" << der_aves[Projection_Shape_Function::ZP][1] <<
    ", " << der_dels[Projection_Shape_Function::ZP][1] <<
    ") Z: (" << der_aves[Projection_Shape_Function::ZP][2] << ", " << der_dels[Projection_Shape_Function::ZP][2] << ")" << std::endl);
  DEBUG_MSG("dTheta(x y z) (ave,max dev): X: (" << der_aves[Projection_Shape_Function::THETA][0] << ", " <<
    der_dels[Projection_Shape_Function::THETA][0] << ") Y: (" << der_aves[Projection_Shape_Function::THETA][1] <<
    ", " << der_dels[Projection_Shape_Function::THETA][1] <<
    ") Z: (" << der_aves[Projection_Shape_Function::THETA][2] << ", " << der_dels[Projection_Shape_Function::THETA][2] <<
    ")" << std::endl);
  DEBUG_MSG("  dPhi(x y z) (ave,max dev): X: (" << der_aves[Projection_Shape_Function::PHI][0] << ", " <<
    der_dels[Projection_Shape_Function::PHI][0] << ") Y: (" << der_aves[Projection_Shape_Function::PHI][1] <<
    ", " << der_dels[Projection_Shape_Function::PHI][1] <<
    ") Z: (" << der_aves[Projection_Shape_Function::PHI][2] << ", " << der_dels[Projection_Shape_Function::PHI][2] << ")" << std::endl);

  DEBUG_MSG(" ");
  DEBUG_MSG(" *************sensor to world derivitives *****************************************");
  //calculate and print out the values including the transformation to world coordinates.
  test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y);
  test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params);
  test_cam.cam_to_world(cam_x, cam_y, cam_z, ret_x, ret_y, ret_z);

  proj_params2[Projection_Shape_Function::ZP] = proj_params[Projection_Shape_Function::ZP] + params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
  test_cam.cam_to_world(cam_x, cam_y, cam_z, ret2_x, ret2_y, ret2_z);
  proj_params2[Projection_Shape_Function::ZP] = proj_params[Projection_Shape_Function::ZP] - params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
  test_cam.cam_to_world(cam_x, cam_y, cam_z, ret1_x, ret1_y, ret1_z);
  for (int_t i = 0; i < num_points; i++) {
    dn_dx[Projection_Shape_Function::ZP][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
    dn_dy[Projection_Shape_Function::ZP][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
    dn_dz[Projection_Shape_Function::ZP][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
  }

  proj_params2[Projection_Shape_Function::ZP] = proj_params[Projection_Shape_Function::ZP];
  proj_params2[Projection_Shape_Function::THETA] = proj_params[Projection_Shape_Function::THETA] + params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
  test_cam.cam_to_world(cam_x, cam_y, cam_z, ret2_x, ret2_y, ret2_z);
  proj_params2[Projection_Shape_Function::THETA] = proj_params[Projection_Shape_Function::THETA] - params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
  test_cam.cam_to_world(cam_x, cam_y, cam_z, ret1_x, ret1_y, ret1_z);
  for (int_t i = 0; i < num_points; i++) {
    dn_dx[Projection_Shape_Function::THETA][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
    dn_dy[Projection_Shape_Function::THETA][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
    dn_dz[Projection_Shape_Function::THETA][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
  }

  proj_params2[Projection_Shape_Function::THETA] = proj_params[Projection_Shape_Function::THETA];
  proj_params2[Projection_Shape_Function::PHI] = proj_params[Projection_Shape_Function::PHI] + params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
  test_cam.cam_to_world(cam_x, cam_y, cam_z, ret2_x, ret2_y, ret2_z);
  proj_params2[Projection_Shape_Function::PHI] = proj_params[Projection_Shape_Function::PHI] - params_delta;
  test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
  test_cam.cam_to_world(cam_x, cam_y, cam_z, ret1_x, ret1_y, ret1_z);
  for (int_t i = 0; i < num_points; i++) {
    dn_dx[Projection_Shape_Function::PHI][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
    dn_dy[Projection_Shape_Function::PHI][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
    dn_dz[Projection_Shape_Function::PHI][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
  }

  //print out additional information for the first ten points
  if (disp_pnts > num_points) disp_pnts = num_points;
  test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params, d1_dx, d1_dy, d1_dz);
  test_cam.cam_to_world(cam_x, cam_y, cam_z, ret1_x, ret1_y, ret1_z, d1_dx, d1_dy, d1_dz, d2_dx, d2_dy, d2_dz);
  for (int_t i = 0; i < disp_pnts; i++) {
    DEBUG_MSG(" ");
    DEBUG_MSG("   dZP(x y z): X: " << dn_dx[Projection_Shape_Function::ZP][i] << " " << d2_dx[Projection_Shape_Function::ZP][i] <<
      " Y: " << dn_dy[Projection_Shape_Function::ZP][i] << " " << d2_dy[Projection_Shape_Function::ZP][i] << " Z: " <<
      dn_dz[Projection_Shape_Function::ZP][i] << " " << d2_dz[Projection_Shape_Function::ZP][i]);
    DEBUG_MSG("dTHETA(x y z): X: " << dn_dx[Projection_Shape_Function::THETA][i] << " " << d2_dx[Projection_Shape_Function::THETA][i] <<
      " Y: " << dn_dy[Projection_Shape_Function::THETA][i] << " " << d2_dy[Projection_Shape_Function::THETA][i] << " Z: " <<
      dn_dz[Projection_Shape_Function::THETA][i] << " " << d2_dz[Projection_Shape_Function::THETA][i]);
    DEBUG_MSG("  dPHI(x y z): X: " << dn_dx[Projection_Shape_Function::PHI][i] << " " << d2_dx[Projection_Shape_Function::PHI][i] <<
      " Y: " << dn_dy[Projection_Shape_Function::PHI][i] << " " << d2_dy[Projection_Shape_Function::PHI][i] << " Z: " <<
      dn_dz[Projection_Shape_Function::PHI][i] << " " << d2_dz[Projection_Shape_Function::PHI][i]);
  }

  //calculate the averages and deviations
  for (int_t i = 0; i < 3; i++) {
    der_dels[i].assign(3, 0.0);
    der_aves[i].assign(3, 0.0);
  }

  for (int_t j = 0; j < 3; j++) {
    for (int_t i = 0; i < num_points; i++) {
      der_aves[j][0] += abs(d2_dx[j][i]);
      der_aves[j][1] += abs(d2_dy[j][i]);
      der_aves[j][2] += abs(d2_dz[j][i]);

      if (der_dels[j][0] < abs(dn_dx[j][i] - d2_dx[j][i])) der_dels[j][0] = abs(dn_dx[j][i] - d2_dx[j][i]);
      if (der_dels[j][1] < abs(dn_dy[j][i] - d2_dy[j][i])) der_dels[j][1] = abs(dn_dy[j][i] - d2_dy[j][i]);
      if (der_dels[j][2] < abs(dn_dz[j][i] - d2_dz[j][i])) der_dels[j][2] = abs(dn_dz[j][i] - d2_dz[j][i]);
    }
    der_aves[j][0] = der_aves[j][0] / num_points;
    der_aves[j][1] = der_aves[j][1] / num_points;
    der_aves[j][2] = der_aves[j][2] / num_points;
  }
  DEBUG_MSG(" ");
  DEBUG_MSG("   dZP(x y z) (ave,max dev): X: (" << der_aves[Projection_Shape_Function::ZP][0] << ", " << der_dels[Projection_Shape_Function::ZP][0] <<
    ") Y: (" << der_aves[Projection_Shape_Function::ZP][1] << ", " << der_dels[Projection_Shape_Function::ZP][1] <<
    ") Z: (" << der_aves[Projection_Shape_Function::ZP][2] << ", " << der_dels[Projection_Shape_Function::ZP][2] << ")" << std::endl);
  DEBUG_MSG("dTheta(x y z) (ave,max dev): X: (" << der_aves[Projection_Shape_Function::THETA][0] << ", " << der_dels[Projection_Shape_Function::THETA][0] <<
    ") Y: (" << der_aves[Projection_Shape_Function::THETA][1] << ", " << der_dels[Projection_Shape_Function::THETA][1] <<
    ") Z: (" << der_aves[Projection_Shape_Function::THETA][2] << ", " << der_dels[Projection_Shape_Function::THETA][2] << ")" << std::endl);
  DEBUG_MSG("  dPhi(x y z) (ave,max dev): X: (" << der_aves[Projection_Shape_Function::PHI][0] << ", " << der_dels[Projection_Shape_Function::PHI][0] <<
    ") Y: (" << der_aves[Projection_Shape_Function::PHI][1] << ", " << der_dels[Projection_Shape_Function::PHI][1] <<
    ") Z: (" << der_aves[Projection_Shape_Function::PHI][2] << ", " << der_dels[Projection_Shape_Function::PHI][2] << ")" << std::endl);


  DEBUG_MSG("testing the inverse lens distortion methods");

  DICe::Camera::Camera_Info ci;
  ci.intrinsics_[Camera::CX] = 9.550537385e+02;
  ci.intrinsics_[Camera::CY] = 6.452301624e+02;
  ci.intrinsics_[Camera::FX] = 6.636033383e+03;
  ci.intrinsics_[Camera::FY] = 6.636440175e+03;
  ci.intrinsics_[Camera::FS] = -5.693077075e-04;
  ci.intrinsics_[Camera::K1] = -1.957515272e-04;
  ci.intrinsics_[Camera::K2] = 8.739544595e-01;
  ci.intrinsics_[Camera::K3] = -2.691426143e+00;
  ci.image_height_ = 1200;
  ci.image_width_  = 1920;

  DEBUG_MSG("NO LENS DISTORTION");
  DICe::Camera dist_cam_no_lens_dist(ci);
  DEBUG_MSG("K1R1_K2R2_K3R3 LENS DISTORTION");
  ci.lens_distortion_model_ = Camera::K1R1_K2R2_K3R3;
  DICe::Camera dist_cam_k1r1(ci);
  DEBUG_MSG("K1R2_K2R4_K3R6 LENS DISTORTION");
  ci.lens_distortion_model_ = Camera::K1R2_K2R4_K3R6;
  DICe::Camera dist_cam_k1r2(ci);
  DEBUG_MSG("K1R3_K2R5_K3R7 LENS DISTORTION");
  ci.lens_distortion_model_ = Camera::K1R3_K2R5_K3R7;
  DICe::Camera dist_cam_k1r3(ci);
  DEBUG_MSG("VIC3D LENS DISTORTION");
  ci.lens_distortion_model_ = Camera::VIC3D_LENS_DISTORTION;
  DICe::Camera dist_cam_vic3d(ci);
  DEBUG_MSG("OPENCV LENS DISTORTION");
  ci.lens_distortion_model_ = Camera::OPENCV_LENS_DISTORTION;
  DICe::Camera dist_cam_opencv(ci);



 // TODO cycle through the lens distortion models

  DICe::finalize();

  if (!all_passed)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

