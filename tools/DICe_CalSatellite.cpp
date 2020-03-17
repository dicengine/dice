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
#include <DICe_Calibration.h>
#include <DICe_Camera.h>
#include <DICe_CameraSystem.h>
#include <DICe_Image.h>
#include <DICe_ImageIO.h>
#include <DICe_Matrix.h>
#include <DICe_NetCDF.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <cassert>

using namespace std;
using namespace DICe;

/// note: this assumes GOES16 or GOES17 FULL_DISK formatted data in a netcdf-4 file type

// REFS
// https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
// https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // read the parameters from the input file:
  if(argc!=3){ // executable, input file, output file name
    std::cout << "Usage: DICe_CalSatellite <cal_input_file.xml> <output_file.xml>" << std::endl;
    exit(1);
  }

  print_banner();
  std::cout << "\nDICe_CalSatellite begin\n" << std::endl;

  std::string input_file = argv[1];
  std::string output_file = argv[2];
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::Ptr<Teuchos::ParameterList> params_ptr(params.get());
  try {
    Teuchos::updateParametersFromXmlFile(input_file,params_ptr);
  }
  catch (std::exception & e) {
    std::cout << e.what() << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"invalid xml cal input file");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(params==Teuchos::null,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("left_file"),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("right_file"),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("estimated_focal_length_in_pixels"),std::runtime_error,"");
  const std::string left_file = params->get<std::string>("left_file");
  const std::string right_file = params->get<std::string>("right_file");
  const float focal_length_in_pixels = params->get<double>("estimated_focal_length_in_pixels");
  const int num_grid_divisions = params->get<int>("num_cal_grid_divisions",100);

  // determine the image size:
  int w = 0, right_w = 0;
  int h = 0, right_h = 0;
  DICe::utils::read_image_dimensions(left_file.c_str(),w,h);
  DICe::utils::read_image_dimensions(right_file.c_str(),right_w,right_h);
  TEUCHOS_TEST_FOR_EXCEPTION(w!=10848,std::runtime_error,"images need to be full disk");
  TEUCHOS_TEST_FOR_EXCEPTION(h!=10848,std::runtime_error,"images need to be full disk");
  TEUCHOS_TEST_FOR_EXCEPTION(w!=right_w,std::runtime_error,"both images need to be full disk and the same dimensions");
  TEUCHOS_TEST_FOR_EXCEPTION(h!=right_h,std::runtime_error,"both images need to be full disk and the same dimensions");
  cv::Size img_size(w,h);
  const float cx = w/2.0;
  const float cy = h/2.0;
  DEBUG_MSG("left image:  " << left_file);
  DEBUG_MSG("right image: " << right_file);
  DEBUG_MSG("image size:                       " << w << " x " << h);
  DEBUG_MSG("focal length (pixels):            " << focal_length_in_pixels);
  DEBUG_MSG("iniatal cx(pixels):               " << cx);
  DEBUG_MSG("iniatal cy(pixels):               " << cy);

  Teuchos::ParameterList lat_long_params = DICe::netcdf::netcdf_to_lat_long_projection_parameters(left_file,right_file);

  // convert pixel coordinates to longitude and latitude:

  const int grid_start_x = w/2; // only use the right side of the image (the left side is not visible to both cameras)
  const int grid_end_x = w;
  const int grid_start_y = 0;
  const int grid_end_y = h;
  const int grid_size = (grid_end_x - grid_start_x)/num_grid_divisions;
  TEUCHOS_TEST_FOR_EXCEPTION(grid_size<1||grid_size>w/4,std::runtime_error,"");
  DEBUG_MSG("cal point grid size (pixels):     " << grid_size);

  std::vector<float> earth_x;
  std::vector<float> earth_y;
  std::vector<float> earth_z;
  std::vector<float> left_pixel_x;
  std::vector<float> left_pixel_y;
  std::vector<float> right_pixel_x;
  std::vector<float> right_pixel_y;

  for(int y=grid_start_y;y<grid_end_y;y+=grid_size){
    for(int x=grid_start_x;x<grid_end_x;x+=grid_size){
      if(std::sqrt((x-w/2)*(x-w/2)+(y-h/2)*(y-h/2)) > 0.95*(w/2.0)) continue; //skip points off the earth's surface
      left_pixel_x.push_back(x);
      left_pixel_y.push_back(y);
    }
  }

  DICe::netcdf::netcdf_left_pixel_points_to_earth_and_right_pixel_coordinates(lat_long_params,
    left_pixel_x,
    left_pixel_y,
    earth_x,
    earth_y,
    earth_z,
    right_pixel_x,
    right_pixel_y);
  const int num_cal_pts = left_pixel_x.size();

  std::vector<std::vector<cv::Point3f> > object_points;
  std::vector<std::vector<cv::Point2f> > image_points_left;
  std::vector<std::vector<cv::Point2f> > image_points_right;

  object_points.push_back(std::vector<cv::Point3f>(num_cal_pts));
  image_points_left.push_back(std::vector<cv::Point2f>(num_cal_pts));
  image_points_right.push_back(std::vector<cv::Point2f>(num_cal_pts));
  for(int i=0;i<num_cal_pts;++i){
    object_points[0][i] = cv::Point3f(earth_x[i],earth_y[i],earth_z[i]);
    image_points_left[0][i] = cv::Point2f(left_pixel_x[i],left_pixel_y[i]);
    image_points_right[0][i] = cv::Point2f(right_pixel_x[i],right_pixel_y[i]);
  }

  DEBUG_MSG("calibrating");
  const int num_cams = 2;
  std::vector<cv::Mat> camera_matrix(num_cams);
  for(int i=0;i<num_cams;++i){
    camera_matrix[i] = cv::Mat::eye(3, 3, CV_32F);
    camera_matrix[i].at<float>(0,0) = focal_length_in_pixels;
    camera_matrix[i].at<float>(1,1) = focal_length_in_pixels;
    camera_matrix[i].at<float>(0,2) = cx;
    camera_matrix[i].at<float>(1,2) = cy;
  }
  std::vector<cv::Mat> dist_coeffs(num_cams);
  std::vector<cv::Mat> Rs(num_cams);
  std::vector<cv::Mat> Ts(num_cams);
  cv::Mat E, F;
  const int options = cv::CALIB_ZERO_TANGENT_DIST
      + cv::CALIB_USE_INTRINSIC_GUESS;
//      + cv::CALIB_FIX_INTRINSIC
//      + cv::CALIB_FIX_PRINCIPAL_POINT
//      + cv::CALIB_FIX_K1 + cv::CALIB_FIX_K2 + cv::CALIB_FIX_K3
//     + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5 + cv::CALIB_FIX_K6;

  float rms = cv::stereoCalibrate(object_points, image_points_left, image_points_right,
        camera_matrix[0], dist_coeffs[0],
        camera_matrix[1], dist_coeffs[1],
        img_size, Rs[1], Ts[1], E, F, // Rs and Ts use the second entry because this method finds the transform from right to left
        options,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 1e-7));

  Matrix<scalar_t,3> dice_R;
  for (size_t i=0; i<3; i++) {
    for (size_t j=0; j<3; j++) {
      dice_R(i,j) = Rs[1].at<double>(i,j);
    }
  }
//  Matrix<scalar_t,3> R_prod = dice_R.transpose()*dice_R;
//  std::cout << " R " << dice_R << " R_prod " << R_prod << " " << dice_R.transpose() <<  std::endl;
  scalar_t alpha = 0.0;
  scalar_t beta = 0.0;
  scalar_t gamma = 0.0;
  Camera::Camera_Info::rotation_matrix_to_eulers(dice_R,alpha,beta,gamma);
  std::cout << "orientation of the right camera (GOES 16) to the left (GOES 17) defined by the Euler angles:" << std::endl;
  std::cout << "alpha " << alpha*180.0/DICE_PI << " (deg) beta " << beta*180.0/DICE_PI << " (deg) gamma " << gamma*180.0/DICE_PI << " (deg)" << std::endl;
  std::cout << "*** RMS stereo error: " << rms << std::endl;
  //  std::cout << " camera left " << camera_matrix_left << std::endl;
  //  std::cout << " camera right " << camera_matrix_right << std::endl;
  //  std::cout << " T " << T << std::endl;
  //  std::cout << " R " << R << std::endl;
  //  std::cout << " E " << E << std::endl;
  //  std::cout << " F " << F << std::endl;
  //  std::cout << " dist left " << dist_coeffs_left << std::endl;
  //  std::cout << " dist right " << dist_coeffs_right << std::endl;

  // the default output from opencv is to provide the world or model coordinates in terms of camera 0 so the
  // R matrix for camera 0 is identity and the t-vecs are zeros. The R matrix output from stereocalibrate is from
  // the left camera to the right camera. To get the transformation from camera 0 to the earth-centeted coordinate
  // system we need to use calibrateCamera which provides the pose estimation of camera 0 (or the transform from world coords
  // to camera_0 coords.

  cv::Mat left_camera = camera_matrix[0].clone();
  cv::Mat dummy_coeffs;
  std::vector<cv::Mat> rvecs, tvecs;
  float rms_mono = cv::calibrateCamera(object_points, image_points_left,img_size,
    left_camera,dummy_coeffs,rvecs,tvecs,options,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 1e-7));
  cv::Rodrigues(rvecs[0],Rs[0]);
  Ts[0] = tvecs[0].clone();
  std::cout << Rs[0] << std::endl;
  std::cout << tvecs[0] << std::endl;

  std::cout << "*** RMS mono error: " << rms_mono << std::endl;

  Matrix<scalar_t,3> dice_R_;
  for (size_t i=0; i<3; i++) {
    for (size_t j=0; j<3; j++) {
      dice_R_(i,j) = Rs[0].at<double>(i,j);
    }
  }
//  Matrix<scalar_t,3> R_prod = dice_R.transpose()*dice_R;
//  std::cout << " R " << dice_R << " R_prod " << R_prod << " " << dice_R.transpose() <<  std::endl;
  scalar_t alpha_ = 0.0;
  scalar_t beta_ = 0.0;
  scalar_t gamma_ = 0.0;
  Camera::Camera_Info::rotation_matrix_to_eulers(dice_R_,alpha_,beta_,gamma_);
  std::cout << "orientation of the earth coords to left camera (GOES 17) defined by the Euler angles:" << std::endl;
  std::cout << "alpha_ " << alpha_*180.0/DICE_PI << " (deg) beta_ " << beta_*180.0/DICE_PI << " (deg) gamma_ " << gamma_*180.0/DICE_PI << " (deg)" << std::endl;

  // create calibration file:

  Teuchos::RCP<DICe::Camera_System> camera_system = Teuchos::rcp(new DICe::Camera_System());

  for (size_t i_cam = 0; i_cam < num_cams; i_cam++) {
    DICe::Camera::Camera_Info camera_info;
    camera_info.image_height_ = w;
    camera_info.image_width_ = h;
    //assign the intrinsic and extrinsic values for the first camera
    camera_info.intrinsics_[Camera::CX] = camera_matrix[i_cam].at<double>(0, 2);
    camera_info.intrinsics_[Camera::CY] = camera_matrix[i_cam].at<double>(1, 2);
    camera_info.intrinsics_[Camera::FX] = camera_matrix[i_cam].at<double>(0, 0);
    camera_info.intrinsics_[Camera::FY] = camera_matrix[i_cam].at<double>(1, 1);
    camera_info.intrinsics_[Camera::K1] = dist_coeffs[i_cam].at<double>(0);
    camera_info.intrinsics_[Camera::K2] = dist_coeffs[i_cam].at<double>(1);
    camera_info.intrinsics_[Camera::P1] = dist_coeffs[i_cam].at<double>(2);
    camera_info.intrinsics_[Camera::P2] = dist_coeffs[i_cam].at<double>(3);
    if (dist_coeffs[i_cam].cols > 4) camera_info.intrinsics_[Camera::K3] = dist_coeffs[i_cam].at<double>(4);
    if (dist_coeffs[i_cam].cols > 5) camera_info.intrinsics_[Camera::K4] = dist_coeffs[i_cam].at<double>(5);
    if (dist_coeffs[i_cam].cols > 6) camera_info.intrinsics_[Camera::K5] = dist_coeffs[i_cam].at<double>(6);
    if (dist_coeffs[i_cam].cols > 7) camera_info.intrinsics_[Camera::K6] = dist_coeffs[i_cam].at<double>(7);
    if (dist_coeffs[i_cam].cols > 8) camera_info.intrinsics_[Camera::S1] = dist_coeffs[i_cam].at<double>(8);
    if (dist_coeffs[i_cam].cols > 9) camera_info.intrinsics_[Camera::S2] = dist_coeffs[i_cam].at<double>(9);
    if (dist_coeffs[i_cam].cols > 10) camera_info.intrinsics_[Camera::S3] = dist_coeffs[i_cam].at<double>(10);
    if (dist_coeffs[i_cam].cols > 11) camera_info.intrinsics_[Camera::S4] = dist_coeffs[i_cam].at<double>(11);
    if (dist_coeffs[i_cam].cols > 12) camera_info.intrinsics_[Camera::T1] = dist_coeffs[i_cam].at<double>(12);
    if (dist_coeffs[i_cam].cols > 13) camera_info.intrinsics_[Camera::T2] = dist_coeffs[i_cam].at<double>(13);
    camera_info.lens_distortion_model_ = Camera::OPENCV_LENS_DISTORTION;
    //if(i_cam==1){
      camera_info.tx_ = Ts[i_cam].at<double>(0);
      camera_info.ty_ = Ts[i_cam].at<double>(1);
      camera_info.tz_ = Ts[i_cam].at<double>(2);
      for (size_t i_a = 0; i_a < 3; i_a++) {
        for (size_t i_b = 0; i_b < 3; i_b++) {
          camera_info.rotation_matrix_(i_a,i_b) = Rs[i_cam].at<double>(i_a, i_b);
        }
      }
    //}
    Teuchos::RCP<DICe::Camera> camera_ptr = Teuchos::rcp(new DICe::Camera(camera_info));
    camera_system->add_camera(camera_ptr);
  }
  camera_system->set_system_type(Camera_System::OPENCV);
  std::cout << camera_system << std::endl;
  if(!output_file.empty())
    camera_system->write_camera_system_file(output_file);

  std::cout << "\nDICe_CalSatellite complete\n" << std::endl;

  DICe::finalize();

  return 0;
}

