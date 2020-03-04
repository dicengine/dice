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

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "netcdf.h"

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

  // Open the file for read access
  int ncid_left = 0, ncid_right = 0;
  int error_int = nc_open(left_file.c_str(), 0, &ncid_left);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::runtime_error,"Error, could not open NetCDF file " << left_file);
  error_int = nc_open(right_file.c_str(), 0, &ncid_right);
  TEUCHOS_TEST_FOR_EXCEPTION(error_int,std::runtime_error,"Error, could not open NetCDF file " << right_file);
  int num_vars_left = 0, num_vars_right = 0;
  nc_inq_nvars(ncid_left, &num_vars_left);
  nc_inq_nvars(ncid_right, &num_vars_right);
  TEUCHOS_TEST_FOR_EXCEPTION(num_vars_left!=num_vars_right,std::runtime_error,"");

  std::string coord_x_str = "x";
  std::string coord_y_str = "y";
  std::string scale_str = "scale_factor";
  std::string offset_str = "add_offset";
  std::string proj_str = "goes_imager_projection";
  std::string proj_height = "perspective_point_height";
  std::string proj_major = "semi_major_axis";
  std::string proj_minor = "semi_minor_axis";
  std::string proj_long_origin = "longitude_of_projection_origin";
  float perspective_point_height_left = 0.0;
  float semi_major_axis_left = 0.0;
  float semi_minor_axis_left = 0.0;
  float long_of_proj_origin_left = 0.0;
  float coord_x_offset_left = 0.0;
  float coord_y_offset_left = 0.0;
  float coord_x_scale_factor_left = 0.0;
  float coord_y_scale_factor_left = 0.0;
  float perspective_point_height_right = 0.0;
  float semi_major_axis_right = 0.0;
  float semi_minor_axis_right = 0.0;
  float long_of_proj_origin_right = 0.0;
  float coord_x_offset_right = 0.0;
  float coord_y_offset_right = 0.0;
  float coord_x_scale_factor_right = 0.0;
  float coord_y_scale_factor_right = 0.0;
  float att_value = 0.0;
  // harvest background info for both imagers:
  for(int i=0;i<num_vars_left;++i){
    char var_name[100];
    int nc_type;
    int num_dims = 0;
    int dim_ids[NC_MAX_VAR_DIMS]; // assume less than 100 ids
    int num_var_attr = 0;
    nc_inq_var(ncid_left,i, &var_name[0], &nc_type,&num_dims, dim_ids, &num_var_attr);
    nc_inq_varname(ncid_left, i, &var_name[0]);
    std::string var_name_str = var_name;
    if(strcmp(var_name, proj_str.c_str()) == 0){
      nc_get_att_float(ncid_left,i,proj_height.c_str(),&att_value);
      perspective_point_height_left = att_value;
      nc_get_att_float(ncid_left,i,proj_major.c_str(),&att_value);
      semi_major_axis_left = att_value;
      nc_get_att_float(ncid_left,i,proj_minor.c_str(),&att_value);
      semi_minor_axis_left = att_value;
      nc_get_att_float(ncid_left,i,proj_long_origin.c_str(),&att_value);
      long_of_proj_origin_left = att_value;
    }else if(strcmp(var_name, coord_x_str.c_str()) == 0){
      nc_get_att_float(ncid_left,i,scale_str.c_str(),&att_value);
      coord_x_scale_factor_left = att_value;
      nc_get_att_float(ncid_left,i,offset_str.c_str(),&att_value);
      coord_x_offset_left = att_value;
    }else if(strcmp(var_name, coord_y_str.c_str()) == 0){
      nc_get_att_float(ncid_left,i,scale_str.c_str(),&att_value);
      coord_y_scale_factor_left = att_value;
      nc_get_att_float(ncid_left,i,offset_str.c_str(),&att_value);
      coord_y_offset_left = att_value;
    }
  }
  DEBUG_MSG("imager height left (m):           " << perspective_point_height_left);
  DEBUG_MSG("earth major axis left (m):        " << semi_major_axis_left);
  DEBUG_MSG("earth minor axis left (m):        " << semi_minor_axis_left);
  DEBUG_MSG("longitude of left imager (deg):   " << long_of_proj_origin_left);
  DEBUG_MSG("imager x scale factor left:       " << coord_x_scale_factor_left);
  DEBUG_MSG("imager x offset left (rad):       " << coord_x_offset_left);
  DEBUG_MSG("imager y scale factor left:       " << coord_y_scale_factor_left);
  DEBUG_MSG("imager y offset left (rad):       " << coord_y_offset_left);
  for(int i=0;i<num_vars_right;++i){
    char var_name[100];
    int nc_type;
    int num_dims = 0;
    int dim_ids[NC_MAX_VAR_DIMS]; // assume less than 100 ids
    int num_var_attr = 0;
    nc_inq_var(ncid_right,i, &var_name[0], &nc_type,&num_dims, dim_ids, &num_var_attr);
    nc_inq_varname(ncid_right, i, &var_name[0]);
    std::string var_name_str = var_name;
    if(strcmp(var_name, proj_str.c_str()) == 0){
      float att_value = 0.0;
      nc_get_att_float(ncid_right,i,proj_height.c_str(),&att_value);
      perspective_point_height_right = att_value;
      nc_get_att_float(ncid_right,i,proj_major.c_str(),&att_value);
      semi_major_axis_right = att_value;
      nc_get_att_float(ncid_right,i,proj_minor.c_str(),&att_value);
      semi_minor_axis_right = att_value;
      nc_get_att_float(ncid_right,i,proj_long_origin.c_str(),&att_value);
      long_of_proj_origin_right = att_value;
    }else if(strcmp(var_name, coord_x_str.c_str()) == 0){
      nc_get_att_float(ncid_right,i,scale_str.c_str(),&att_value);
      coord_x_scale_factor_right = att_value;
      nc_get_att_float(ncid_right,i,offset_str.c_str(),&att_value);
      coord_x_offset_right = att_value;
    }else if(strcmp(var_name, coord_y_str.c_str()) == 0){
      nc_get_att_float(ncid_right,i,scale_str.c_str(),&att_value);
      coord_y_scale_factor_right = att_value;
      nc_get_att_float(ncid_right,i,offset_str.c_str(),&att_value);
      coord_y_offset_right = att_value;
    }
  }
  // close the nc_files
  nc_close(ncid_left);
  nc_close(ncid_right);
  DEBUG_MSG("imager height right (m):          " << perspective_point_height_right);
  DEBUG_MSG("earth major axis right (m):       " << semi_major_axis_right);
  DEBUG_MSG("earth minor axis right (m):       " << semi_minor_axis_right);
  DEBUG_MSG("longitude of imager right (deg):  " << long_of_proj_origin_right);
  DEBUG_MSG("imager x scale factor right:      " << coord_x_scale_factor_right);
  DEBUG_MSG("imager x offset right (rad):      " << coord_x_offset_right);
  DEBUG_MSG("imager y scale factor right:      " << coord_y_scale_factor_right);
  DEBUG_MSG("imager y offset right (rad):      " << coord_y_offset_right);
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(perspective_point_height_right-perspective_point_height_left)>0.1,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(semi_major_axis_right-semi_major_axis_left)>0.1,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(semi_minor_axis_right-semi_minor_axis_left)>0.1,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(coord_x_scale_factor_right-coord_x_scale_factor_left)>0.001,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(coord_y_scale_factor_right-coord_y_scale_factor_left)>0.001,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(coord_x_offset_right-coord_x_offset_left)>0.001,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(coord_y_offset_right-coord_y_offset_left)>0.001,std::runtime_error,"");

  DEBUG_MSG("converting image scan angle in x and y to lat and long for both images");

  // convert pixel coordinates to longitude and latitude:
  const float r_eq = semi_major_axis_right;
  const float r_np = semi_minor_axis_right;
  const float H = perspective_point_height_right + r_eq;
  const float offset_x = coord_x_offset_left;
  const float offset_y = coord_y_offset_left;
  const float scale_x = coord_x_scale_factor_left;
  const float scale_y = coord_y_scale_factor_left;
  const float eccentricity = (r_eq*r_eq - r_np*r_np)/(r_eq*r_eq);

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
//  int x = 7364;
//  int y = 2204;
    const float y_rad = y*scale_y + offset_y;
    for(int x=grid_start_x;x<grid_end_x;x+=grid_size){
      if(std::sqrt((x-w/2)*(x-w/2)+(y-h/2)*(y-h/2)) > 0.95*(w/2.0)) continue; //skip points off the earth's surface
      left_pixel_x.push_back(x);
      left_pixel_y.push_back(y);
      // convert from pixel to scan angle
      const float x_rad = x*scale_x + offset_x;
      // convert from scan angle to satellite coords
      const float sinx = std::sin(x_rad);
      const float cosx = std::cos(x_rad);
      const float cosy = std::cos(y_rad);
      const float siny = std::sin(y_rad);
      const float a = sinx*sinx + cosx*cosx*(cosy*cosy + r_eq*r_eq*siny*siny/(r_np*r_np));
      const float b = -2.0*H*cosx*cosy;
      const float c = H*H - r_eq*r_eq;
      const float d = b*b - 4.0*a*c;
      const float r_s = d > 0.0? (-1.0*b - std::sqrt(d))/(2.0*a):1.0;
      const float sx = r_s*cosx*cosy;
      const float sy = -1.0*r_s*sinx;
      const float sz = r_s*cosx*siny;
      // convert the satellite coordinates to lat and long
      const float lat = d > 0.0? std::atan((r_eq*r_eq)/(r_np*r_np)*sz/std::sqrt((H-sx)*(H-sx)+sy*sy)) : -1.0;
      const float lon_left = (DICE_PI/180.0)*long_of_proj_origin_left - std::atan(sy/(H-sx));
      //const float lon_right = (DICE_PI/180.0)*long_of_proj_origin_right - std::atan(sy/(H-sx));
      // convert lat and long to x y z on the surface of the earth (with earth centered coords)
      const float sin_lat = std::sin(lat);
      const float sin_lon_left = std::sin(lon_left);
      const float cos_lat = std::cos(lat);
      const float cos_lon_left = std::cos(lon_left);
      const float N = r_eq/(std::sqrt(1.0 - eccentricity*sin_lat*sin_lat));
      const float px = N*cos_lat*cos_lon_left;
      const float py = N*cos_lat*sin_lon_left;
      const float pz = (1.0-eccentricity*eccentricity)*N*sin_lat;
      earth_x.push_back(px);
      earth_y.push_back(py);
      earth_z.push_back(pz);
      //std::cout << left_pixel_x.size() << " " << px << " " << py << " " << pz << std::endl;
      // convert from earth-centered coords to right camera pixel
      const float plen = std::sqrt(px*px+py*py);
      const float cos_adiff = std::cos((DICE_PI/180.0)*long_of_proj_origin_right - lon_left);
      const float sin_adiff = std::sin((DICE_PI/180.0)*long_of_proj_origin_right - lon_left);
      const float psi = -1.0*std::tan(plen*sin_adiff/(H - plen*cos_adiff));
      right_pixel_x.push_back((psi - offset_x)/scale_x);
      right_pixel_y.push_back(y); // assume y pixel is the same for both imagers
//      right_x.push_back(N*cos_lat*cos_lon_right);
//      right_y.push_back(N*cos_lat*sin_lon_right);
//      right_z.push_back((1.0-eccentricity*eccentricity)*N*sin_lat);
//      latitude.push_back(lat);
//      longitude_left.push_back(long_of_proj_origin_left - std::atan(sy/(H-sx))*180.0/DICE_PI);
//      longitude_right.push_back(long_of_proj_origin_right - std::atan(sy/(H-sx))*180.0/DICE_PI);
//      std::cout << "point:  x " << left_x[left_x.size()-1] << " y " << left_y[left_y.size()-1] << " z " << left_z[left_z.size()-1] <<  " rpx "
//          << right_pixel_x[right_pixel_x.size()-1] << " rpy " << right_pixel_y[right_pixel_y.size()-1] << std::endl;
//      std::cout << " lpx " << left_pixel_x[left_pixel_x.size()-1] << " lpy " << left_pixel_y[left_pixel_y.size()-1] <<
//          " rpx " << right_pixel_x[right_pixel_x.size()-1] << " rpy " << right_pixel_y[right_pixel_y.size()-1] << std::endl;
    }
  }
  const int num_cal_pts = left_pixel_x.size();
  DEBUG_MSG("cal point grid num points:        " << left_pixel_x.size());

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
  cv::Mat R, T, E, F;
  const int options = cv::CALIB_ZERO_TANGENT_DIST
      + cv::CALIB_USE_INTRINSIC_GUESS;
//      + cv::CALIB_FIX_INTRINSIC
//      + cv::CALIB_FIX_PRINCIPAL_POINT
//      + cv::CALIB_FIX_K1 + cv::CALIB_FIX_K2 + cv::CALIB_FIX_K3
//     + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5 + cv::CALIB_FIX_K6;

  float rms = cv::stereoCalibrate(object_points, image_points_left, image_points_right,
        camera_matrix[0], dist_coeffs[0],
        camera_matrix[1], dist_coeffs[1],
        img_size, R, T, E, F,
        options,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 1e-7));

  std::cout << "***RMS error: " << rms << std::endl;

//  std::cout << " camera left " << camera_matrix_left << std::endl;
//  std::cout << " camera right " << camera_matrix_right << std::endl;
//  std::cout << " T " << T << std::endl;
//  std::cout << " R " << R << std::endl;
//  std::cout << " E " << E << std::endl;
//  std::cout << " F " << F << std::endl;
//  std::cout << " dist left " << dist_coeffs_left << std::endl;
//  std::cout << " dist right " << dist_coeffs_right << std::endl;

  Matrix<scalar_t,3> dice_R;
  for (size_t i=0; i<3; i++) {
    for (size_t j=0; j<3; j++) {
      dice_R(i,j) = R.at<double>(i,j);
    }
  }
//  Matrix<scalar_t,3> R_prod = dice_R.transpose()*dice_R;
//  std::cout << " R " << dice_R << " R_prod " << R_prod << " " << dice_R.transpose() <<  std::endl;
  scalar_t alpha = 0.0;
  scalar_t beta = 0.0;
  scalar_t gamma = 0.0;
  Camera::Camera_Info::rotation_matrix_to_eulers(dice_R,alpha,beta,gamma);
  std::cout << "alpha " << alpha*180.0/DICE_PI << " (deg) beta " << beta*180.0/DICE_PI << " (deg) gamma " << gamma*180.0/DICE_PI << " (deg)" << std::endl;

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
    if(i_cam==1){
      camera_info.tx_ = T.at<double>(0);
      camera_info.ty_ = T.at<double>(1);
      camera_info.tz_ = T.at<double>(2);
      for (size_t i_a = 0; i_a < 3; i_a++) {
        for (size_t i_b = 0; i_b < 3; i_b++) {
          camera_info.rotation_matrix_(i_a,i_b) = R.at<double>(i_a, i_b);
        }
      }
    }
    Teuchos::RCP<DICe::Camera> camera_ptr = Teuchos::rcp(new DICe::Camera(camera_info));
    camera_system->add_camera(camera_ptr);
  }
  std::cout << camera_system << std::endl;
  camera_system->set_system_type(Camera_System::OPENCV);
  if(!output_file.empty())
    camera_system->write_camera_system_file(output_file);

  std::cout << "\nDICe_CalSatellite complete\n" << std::endl;

  DICe::finalize();

  return 0;
}

