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

#include <DICe_Camera.h>
#include <DICe_CameraSystem.h>
#include <DICe_Calibration.h>
#include <DICe_Parser.h>
#include <DICe_XMLUtils.h>
#include <DICe_OpenCVServerUtils.h>
#include <DICe_ImageIO.h>

#include <Teuchos_XMLParameterListHelpers.hpp>

#include <fstream>

using namespace cv;

namespace DICe {

Calibration::Calibration(const std::string & cal_input_file){
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::Ptr<Teuchos::ParameterList> params_ptr(params.get());
  try {
    Teuchos::updateParametersFromXmlFile(cal_input_file,params_ptr);
  }
  catch (std::exception & e) {
    std::cout << e.what() << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"invalid xml cal input file");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(params==Teuchos::null,std::runtime_error,"");
  init(params);
}

void
Calibration::init(const Teuchos::RCP<Teuchos::ParameterList> params){
  // print out the parameters for debugging
  std::cout << "Calibration::init(): user specified input parameters:" << std::endl;
//#ifdef DICE_DEBUG_MSG
  params->print(std::cout);
//#endif
  // save the parameters for output later if needed
  input_params_ = *params;

  // read in the image file names
  std::vector<std::string> left_images;
  std::vector<std::string> right_images;
  DICe::decipher_image_file_names(params,left_images,right_images);
  // the decipher image file names function automatically adds the reference image to the start of the vec
  // skip that image if it exists
  if(left_images.size()>1){
    left_images.erase(left_images.begin());
    image_list_.push_back(left_images);
  }
  if(right_images.size()>1){
    right_images.erase(right_images.begin());
    image_list_.push_back(right_images);
  }

  // establish the calibration plate properties
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::num_cal_fiducials_x),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::num_cal_fiducials_y),std::runtime_error,"");
  num_fiducials_x_ = params->get<int_t>(DICe::num_cal_fiducials_x);
  num_fiducials_y_ = params->get<int_t>(DICe::num_cal_fiducials_y);
  TEUCHOS_TEST_FOR_EXCEPTION(num_fiducials_x_<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(num_fiducials_y_<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::cal_target_spacing_size),std::runtime_error,"");
  target_spacing_ = params->get<double>(DICe::cal_target_spacing_size);
  origin_loc_x_ = 0;
  origin_loc_y_ = 0;
  if(params->isParameter(DICe::cal_origin_x))
      origin_loc_x_ = params->get<int_t>(DICe::cal_origin_x);
  if(params->isParameter(DICe::cal_origin_y))
      origin_loc_y_ = params->get<int_t>(DICe::cal_origin_y);
  num_fiducials_origin_to_x_marker_ = num_fiducials_x_;
  num_fiducials_origin_to_y_marker_ = num_fiducials_y_;
  if(params->isParameter(DICe::num_cal_fiducials_origin_to_x_marker))
    num_fiducials_origin_to_x_marker_ = params->get<int_t>(DICe::num_cal_fiducials_origin_to_x_marker);
  if(params->isParameter(DICe::num_cal_fiducials_origin_to_y_marker))
    num_fiducials_origin_to_y_marker_ = params->get<int_t>(DICe::num_cal_fiducials_origin_to_y_marker);
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::cal_target_type),std::runtime_error,"");
  pose_estimation_index_ = params->get<int_t>(DICe::pose_estimation_index,-1);
  target_type_ = Calibration::string_to_target_type(params->get<std::string>(DICe::cal_target_type));
  draw_intersection_image_ = false;
  if(params->isParameter(DICe::draw_intersection_image))
    draw_intersection_image_ = params->get<bool>(DICe::draw_intersection_image);
  if(params->isParameter(DICe::cal_debug_folder)){
    debug_folder_ = params->get<std::string>(DICe::cal_debug_folder);
    create_directory(debug_folder_);
  }

  // read the calibration options
  Teuchos::ParameterList opencv_options;
  if(params->isSublist(DICe::cal_opencv_options)){
    opencv_options = params->get<Teuchos::ParameterList>(DICe::cal_opencv_options);
  }
  set_calibration_options(opencv_options);

  //try to use a consistant coordinate grid system for all targets
  grid_points_.resize(num_fiducials_x_); // = std::vector<std::vector<cv::Point3f>(num_fiducials_y_)>(num_fiducials_x_);
  for (int_t m = 0; m < num_fiducials_x_; m++) {
    grid_points_[m].resize(num_fiducials_y_);
    for (int_t n = 0; n < num_fiducials_y_; n++) {
      grid_points_[m][n].x = (m - origin_loc_x_) * target_spacing_;
      grid_points_[m][n].y = (n - origin_loc_y_) * target_spacing_;
      grid_points_[m][n].z = 0;
    }
  }
  //initialize the image point array
  //initialize the temporary (0,0) image points
  Point2f zero_point;
  zero_point.x = 0;
  zero_point.y = 0;
  image_points_.resize(num_cams());
  for (size_t i_cam = 0; i_cam < num_cams(); i_cam++) {
    image_points_[i_cam].resize(num_images());
    for (size_t i_image = 0; i_image < num_images(); i_image++) {
      image_points_[i_cam][i_image].resize(num_fiducials_x_);
      for (int_t i_xpnt = 0; i_xpnt < num_fiducials_x_; i_xpnt++) {
        image_points_[i_cam][i_image][i_xpnt].assign(num_fiducials_y_, zero_point);
      }
    }
  }

  // initialize the include set
  include_set_ = std::vector<bool>(num_images(),true);

  // initialize the image size
  Mat img = utils::read_image(image_list_[0][0].c_str());
  TEUCHOS_TEST_FOR_EXCEPTION(img.empty(),std::runtime_error,"image read failure " << image_list_[0][0]);
  image_size_ = img.size();

  // read the images that have been turned off if the list exists
  if(input_params_.isParameter(DICe::cal_disable_image_indices_)){
    Teuchos::Array<scalar_t> array = Teuchos::fromStringToArray<scalar_t>(input_params_.get<std::string>(DICe::cal_disable_image_indices_));
    for(int_t i=0;i<array.size();++i){
      TEUCHOS_TEST_FOR_EXCEPTION(array[i]<0||array[i]>=include_set_.size(),std::runtime_error,"");
      include_set_[array[i]] = false;
    }
  }

  has_intersection_points_ = false;
  // read the image intersections if they exist
  if(input_params_.isParameter(DICe::cal_image_intersections)){
    has_intersection_points_ = true;
    Teuchos::ParameterList image_points = input_params_.get<Teuchos::ParameterList>(DICe::cal_image_intersections);
    for (size_t i_cam = 0; i_cam < num_cams(); i_cam++) {
      for (size_t i_image = 0; i_image < num_images(); i_image++) {
        //read the grid file infomation
        TEUCHOS_TEST_FOR_EXCEPTION(!image_points.isSublist(image_list_[i_cam][i_image]),std::runtime_error,
          "missing image points for image " << image_list_[i_cam][i_image]);
        Teuchos::ParameterList grid_data = image_points.sublist(image_list_[i_cam][i_image]);
        //read each of the rows
        for (int_t i_row = 0; i_row < num_fiducials_y_; i_row++) {
          std::stringstream param_title;
          param_title << "ROW_" << i_row;
          TEUCHOS_TEST_FOR_EXCEPTION(!grid_data.isSublist(param_title.str()),std::runtime_error,
            "missing row " << i_row << " image point information for image " <<  image_list_[i_cam][i_image]);
          Teuchos::ParameterList row_data = grid_data.sublist(param_title.str());
          TEUCHOS_TEST_FOR_EXCEPTION(!row_data.isParameter("X") || !row_data.isParameter("Y"),std::runtime_error,
            "missing X and Y data for row " << i_row << " image " << image_list_[i_cam][i_image]);
          Teuchos::Array<scalar_t> tArrayX = Teuchos::fromStringToArray<scalar_t>(row_data.get<std::string>("X"));
          Teuchos::Array<scalar_t> tArrayY = Teuchos::fromStringToArray<scalar_t>(row_data.get<std::string>("Y"));
          TEUCHOS_TEST_FOR_EXCEPTION(tArrayX.size()!=num_fiducials_x_ || tArrayY.size()!=num_fiducials_x_,std::runtime_error,
            "invalid number of image points for row " << i_row << " image " << image_list_[i_cam][i_image]);
          for (int_t i_x = 0; i_x < num_fiducials_x_; i_x++) {
            image_points_[i_cam][i_image][i_x][i_row].x = tArrayX[i_x];
            image_points_[i_cam][i_image][i_x][i_row].y = tArrayY[i_x];
          }
        }
      }
    }
  }

DEBUG_MSG("Calibration class as initialized: ");
#ifdef DICE_DEBUG_MSG
  std::cout << *this << std::endl;
#endif

}

void
Calibration::set_calibration_options(const Teuchos::ParameterList & opencv_options){
  // default calibration options
  calib_options_ = 0;
  calib_options_text_.clear();
  if(opencv_options.numParams()==0){
    calib_options_ += CALIB_USE_INTRINSIC_GUESS;
    calib_options_text_.push_back("CALIB_USE_INTRINSIC_GUESS");
    calib_options_ += CALIB_ZERO_TANGENT_DIST;
    calib_options_text_.push_back("CALIB_ZERO_TANGENT_DIST");
  }else{
    std::map<std::string,int_t> cal_options;
    cal_options["CALIB_FIX_INTRINSIC"] = CALIB_FIX_INTRINSIC;
    cal_options["CALIB_USE_INTRINSIC_GUESS"] = CALIB_USE_INTRINSIC_GUESS;
#ifdef CALIB_USE_EXTRINSIC_GUESS
    cal_options["CALIB_USE_EXTRINSIC_GUESS"] = CALIB_USE_EXTRINSIC_GUESS;
#endif
    cal_options["CALIB_FIX_PRINCIPAL_POINT"] = CALIB_FIX_PRINCIPAL_POINT;
    cal_options["CALIB_FIX_FOCAL_LENGTH"] = CALIB_FIX_FOCAL_LENGTH;
    cal_options["CALIB_FIX_ASPECT_RATIO"] = CALIB_FIX_ASPECT_RATIO;
    cal_options["CALIB_SAME_FOCAL_LENGTH"] = CALIB_SAME_FOCAL_LENGTH;
    cal_options["CALIB_ZERO_TANGENT_DIST"] = CALIB_ZERO_TANGENT_DIST;
    cal_options["CALIB_FIX_K1"] = CALIB_FIX_K1;
    cal_options["CALIB_FIX_K2"] = CALIB_FIX_K2;
    cal_options["CALIB_FIX_K3"] = CALIB_FIX_K3;
    cal_options["CALIB_FIX_K4"] = CALIB_FIX_K4;
    cal_options["CALIB_FIX_K5"] = CALIB_FIX_K5;
    cal_options["CALIB_FIX_K6"] = CALIB_FIX_K6;
    cal_options["CALIB_RATIONAL_MODEL"] = CALIB_RATIONAL_MODEL;
    cal_options["CALIB_THIN_PRISM_MODEL"] = CALIB_THIN_PRISM_MODEL;
    cal_options["CALIB_FIX_S1_S2_S3_S4"] = CALIB_FIX_S1_S2_S3_S4;
    cal_options["CALIB_TILTED_MODEL"] = CALIB_TILTED_MODEL;
    cal_options["CALIB_FIX_TAUX_TAUY"] = CALIB_FIX_TAUX_TAUY;
    for(Teuchos::ParameterList::ConstIterator it=opencv_options.begin();it!=opencv_options.end();++it){
      if(cal_options.find(it->first)!=cal_options.end() && opencv_options.get<bool>(it->first)){
        calib_options_ += cal_options.find(it->first)->second;
        calib_options_text_.push_back(it->first);
      }
      if(cal_options.find(it->first)==cal_options.end())
        std::cout << "warning: ignoring invalid calibration option: " << it->first << std::endl;
    }
  }
}

Teuchos::RCP<DICe::Camera_System>
Calibration::calibrate(const std::string & output_file,
  scalar_t & rms_error){
  DEBUG_MSG("Calibration::calibrate(): begin");
  if(!has_intersection_points_){
    std::cout << "Calibration::calibrate(): extracting the target points because they have not been initialized" << std::endl;
    extract_target_points();
  }
  TEUCHOS_TEST_FOR_EXCEPTION(!has_intersection_points_, std::runtime_error,
    "Calibration::calibrate(): calibration cannot be run until the target points are extracted");
  TEUCHOS_TEST_FOR_EXCEPTION(num_cams()!=1&&num_cams()!=2,std::runtime_error,"calibration not implemented for this number of cameras");

  //assemble the intersection and object points from the grid and image points
  assemble_intersection_object_points();

  std::cout << "Calibration::calibrate(): performing OpenCV calibration" << std::endl;

  //do the intrinsic calibration for the initial guess
  int_t adjusted_pose_index = pose_estimation_index_;
  Mat cameraMatrix[2], distCoeffs[2];
  cameraMatrix[0] = initCameraMatrix2D(object_points_, intersection_points_[0], image_size_, 0);
  if(num_cams()==2)
    cameraMatrix[1] = initCameraMatrix2D(object_points_, intersection_points_[1], image_size_, 0);
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(pose_estimation_index_<0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(pose_estimation_index_>=(int_t)num_images(),std::runtime_error,"error, invalid pose estimation index");
    for(int_t i=0;i<pose_estimation_index_;++i){
      if(!include_set_[i])
        adjusted_pose_index--;
    }
    DEBUG_MSG("Calibration::calibrate(): pose estimation index: " << pose_estimation_index_);
  }
  //do the openCV calibration
  Mat R, T, E, F;
  std::vector< Mat > rvecs, tvecs;

  double rms = 0.0;
  if(num_cams()==1){
    rms = calibrateCamera(object_points_, intersection_points_[0],
      image_size_, cameraMatrix[0], distCoeffs[0],
      rvecs,tvecs,calib_options_,
      TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 1000, 1e-7));
      assert(adjusted_pose_index>=0&&adjusted_pose_index<(int_t)tvecs.size());
      assert(adjusted_pose_index>=0&&adjusted_pose_index<(int_t)rvecs.size());
      Rodrigues(rvecs[adjusted_pose_index],R); // convert Rodrigues angles to R matrix
      T = tvecs[adjusted_pose_index];
  }else{
    assert(num_cams()==2);
    rms = stereoCalibrate(object_points_, intersection_points_[0], intersection_points_[1],
      cameraMatrix[0], distCoeffs[0],
      cameraMatrix[1], distCoeffs[1],
      image_size_, R, T, E, F,
      calib_options_,
      TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 1000, 1e-7));

    // CALIBRATION QUALITY CHECK
    // because the output fundamental matrix implicitly
    // includes all the output information,
    // we can check the quality of calibration using the
    // epipolar geometry constraint: m2^t*F*m1=0
    std::fstream epipolar_file("cal_errors.txt", std::ios_base::out);

    double err = 0;
    int npoints = 0;
    std::vector<Vec3f> lines[2];
    size_t idx = 0;
    for(size_t i = 0; i < num_images(); i++ ){
      if(!include_set_[i]){
        std::cout << "Calibration::calibrate(): image set "<< i <<" skipped" << std::endl;
        epipolar_file << "skipped" << std::endl;
        continue;
      }
      assert(idx<=intersection_points_[0].size());
      //for(size_t i = 0; i < intersection_points_[0].size(); i++ )
      int npt = (int)intersection_points_[0][idx].size();
      assert(npt!=0);
      Mat imgpt[2];
      for( size_t k = 0; k < 2; k++ )
      {
        if(k==0)
          imgpt[k] = Mat(intersection_points_[0][idx]);
        else
          imgpt[k] = Mat(intersection_points_[1][idx]);
        undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
        computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
      }
      double imgErr = 0.0;
      for(int j = 0; j < npt; j++ )
      {
        double errij = fabs(intersection_points_[0][idx][j].x*lines[1][j][0] +
          intersection_points_[0][idx][j].y*lines[1][j][1] + lines[1][j][2]) +
              fabs(intersection_points_[1][idx][j].x*lines[0][j][0] +
                intersection_points_[1][idx][j].y*lines[0][j][1] + lines[0][j][2]);
        err += errij;
        imgErr += errij;
      }
      double epipolar = imgErr/npt;
      std::cout << "Calibration::calibrate(): image set "<< i <<" epipolar error: " << epipolar << std::endl;
      epipolar_file << epipolar << std::endl;
      npoints += npt;
      idx++;
    }
    epipolar_file.close();
    assert(npoints!=0);
    std::cout << "Calibration::calibrate(): average epipolar error: " <<  err/npoints << std::endl;
  }

  Teuchos::RCP<DICe::Camera_System> camera_system = Teuchos::rcp(new DICe::Camera_System());

  for (size_t i_cam = 0; i_cam < num_cams(); i_cam++) {

    DICe::Camera::Camera_Info camera_info;

    camera_info.image_height_ = image_size_.height;
    camera_info.image_width_ = image_size_.width;

    //assign the intrinsic and extrinsic values for the first camera
    camera_info.intrinsics_[Camera::CX] = cameraMatrix[i_cam].at<double>(0, 2);
    camera_info.intrinsics_[Camera::CY] = cameraMatrix[i_cam].at<double>(1, 2);
    camera_info.intrinsics_[Camera::FX] = cameraMatrix[i_cam].at<double>(0, 0);
    camera_info.intrinsics_[Camera::FY] = cameraMatrix[i_cam].at<double>(1, 1);
    camera_info.intrinsics_[Camera::K1] = distCoeffs[i_cam].at<double>(0);
    camera_info.intrinsics_[Camera::K2] = distCoeffs[i_cam].at<double>(1);
    camera_info.intrinsics_[Camera::P1] = distCoeffs[i_cam].at<double>(2);
    camera_info.intrinsics_[Camera::P2] = distCoeffs[i_cam].at<double>(3);
    if (distCoeffs[0].cols > 4) camera_info.intrinsics_[Camera::K3] = distCoeffs[i_cam].at<double>(4);
    if (distCoeffs[0].cols > 5) camera_info.intrinsics_[Camera::K4] = distCoeffs[i_cam].at<double>(5);
    if (distCoeffs[0].cols > 6) camera_info.intrinsics_[Camera::K5] = distCoeffs[i_cam].at<double>(6);
    if (distCoeffs[0].cols > 7) camera_info.intrinsics_[Camera::K6] = distCoeffs[i_cam].at<double>(7);
    if (distCoeffs[0].cols > 8) camera_info.intrinsics_[Camera::S1] = distCoeffs[i_cam].at<double>(8);
    if (distCoeffs[0].cols > 9) camera_info.intrinsics_[Camera::S2] = distCoeffs[i_cam].at<double>(9);
    if (distCoeffs[0].cols > 10) camera_info.intrinsics_[Camera::S3] = distCoeffs[i_cam].at<double>(10);
    if (distCoeffs[0].cols > 11) camera_info.intrinsics_[Camera::S4] = distCoeffs[i_cam].at<double>(11);
    if (distCoeffs[0].cols > 12) camera_info.intrinsics_[Camera::T1] = distCoeffs[i_cam].at<double>(12);
    if (distCoeffs[0].cols > 13) camera_info.intrinsics_[Camera::T2] = distCoeffs[i_cam].at<double>(13);
    camera_info.lens_distortion_model_ = Camera::OPENCV_LENS_DISTORTION;

    if((i_cam==1&&num_cams()==2)||(i_cam==0&&num_cams()==1)){
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

  camera_system->set_system_type(Camera_System::OPENCV);
  if(!output_file.empty())
    camera_system->write_camera_system_file(output_file);
  std::cout << "\nRMS error: " << rms << "\n" << std::endl;
  rms_error = rms;
  DEBUG_MSG("Calibration::calibrate(): end");
  return camera_system;
}

//extract the intersection/dot locations from a calibration target
void
Calibration::extract_target_points(){
  if(has_intersection_points_){
    DEBUG_MSG("Calibration::extract_target_points(): has_intersection_points_ is true, aborting function");
    return;
  }
  std::cout << "Calibration::extract_target_points(): target type is " << to_string(target_type_) << std::endl;
  //call the appropriate routine depending on the type of target
  switch (target_type_) {
    case CHECKER_BOARD:
      DEBUG_MSG("Calibration::extract_taget_points(): extracting checkerboard intersections");
      extract_checkerboard_intersections();
      break;
    case BLACK_ON_WHITE_W_DONUT_DOTS:
      DEBUG_MSG("Calibration::extract_taget_points(): extracting black dots on white background");
      DEBUG_MSG("Calibration::extract_taget_points(): (using donut dots to specify axes)");
      extract_dot_target_points();
      break;
    case WHITE_ON_BLACK_W_DONUT_DOTS:
      DEBUG_MSG("Calibration::extract_taget_points(): extracting white dots on black background");
      DEBUG_MSG("Calibration::extract_taget_points(): (using donut dots to specify axes)");
      extract_dot_target_points();
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Invalid target_type");
      break;
  }
  has_intersection_points_ = true;
}

//extract the intersection locations from a checkerboard pattern
void
Calibration::extract_checkerboard_intersections(){
  std::vector<Point2f> corners; //found corner locations
  std::cout << "Calibration::extract_checkerboard_intersections(): extracting intersections" << std::endl;
  for (size_t i_image = 0; i_image < num_images(); i_image++) {
    for (size_t i_cam = 0; i_cam < num_cams(); i_cam++) {
      //put together the file name
      const std::string & filename = image_list_[i_cam][i_image];
      std::cout << "Calibration::extract_checkerboard_intersections(): processing checkerboard cal image: " << filename << std::endl;
      if(include_set_[i_image] == false){
        std::cout << "Calibration::extract_checkerboard_intersections(): skipping due to image being deactivated" << std::endl;
        continue;
      }
      //read the image
      Mat img = utils::read_image(filename.c_str());
      if (img.empty()) {
        //if the image is empth mark the set as not used an move on
        std::cout << "*** warning: image is empty or not found, excluding " << std::endl;
        include_set_[i_image] = false;
        continue;
      }
      //the image was found save the size for the calibration
      TEUCHOS_TEST_FOR_EXCEPTION(img.size()!=image_size_,std::runtime_error,"");
      const int_t error_code = opencv_checkerboard_targets(img,input_params_,corners);
      // draw a debugging image if requested
      if (draw_intersection_image_){
        std::stringstream out_file_name;
        if(!debug_folder_.empty())
          out_file_name << debug_folder_;
        //out_file_name << file_name_no_dir_or_extension(image_list_[i_cam][i_image]);
        if(i_cam==0) out_file_name << ".cal_left.png";
        else out_file_name << ".cal_right.png";
        // copy the image:
        Mat debug_img = img.clone();
        std::stringstream banner;
        banner << image_list_[i_cam][i_image];
        cv::putText(debug_img, banner.str(), Point(30,30),
          FONT_HERSHEY_DUPLEX, 0.7, Scalar(255,255,255), 1, cv::LINE_AA);
        DEBUG_MSG("writing intersections image: " << out_file_name.str());
        imwrite(out_file_name.str(), debug_img);
      }
      if(error_code!=0){
        //remove the image from the calibration and proceed with the next image
        include_set_[i_image] = false;
        std::cout << "*** warning: checkerboard intersections were not found, excluding image" << std::endl;
        continue;
      }
      int_t i_pnt = 0;
      for (int_t i_y = 0; i_y < num_fiducials_y_; i_y++) {
        for (int_t i_x = 0; i_x < num_fiducials_x_; i_x++) {
          image_points_[i_cam][i_image][i_x][num_fiducials_y_ - 1 - i_y] = corners[i_pnt];
          i_pnt++;
        }
      } // end loop over fiducials
    } // end cam loop
  } // end image loop
}

//assembles the image/grid points into the object/intersection points for calibration
void
Calibration::assemble_intersection_object_points() {
  DEBUG_MSG("Calibration::assemble_intersection_object_points(): begin");

  object_points_.clear();
  intersection_points_.clear();
  intersection_points_.resize(num_cams());

  size_t num_included_sets = 0;
  //step through by image set
  for (size_t i_image = 0; i_image < num_images(); i_image++) {
    //if the image set is included in the calibration
    //the function assumes that the include flag is set properly
    if (include_set_[i_image]) {
      object_points_.push_back(std::vector<cv::Point3f>());
      for (size_t i_cam = 0; i_cam < num_cams(); i_cam++)
        intersection_points_[i_cam].push_back(std::vector<cv::Point2f>());
      int_t num_common = 0;
      //find common points in all cameras
      for (int_t i_x = 0; i_x < num_fiducials_x_; i_x++) {
        for (int_t i_y = 0; i_y < num_fiducials_y_; i_y++) {
          bool common_pt = true;
          for (size_t i_cam = 0; i_cam < num_cams(); i_cam++) {
            //do all the cameras have a value for this point?
            if (image_points_[i_cam][i_image][i_x][i_y].x <= 0 || image_points_[i_cam][i_image][i_x][i_y].y <= 0) {
              common_pt = false;
              break;
            }
          }
          if (common_pt) { //save into the image points and objective points for this image set
            num_common++;
            //fill the intersection and object points for the calibration
            for (size_t i_cam = 0; i_cam < num_cams(); i_cam++) {
              intersection_points_[i_cam][num_included_sets].push_back(image_points_[i_cam][i_image][i_x][i_y]);
            }
            object_points_[num_included_sets].push_back(grid_points_[i_x][i_y]);
          }
        }
      }
      num_included_sets++;
      //DEBUG_MSG("image set: " << i_image << " common points: " << num_common);
    }
  }
  DEBUG_MSG("Calibration::assemble_intersection_object_points(): end");
}

//extract the dot locations from a dot target
void
Calibration::extract_dot_target_points(){
  std::cout << "Calibration::extract_dot_target_points(): extracting dots" << std::endl;
  const int_t orig_thresh_start = input_params_.get<int_t>(opencv_server_threshold_start,20);
  const int_t orig_thresh_end =   input_params_.get<int_t>(opencv_server_threshold_end,250);
  const int_t orig_thresh_step =  input_params_.get<int_t>(opencv_server_threshold_step,5);
  scalar_t include_image_set_tol = 0.75; //the search must have found at least 75% of the total to be included
  for (size_t i_image = 0; i_image < num_images(); i_image++){
    //go through each of the camera's images (note only two camera calibration is currently supported)
    for (size_t i_cam = 0; i_cam < num_cams(); i_cam++){
      std::cout << "Calibration::extract_dot_target_points(): processing cal image: " << image_list_[i_cam][i_image] << std::endl;
      //DEBUG_MSG("processing cal image: " << image_list_[i_cam][i_image]);
      if(include_set_[i_image] == false){
        std::cout << "Calibration::extract_dot_target_points(): skipping due to image being deactivated" << std::endl;
        continue;
      }
      Mat img = utils::read_image(image_list_[i_cam][i_image].c_str());
      if (img.empty()) {
        std::cout << "*** warning: image is empty or not found, excluding this image" << std::endl;
        include_set_[i_image] = false;
        continue;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(img.size()!=image_size_,std::runtime_error,"");
      std::vector<KeyPoint> key_points;
      std::vector<KeyPoint> img_points;
      std::vector<KeyPoint> grd_points;
      int_t return_thresh = orig_thresh_start;
      int_t error_code = opencv_dot_targets(img, input_params_,
        key_points,img_points,grd_points,return_thresh);
      // check if extraction failed
      if(error_code==0){
        input_params_.set(opencv_server_threshold_start,return_thresh);
        input_params_.set(opencv_server_threshold_end,return_thresh);
        input_params_.set(opencv_server_threshold_step,return_thresh);
      }else if(i_image!=0&&error_code==1){
        input_params_.set(opencv_server_threshold_start,orig_thresh_start);
        input_params_.set(opencv_server_threshold_end,orig_thresh_end);
        input_params_.set(opencv_server_threshold_step,orig_thresh_step);
        error_code = opencv_dot_targets(img, input_params_,
          key_points,img_points,grd_points,return_thresh);
        if(error_code==0){
          input_params_.set(opencv_server_threshold_start,return_thresh);
          input_params_.set(opencv_server_threshold_end,return_thresh);
          input_params_.set(opencv_server_threshold_step,return_thresh);
        }
      }

      // draw a debugging image if requested
      if (draw_intersection_image_){
        std::stringstream out_file_name;
        if(!debug_folder_.empty())
          out_file_name << debug_folder_;
        //out_file_name << file_name_no_dir_or_extension(image_list_[i_cam][i_image]);
        if(i_cam==0) out_file_name << ".cal_left.png";
        else out_file_name << ".cal_right.png";
        DEBUG_MSG("writing intersections image: " << out_file_name.str());
        // copy the image:
        Mat debug_img = img.clone();
        std::stringstream banner;
        banner << image_list_[i_cam][i_image];
        cv::putText(debug_img, banner.str(), Point(30,30),
          FONT_HERSHEY_DUPLEX, 0.7, Scalar(255,255,255), 1, cv::LINE_AA);
        imwrite(out_file_name.str(), debug_img);
      }
      if(error_code!=0){
        std::cout << "*** warning: " << image_list_[i_cam][i_image] << " failed dot extraction with error code: " << error_code << std::endl;
        include_set_[i_image] = false;
        continue;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(i_cam>=image_points_.size(),
        std::runtime_error,"cal target parameters likely incorrect, invalid camera number");
      TEUCHOS_TEST_FOR_EXCEPTION(i_image>=image_points_[i_cam].size(),
        std::runtime_error,"cal target parameters likely incorrect, invalid image number");
      TEUCHOS_TEST_FOR_EXCEPTION(origin_loc_x_>=(int_t)image_points_[i_cam][i_image].size(),
        std::runtime_error,"cal target parameters likely incorrect, origin is off the grid");
      TEUCHOS_TEST_FOR_EXCEPTION(origin_loc_x_ + num_fiducials_origin_to_x_marker_-1>=(int_t)image_points_[i_cam][i_image].size(),
        std::runtime_error,"cal target parameters likely incorrect, x axis marker is off the grid");
      TEUCHOS_TEST_FOR_EXCEPTION(origin_loc_y_>=(int_t)image_points_[i_cam][i_image][origin_loc_x_+num_fiducials_origin_to_x_marker_-1].size(),
        std::runtime_error,"cal target parameters likely incorrect, origin is off the grid");
      TEUCHOS_TEST_FOR_EXCEPTION(origin_loc_y_+num_fiducials_origin_to_y_marker_-1>=(int_t)image_points_[i_cam][i_image][origin_loc_x_].size(),
        std::runtime_error,"cal target parameters likely incorrect, y axis marker is off the grid");

      //add the keypoints to the found positions
      assert(key_points.size()==3);
      image_points_[i_cam][i_image][origin_loc_x_][origin_loc_y_].x = key_points[0].pt.x;
      image_points_[i_cam][i_image][origin_loc_x_][origin_loc_y_].y = key_points[0].pt.y;
      image_points_[i_cam][i_image][origin_loc_x_ + num_fiducials_origin_to_x_marker_ - 1][origin_loc_y_].x = key_points[1].pt.x;
      image_points_[i_cam][i_image][origin_loc_x_ + num_fiducials_origin_to_x_marker_ - 1][origin_loc_y_].y = key_points[1].pt.y;
      image_points_[i_cam][i_image][origin_loc_x_][origin_loc_y_ + num_fiducials_origin_to_y_marker_ - 1].x = key_points[2].pt.x;
      image_points_[i_cam][i_image][origin_loc_x_][origin_loc_y_ + num_fiducials_origin_to_y_marker_ - 1].y = key_points[2].pt.y;

      //save the image points
      for (int_t n = 0; n < (int_t)img_points.size(); n++) {
        image_points_[i_cam][i_image][grd_points[n].pt.x][grd_points[n].pt.y].x = img_points[n].pt.x;
        image_points_[i_cam][i_image][grd_points[n].pt.x][grd_points[n].pt.y].y = img_points[n].pt.y;
      }

    }//end camera loop (i_cam)

    DEBUG_MSG("finding common points in all images");

    //we have gridded data for each of the cameras use any point common to all the cameras
    size_t num_common_pts = 0;
    for (int_t m = 0; m < num_fiducials_x_; m++) {
      for (int_t n = 0; n < num_fiducials_y_; n++) {
        bool common_pt = true;
        //if the values for this location are non-zero for all cameras we have a common point
        for (size_t i_cam = 0; i_cam < num_cams(); i_cam++) {
          if (image_points_[i_cam][i_image][m][n].x <= 0 || image_points_[i_cam][i_image][m][n].y <= 0)
            common_pt = false;  //do all the cameras have a value for this point?
        }
        if (common_pt) { //save into the image points and objective points for this image set
          num_common_pts++;  //increment the commont point counter for this image
        }
      }
    }
    DEBUG_MSG("numer of common dots: " << num_common_pts);
    if (num_common_pts < (num_fiducials_x_*num_fiducials_y_*include_image_set_tol) && include_set_[i_image]){
      //exclude the set
      std::cout << "*** warning: excluding this image set due to not enough dots common among all images" << std::endl;
      include_set_[i_image] = false;
    }
  }//end image loop
}//end extract_dot_target_points

//write a file with the intersection information
void
Calibration::write_calibration_file(const std::string & filename) {

  TEUCHOS_TEST_FOR_EXCEPTION(filename.find("xml") == std::string::npos,std::runtime_error,"invalid file extension (should be .xml)");
  std::cout << "Calibration::write_calibration_file(): writing calibration file: " << filename << std::endl;

  Teuchos::ParameterList output_params = input_params_;

  // remove image_set true false if it exists
  if(output_params.isParameter(DICe::cal_disable_image_indices_))
    output_params.remove(DICe::cal_disable_image_indices_);
  // write the indices of disabled images:
  std::stringstream param_val;
  param_val << "{ ";
  bool first_value = true;
  for (size_t i = 0; i < include_set_.size(); ++i){
    if(!include_set_[i]){
      if(first_value){
        param_val << i;
        first_value = false;
      }
      else
        param_val << ", " << i;
    }
  }
  param_val << " }";
  output_params.set<std::string>(DICe::cal_disable_image_indices_,param_val.str());

  Teuchos::ParameterList image_sublist;
  for (size_t i_cam = 0; i_cam < num_cams(); i_cam++) {
    for (size_t i_image = 0; i_image < num_images(); i_image++) {
      Teuchos::ParameterList image_points_sublist;
      for (int_t i_row = 0; i_row < num_fiducials_y_; i_row++) {
        Teuchos::ParameterList image_points_row_sublist;
        std::stringstream x_row;
        std::stringstream y_row;
        x_row.precision(12);
        x_row << std::scientific;
        y_row.precision(12);
        y_row << std::scientific;
        x_row << "{ ";
        y_row << "{ ";
        for (int_t i_x = 0; i_x < (num_fiducials_x_ - 1); i_x++) {
          x_row << image_points_[i_cam][i_image][i_x][i_row].x << ",  ";
          y_row << image_points_[i_cam][i_image][i_x][i_row].y << ",  ";
        }
        x_row << image_points_[i_cam][i_image][num_fiducials_x_ - 1][i_row].x << "}";
        y_row << image_points_[i_cam][i_image][num_fiducials_x_ - 1][i_row].y << "}";
        image_points_row_sublist.set<std::string>("X",x_row.str());
        image_points_row_sublist.set<std::string>("Y",y_row.str());
        std::stringstream param_title;
        param_title << "ROW_" << i_row;
        image_points_sublist.set<Teuchos::ParameterList>(param_title.str(),image_points_row_sublist);
      } // end row loop
      image_sublist.set<Teuchos::ParameterList>(image_list_[i_cam][i_image],image_points_sublist);
    } // end image loop
  } // end camera loop
  if(output_params.isParameter(DICe::cal_image_intersections))
    output_params.remove(DICe::cal_image_intersections);
  output_params.set<Teuchos::ParameterList>(DICe::cal_image_intersections,image_sublist);

  //output_params.print(std::cout);

  write_xml_file(filename,output_params);

  DEBUG_MSG("Calibration::write_calibration_file(): file write complete");
}

std::ostream & operator<<(std::ostream & os, const Calibration & cal){
  std::ios_base::fmtflags f( std::cout.flags() );
  os << "---------- Calibration ----------" << std::endl;
  os << "num cameras:             " << cal.num_cams() << std::endl;
  os << "target type:             " << Calibration::to_string(cal.target_type_) << std::endl;
  os << "num fiducials x:         " << cal.num_fiducials_x_ << std::endl;
  os << "num fiducials y:         " << cal.num_fiducials_y_ << std::endl;
  os << "origin loc x:            " << cal.origin_loc_x_ << std::endl;
  os << "origin loc y:            " << cal.origin_loc_y_ << std::endl;
  os << "num fiducials between\n"
        "the origin and x marker: " << cal.num_fiducials_origin_to_x_marker_ << std::endl;
  os << "num fiducials between\n"
        "the origin and y marker: " << cal.num_fiducials_origin_to_y_marker_ << std::endl;
  os << "draw intersection image: " << cal.draw_intersection_image_ << std::endl;
  os << "cal debug folder:        " << cal.debug_folder_ << std::endl;
  os << "image size (h x w):      " << cal.image_size_.height  << " x " << cal.image_size_.width << std::endl;
  os << "calibration options:     " << std::endl;
  for(size_t i=0;i<cal.calib_options_text_.size();++i)
    os << "  " << cal.calib_options_text_[i] << std::endl;
  os << "has intersection points: " << cal.has_intersection_points_ << std::endl;
  os.precision(4);
  os << std::scientific;
  os << "target spacing:          " << cal.target_spacing_ << std::endl;
  std::cout.flags( f );
  os << "number of images:        " << cal.num_images() << std::endl;
  for(size_t i=0;i<cal.num_cams();++i){
    os << "camera " << i << " image list:" << std::endl;
    for(size_t j=0;j<cal.num_images();++j){
      std::string active = " active";
      if(!cal.include_set_[j]) active = " deactivated";
      os << cal.image_list_[i][j] << active << std::endl;
    }
  }
  os << "grid points: " << std::endl;
  os.precision(2);
  os << std::scientific;
  os << std::setw(10);
  os << std::left;
  for (int_t n = 0; n < cal.num_fiducials_y_; n++) {
    os << "row : " << n << std::endl;
    os << "X:";
    for (int_t m = 0; m < cal.num_fiducials_x_; m++) {
      os << " " << cal.grid_points_[m][n].x;
    }
    os << std::endl << "Y:";
    for (int_t m = 0; m < cal.num_fiducials_x_; m++) {
      os << " " << cal.grid_points_[m][n].y;
    }
    os << std::endl << "Z:";
    for (int_t m = 0; m < cal.num_fiducials_x_; m++) {
      os << " " << cal.grid_points_[m][n].z;
    }
    os << std::endl;
  }
  os.precision(3);
  if(cal.has_intersection_points_){
    for (size_t i_cam = 0; i_cam < cal.num_cams(); i_cam++) {
      for (size_t i_image = 0; i_image < cal.num_images(); i_image++) {
        os << "image: " << cal.image_list_[i_cam][i_image] << std::endl;
        for (int_t i_row = 0; i_row < cal.num_fiducials_y_; i_row++) {
          os << "row : " << i_row << std::endl;
          os << "X:";
          for (int_t i_x = 0; i_x < cal.num_fiducials_x_; i_x++) {
            os << " " << cal.image_points_[i_cam][i_image][i_x][i_row].x ;
          }
          os << std::endl << "Y:";
          for (int_t i_x = 0; i_x < cal.num_fiducials_x_; i_x++) {
            os << " " << cal.image_points_[i_cam][i_image][i_x][i_row].y ;
          }
          os << std::endl;
        } // end row loop
      } // end image loop
    } // end camera loop
  } // end has intersections
  std::cout.flags( f );
  os << "---------------------------------" << std::endl;
  return os;
};

} //end of DICe namespace
