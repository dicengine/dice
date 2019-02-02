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

#include <Teuchos_XMLParameterListHelpers.hpp>

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
  DEBUG_MSG("user specified input parameters");
#ifdef DICE_DEBUG_MSG
  params->print(std::cout);
#endif
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

  // 6 transformation coefficients
  img_to_grdx_.resize(6);
  grd_to_imgx_.resize(6);
  img_to_grdy_.resize(6);
  grd_to_imgy_.resize(6);

  // initialize the image size
  Mat img = imread(image_list_[0][0], IMREAD_GRAYSCALE);
  TEUCHOS_TEST_FOR_EXCEPTION(img.empty(),std::runtime_error,"image read failure");
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
    DEBUG_MSG("Calibration::calibrate(): extracting the target points because they have not been initialized");
    extract_target_points();
  }
  TEUCHOS_TEST_FOR_EXCEPTION(!has_intersection_points_, std::runtime_error,
    "Calibration::calibrate(): calibration cannot be run until the target points are extracted");
  TEUCHOS_TEST_FOR_EXCEPTION(num_cams()!=1&&num_cams()!=2,std::runtime_error,"calibration not implemented for this number of cameras");

  //assemble the intersection and object points from the grid and image points
  assemble_intersection_object_points();

  //do the intrinsic calibration for the initial guess
  Mat cameraMatrix[2], distCoeffs[2];
  cameraMatrix[0] = initCameraMatrix2D(object_points_, intersection_points_[0], image_size_, 0);
  if(num_cams()==2)
    cameraMatrix[1] = initCameraMatrix2D(object_points_, intersection_points_[1], image_size_, 0);

  //do the openCV calibration
  Mat R, T, E, F;

  double rms = 0.0;
  if(num_cams()==1){
    rms = calibrateCamera(object_points_, intersection_points_[0],
      image_size_, cameraMatrix[0], distCoeffs[0],
      R,T,calib_options_,
      TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 1000, 1e-7));
  }else{
    rms = stereoCalibrate(object_points_, intersection_points_[0], intersection_points_[1],
      cameraMatrix[0], distCoeffs[0],
      cameraMatrix[1], distCoeffs[1],
      image_size_, R, T, E, F,
      calib_options_,
      TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 1000, 1e-7));
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

  camera_system->set_system_type(Camera_System::OPENCV);
  if(!output_file.empty())
    camera_system->write_camera_system_file(output_file);
  DEBUG_MSG("Calibration::calibrate(): RMS value: " << rms);
  rms_error = rms;
  DEBUG_MSG("Calibration::calibrate(): end");
  return camera_system;
}

//extract the intersection/dot locations from a calibration target
void
Calibration::extract_target_points(const int_t threshold_start,
  const int_t threshold_end,
  const int_t threshold_step){
  if(has_intersection_points_){
    DEBUG_MSG("Calibration::extract_target_points(): has_intersection_points_ is true, aborting function");
    return;
  }
  //call the appropriate routine depending on the type of target
  switch (target_type_) {
    case CHECKER_BOARD:
      DEBUG_MSG("Calibration::extract_taget_points(): extracting checkerboard intersections");
      extract_checkerboard_intersections();
      break;
    case BLACK_ON_WHITE_W_DONUT_DOTS:
      DEBUG_MSG("Calibration::extract_taget_points(): extracting black dots on white background");
      DEBUG_MSG("Calibration::extract_taget_points(): (using donut dots to specify axes)");
      extract_dot_target_points(threshold_start, threshold_end, threshold_step);
      break;
    case WHITE_ON_BLACK_W_DONUT_DOTS:
      DEBUG_MSG("Calibration::extract_taget_points(): extracting white dots on black background");
      DEBUG_MSG("Calibration::extract_taget_points(): (using donut dots to specify axes)");
      extract_dot_target_points(threshold_start, threshold_end, threshold_step);
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
  Mat out_img;
  bool found;
  Size boardSize; //used by openCV
  std::vector<Point2f> corners; //found corner locations

  std::fill(include_set_.begin(),include_set_.end(),1);

  //set the height and width of the board in intersections
  boardSize.width = num_fiducials_x_;
  boardSize.height = num_fiducials_y_;

  for (size_t i_image = 0; i_image < num_images(); i_image++) {
    for (size_t i_cam = 0; i_cam < num_cams(); i_cam++) {

      //put together the file name
      const std::string & filename = image_list_[i_cam][i_image];
      DEBUG_MSG("processing checkerboard cal image: " << filename);

      //read the image
      Mat img = imread(filename, IMREAD_GRAYSCALE);
      if (img.empty()) {
        //if the image is empth mark the set as not used an move on
        DEBUG_MSG("warning: image is empty or not found, excluding ");
        include_set_[i_image] = false;
      }
      else {
        //the image was found save the size for the calibration
        TEUCHOS_TEST_FOR_EXCEPTION(img.size()!=image_size_,std::runtime_error,"");
        //copy the image if we are going to save intersection images
        if (draw_intersection_image_) cvtColor(img, out_img, cv::COLOR_GRAY2RGB);
        //find the intersections in the checkerboard
        found = findChessboardCorners(img, boardSize, corners,
          CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
          DEBUG_MSG("checkerboard intersections found");
          //improve the locations with cornerSubPixel
          //need to check out the window and zero zone parameters to see their effect
          cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
            TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
              30, 0.01));
          //store the points and draw the intersection markers if needed
          int_t i_pnt = 0;
          for (int_t i_y = 0; i_y < num_fiducials_y_; i_y++) {
            for (int_t i_x = 0; i_x < num_fiducials_x_; i_x++) {
              image_points_[i_cam][i_image][i_x][num_fiducials_y_ - 1 - i_y] = corners[i_pnt];
              if (draw_intersection_image_) {
                circle(out_img, corners[i_pnt], 3, Scalar(0, 255, 0));
              }
              i_pnt++;
            }
          }
        } // end found
        else {
          //remove the image from the calibration and proceed with the next image
          include_set_[i_image] = false;
          DEBUG_MSG("warning: checkerboard intersections were not found, excluding image");
        }
        //save the intersection point image if needed
        if (draw_intersection_image_){
          std::stringstream out_file_name;
          if(!debug_folder_.empty())
            out_file_name << debug_folder_;
          out_file_name << file_name_no_dir_or_extension(image_list_[i_cam][i_image]);
          out_file_name << "_markers.png";
          DEBUG_MSG("writing intersections image: " << out_file_name.str());
          imwrite(out_file_name.str(), out_img);
        }
      }
    }
  }
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
Calibration::extract_dot_target_points(const int_t threshold_start,
  const int_t threshold_end,
  const int_t threshold_step){

  scalar_t include_image_set_tol = 0.75; //the search must have found at least 75% of the total to be included
  float dot_tol = 0.25; //how close a dot must be to the projected loation to be considered good
  Mat out_img; //output image

  //integer grid coordinates of the marker locations
  std::vector<KeyPoint> marker_grid_locs;
  marker_grid_locs.resize(3);
  marker_grid_locs[0].pt.x = origin_loc_x_;
  marker_grid_locs[0].pt.y = origin_loc_y_;
  marker_grid_locs[1].pt.x = origin_loc_x_ + num_fiducials_origin_to_x_marker_ - 1;
  marker_grid_locs[1].pt.y = origin_loc_y_;
  marker_grid_locs[2].pt.x = origin_loc_x_;
  marker_grid_locs[2].pt.y = origin_loc_y_ + num_fiducials_origin_to_y_marker_ - 1;

  //initialize the include set array
  std::fill(include_set_.begin(),include_set_.end(),1);

  for (size_t i_image = 0; i_image < num_images(); i_image++){
    //go through each of the camera's images (note only two camera calibration is currently supported)
    for (size_t i_cam = 0; i_cam < num_cams(); i_cam++){
      DEBUG_MSG("processing cal image: " << image_list_[i_cam][i_image]);
      Mat img = imread(image_list_[i_cam][i_image], IMREAD_GRAYSCALE);
      if (img.empty()) {
        DEBUG_MSG("warning: image is empty or not found, excluding this image");
        include_set_[i_image] = false;
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(img.size()!=image_size_,std::runtime_error,"");
        //find the keypoints in the image
        bool keypoints_found = true;
        std::vector<KeyPoint> keypoints;
        DEBUG_MSG("extracting the key points");
        //try to find the keypoints at different thresholds
        int_t i_thresh_first = 0;
        int_t i_thresh_last = 0;
        int_t i_thresh = 0;
        for (i_thresh = threshold_start; i_thresh <= threshold_end; i_thresh += threshold_step) {
          //get the dots using an inverted image to get the donut holes
          get_dot_markers(img, keypoints, i_thresh, true);
          // were three keypoints found?
          if (keypoints.size() != 3) {
            keypoints_found = false;
            if (i_thresh_first != 0) {
              keypoints_found = true; //keypoints found in a previous pass
              break; // get out of threshold loop
            }
          }
          else
          {
            //save the threshold value if needed
            if (i_thresh_first == 0) {
              i_thresh_first = i_thresh;
            }
            i_thresh_last = i_thresh;
            keypoints_found = true;
          }
        } // end marker (donut) dot threshholding loop

        //calculate the average threshold value
        i_thresh = (i_thresh_first + i_thresh_last) / 2;
        //get the key points at the average threshold value
        get_dot_markers(img, keypoints, i_thresh, true);
        //it is possible that this threshold does not have 3 points.
        //chances are that this indicates p thresholding problem to begin with
        //use the
        if (keypoints.size() != 3) {
          DEBUG_MSG("warning: unable to identify three keypoints");
          DEBUG_MSG("         other points will not be extracted");
          keypoints_found = false;
        }

        //now that we have the keypoints try to get the rest of the dots
        if (keypoints_found) { //three keypoints were found

          //reorder the keypoints into an origin, xaxis, yaxis order
          reorder_keypoints(keypoints);

          //report the results
          DEBUG_MSG("    threshold: " << i_thresh);
          DEBUG_MSG("    ordered keypoints: ");
          for (size_t i = 0; i < keypoints.size(); ++i) //save and display the keypoints
            DEBUG_MSG("      keypoint: " << keypoints[i].pt.x << " " << keypoints[i].pt.y);

          TEUCHOS_TEST_FOR_EXCEPTION(i_cam>=image_points_.size(),
            std::runtime_error,"cal target parameters likely incorrect");
          TEUCHOS_TEST_FOR_EXCEPTION(i_image>=image_points_[i_cam].size(),
            std::runtime_error,"cal target parameters likely incorrect");
          TEUCHOS_TEST_FOR_EXCEPTION(origin_loc_x_>=(int_t)image_points_[i_cam][i_image].size(),
            std::runtime_error,"cal target parameters likely incorrect");
          TEUCHOS_TEST_FOR_EXCEPTION(origin_loc_x_ + num_fiducials_origin_to_x_marker_-1>=(int_t)image_points_[i_cam][i_image].size(),
            std::runtime_error,"cal target parameters likely incorrect");
          TEUCHOS_TEST_FOR_EXCEPTION(origin_loc_y_>=(int_t)image_points_[i_cam][i_image][origin_loc_x_+num_fiducials_origin_to_x_marker_-1].size(),
            std::runtime_error,"cal target parameters likely incorrect");
          TEUCHOS_TEST_FOR_EXCEPTION(origin_loc_x_+num_fiducials_origin_to_x_marker_-1>=(int_t)image_points_[i_cam][i_image][origin_loc_x_].size(),
            std::runtime_error,"cal target parameters likely incorrect");

          //add the keypoints to the found positions
          assert(keypoints.size()==3);
          image_points_[i_cam][i_image][origin_loc_x_][origin_loc_y_].x = keypoints[0].pt.x;
          image_points_[i_cam][i_image][origin_loc_x_][origin_loc_y_].y = keypoints[0].pt.y;
          image_points_[i_cam][i_image][origin_loc_x_ + num_fiducials_origin_to_x_marker_ - 1][origin_loc_y_].x = keypoints[1].pt.x;
          image_points_[i_cam][i_image][origin_loc_x_ + num_fiducials_origin_to_x_marker_ - 1][origin_loc_y_].y = keypoints[1].pt.y;
          image_points_[i_cam][i_image][origin_loc_x_][origin_loc_y_ + num_fiducials_origin_to_y_marker_ - 1].x = keypoints[2].pt.x;
          image_points_[i_cam][i_image][origin_loc_x_][origin_loc_y_ + num_fiducials_origin_to_y_marker_ - 1].y = keypoints[2].pt.y;

          //if drawing an output image mark the keypoints with a alternate pattern
          if (draw_intersection_image_) { //if creating an output image
            Point cvpoint;
            //copy the image into the output image
            cvtColor(img, out_img, cv::COLOR_GRAY2RGB);
            for (int_t n = 0; n < 3; n++) {
              cvpoint.x = keypoints[n].pt.x;
              cvpoint.y = keypoints[n].pt.y;
              circle(out_img, cvpoint, 20, Scalar(0, 255, 255), 4);
            }
          }

          //from the keypoints calculate the image to grid and grid to image transforms (no keystoning)
          calc_trans_coeff(keypoints, marker_grid_locs);

          //determine a threshold from the gray levels between the keypoints
          int_t xstart, xend, ystart, yend;
          float maxgray, mingray;
          xstart = keypoints[0].pt.x;
          xend = keypoints[1].pt.x;
          if (xend < xstart) {
            xstart = keypoints[1].pt.x;
            xend = keypoints[0].pt.x;
          }
          ystart = keypoints[0].pt.y;
          yend = keypoints[1].pt.y;
          if (yend < ystart) {
            ystart = keypoints[1].pt.y;
            yend = keypoints[0].pt.y;
          }

          maxgray = img.at<uchar>(ystart, xstart);
          mingray = maxgray;
          int_t curgray;
          for (int_t ix = xstart; ix <= xend; ix++) {
            for (int_t iy = ystart; iy <= yend; iy++) {
              curgray = img.at<uchar>(iy, ix);
              if (maxgray < curgray) maxgray = curgray;
              if (mingray > curgray) mingray = curgray;
            }
          }
          i_thresh = (maxgray + mingray) / 2;
          DEBUG_MSG("  getting the rest of the dots");
          DEBUG_MSG("    threshold to get dots: " << i_thresh);

          //get the rest of the dots
          std::vector<KeyPoint> dots;
          get_dot_markers(img, dots, i_thresh, false);
          DEBUG_MSG("    prospective grid points found: " << dots.size());

          //filter the dots
          std::vector<KeyPoint> img_points;
          std::vector<KeyPoint> grd_points;
          // filter dots based on avg size and whether the dots fall in the central box
          filter_dot_markers(i_cam, i_image, dots, img_points, grd_points, dot_tol, out_img, false);

          //initialize the process variables
          int_t filter_passes = 1;
          int_t old_dot_num = 3;
          int_t new_dot_num = img_points.size();
          int_t max_dots = num_fiducials_x_ * num_fiducials_x_ - 3;

          //if the number of dots has not changed
          while ((old_dot_num != new_dot_num && new_dot_num != max_dots && filter_passes < 20) || filter_passes < 3) {
            //update the old dot count
            old_dot_num = new_dot_num;
            //from the good points that were found improve the mapping parameters
            calc_trans_coeff(img_points, grd_points);
            // filter dots based on avg size and whether the dots fall in the central box with the new parameters
            // the transformation now includes keystoning
            filter_dot_markers(i_cam, i_image, dots, img_points, grd_points, dot_tol, out_img, false);
            filter_passes++;
            new_dot_num = img_points.size();
          }

          //if drawing the images run filter one more time and draw the intersection locations
          if (draw_intersection_image_)
            filter_dot_markers(i_cam, i_image, dots, img_points, grd_points, dot_tol, out_img, true);

          //save the information about the found dots
          DEBUG_MSG("    good dots identified: " << new_dot_num);
          DEBUG_MSG("    filter passes: " << filter_passes);

          //save the image points
          for (int_t n = 0; n < (int_t)img_points.size(); n++) {
            image_points_[i_cam][i_image][grd_points[n].pt.x][grd_points[n].pt.y].x = img_points[n].pt.x;
            image_points_[i_cam][i_image][grd_points[n].pt.x][grd_points[n].pt.y].y = img_points[n].pt.y;
          }

          //write the intersection image file if requested
          if (draw_intersection_image_){
            std::stringstream out_file_name;
            if(!debug_folder_.empty())
              out_file_name << debug_folder_;
            out_file_name << file_name_no_dir_or_extension(image_list_[i_cam][i_image]);
            out_file_name << "_markers.png";
            DEBUG_MSG("writing found dots image: " << out_file_name.str());
            imwrite(out_file_name.str(), out_img);
          }
        }//end if keypoints_found
      }//end if not empty image
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
    if (num_common_pts < (num_fiducials_x_*num_fiducials_y_*include_image_set_tol)){
      //exclude the set
      DEBUG_MSG("warning: excluding this image set due to not enough dots common among all images");
      include_set_[i_image] = false;
    }
  }//end image loop
}//end extract_dot_target_points

//get all the possible dot markers
void
Calibration::get_dot_markers(cv::Mat img,
  std::vector<KeyPoint> & keypoints,
  int_t thresh,
  bool invert) {
  DEBUG_MSG("Calibration::get_dot_markers(): thresh " << thresh << " invert " << invert);
  // setup the blob detector
  SimpleBlobDetector::Params params;
  params.maxArea = 10e4;
  params.minArea = 100;
  cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

  //clear the points vector
  keypoints.clear();

  //create a temporary image
  Mat timg;
  img.copyTo(timg);
  // setup the binary image
  Mat bi_src(timg.size(), timg.type());
  // apply thresholding
  threshold(timg, bi_src, thresh, 255, cv::THRESH_BINARY);
  // invert the source image
  Mat not_src(bi_src.size(), bi_src.type());
  bitwise_not(bi_src, not_src);

  //detect dots on the appropriately inverted image
  if (target_type_ == BLACK_ON_WHITE_W_DONUT_DOTS) {
    if (!invert) detector->detect(bi_src, keypoints);
    if (invert) detector->detect(not_src, keypoints);
  }
  else {
    if (invert) detector->detect(bi_src, keypoints);
    if (!invert) detector->detect(not_src, keypoints);
  }
  DEBUG_MSG("Calibration::get_dot_markers(): num keypoints " << keypoints.size());
}

//reorder the keypoints into origin, xaxis, yaxis order
void
Calibration::reorder_keypoints(std::vector<KeyPoint> & keypoints) {
  std::vector<float> dist(3, 0.0); //holds the distances between the points
  std::vector<KeyPoint> temp_points;
  std::vector<int> dist_order(3, 0); //index order of the distances max to min
  float cross; //cross product and indicies
  temp_points.clear();
  //save the distances between the points (note if dist(1,2) is max point 0 is the origin)
  dist[0] = dist2(keypoints[1], keypoints[2]);
  dist[1] = dist2(keypoints[0], keypoints[2]);
  dist[2] = dist2(keypoints[0], keypoints[1]);
  //order the distances
  order_dist3(dist, dist_order);

  //calaulate the cross product to determine the x and y axis
  int_t io = dist_order[0];
  int_t i1 = dist_order[1];
  int_t i2 = dist_order[2];

  //if the cross product is positive i1 was the y axis point because of inverted image coordinates
  cross = ((keypoints[i1].pt.x - keypoints[io].pt.x) * (keypoints[i2].pt.y - keypoints[io].pt.y)) -
      ((keypoints[i1].pt.y - keypoints[io].pt.y) * (keypoints[i2].pt.x - keypoints[io].pt.x));
  if (cross > 0.0) { //i2 is the x axis
    i2 = dist_order[1];
    i1 = dist_order[2];
  }
  //reorder the points and return
  temp_points.push_back(keypoints[io]);
  temp_points.push_back(keypoints[i1]);
  temp_points.push_back(keypoints[i2]);
  keypoints[0] = temp_points[0];
  keypoints[1] = temp_points[1];
  keypoints[2] = temp_points[2];
}

//calculate the transformation coefficients
void
Calibration::calc_trans_coeff(std::vector<cv::KeyPoint> & imgpoints,
  std::vector<cv::KeyPoint> & grdpoints) {

  //DEBUG_MSG("Calibration::calc_trans_coeff(): begin");
  //if only three points are given the six parameter mapping will not include keystoning
  if (imgpoints.size() == 3) {
    Mat A = Mat_<double>(3, 3);
    Mat Ax = Mat_<double>(3, 1);
    Mat Ay = Mat_<double>(3, 1);
    Mat coeff_x = Mat_<double>(3, 1);
    Mat coeff_y = Mat_<double>(3, 1);

    //image to grid transform (not overdetermined so just solve the matrix equation)
    for (int_t i = 0; i < 3; i++) {
      A.at<double>(i, 0) = 1.0;
      A.at<double>(i, 1) = imgpoints[i].pt.x;
      A.at<double>(i, 2) = imgpoints[i].pt.y;
      Ax.at<double>(i, 0) = grdpoints[i].pt.x;
      Ay.at<double>(i, 0) = grdpoints[i].pt.y;
    }
    //solve for the coefficients
    coeff_x = A.inv()*Ax;
    coeff_y = A.inv()*Ay;
    //copy over the coefficients
    for (int_t i_coeff = 0; i_coeff < 3; i_coeff++) {
      img_to_grdx_[i_coeff] = coeff_x.at<double>(i_coeff, 0);
      img_to_grdy_[i_coeff] = coeff_y.at<double>(i_coeff, 0);
    }
    //set the higher order terms to 0
    img_to_grdx_[3] = 0.0;
    img_to_grdy_[3] = 0.0;
    img_to_grdx_[4] = 0.0;
    img_to_grdy_[4] = 0.0;
    img_to_grdx_[5] = 0.0;
    img_to_grdy_[5] = 0.0;

    //grid to image transform (not overdetermined so just solve the matrix equation)
    for (int_t i = 0; i < 3; i++) {
      A.at<double>(i, 0) = 1.0;
      A.at<double>(i, 1) = grdpoints[i].pt.x;
      A.at<double>(i, 2) = grdpoints[i].pt.y;
      Ax.at<double>(i, 0) = imgpoints[i].pt.x;
      Ay.at<double>(i, 0) = imgpoints[i].pt.y;
    }
    //solve for the coefficients
    coeff_x = A.inv()*Ax;
    coeff_y = A.inv()*Ay;
    //copy over the coefficients
    for (int_t i_coeff = 0; i_coeff < 3; i_coeff++) {
      grd_to_imgx_[i_coeff] = coeff_x.at<double>(i_coeff, 0);
      grd_to_imgy_[i_coeff] = coeff_y.at<double>(i_coeff, 0);
    }
    //set the higher order terms to 0
    grd_to_imgx_[3] = 0.0;
    grd_to_imgy_[3] = 0.0;
    grd_to_imgx_[4] = 0.0;
    grd_to_imgy_[4] = 0.0;
    grd_to_imgx_[5] = 0.0;
    grd_to_imgy_[5] = 0.0;
  }

  //if more than three points are supplied use a 12 parameter mapping to reflect keystoning
  if (imgpoints.size() > 3) {

    Mat A = Mat_<double>(imgpoints.size(), 6);
    Mat AtA = Mat_<double>(6, 6);
    Mat bx = Mat_<double>(imgpoints.size(), 1);
    Mat by = Mat_<double>(imgpoints.size(), 1);
    Mat Atb = Mat_<double>(6, 1);
    Mat coeff_x = Mat_<double>(6, 1);
    Mat coeff_y = Mat_<double>(6, 1);

    //image to grid transform (least squares fit)
    for (int_t i = 0; i < (int_t)imgpoints.size(); i++) {
      A.at<double>(i, 0) = 1.0;
      A.at<double>(i, 1) = imgpoints[i].pt.x;
      A.at<double>(i, 2) = imgpoints[i].pt.y;
      A.at<double>(i, 3) = imgpoints[i].pt.x * imgpoints[i].pt.y;
      A.at<double>(i, 4) = imgpoints[i].pt.x * imgpoints[i].pt.x;
      A.at<double>(i, 5) = imgpoints[i].pt.y * imgpoints[i].pt.y;
      bx.at<double>(i, 0) = grdpoints[i].pt.x;
      by.at<double>(i, 0) = grdpoints[i].pt.y;
    }
    //solve for the coefficients
    AtA = A.t()*A;
    Atb = A.t()*bx;
    coeff_x = AtA.inv()*Atb;
    Atb = A.t()*by;
    coeff_y = AtA.inv()*Atb;

    //copy over the coefficients
    for (int_t i_coeff = 0; i_coeff < 6; i_coeff++) {
      img_to_grdx_[i_coeff] = coeff_x.at<double>(i_coeff, 0);
      img_to_grdy_[i_coeff] = coeff_y.at<double>(i_coeff, 0);
    }

    //grid to image transform
    for (int_t i = 0; i < (int_t)imgpoints.size(); i++) {
      A.at<double>(i, 0) = 1.0;
      A.at<double>(i, 1) = grdpoints[i].pt.x;
      A.at<double>(i, 2) = grdpoints[i].pt.y;
      A.at<double>(i, 3) = grdpoints[i].pt.x * grdpoints[i].pt.y;
      A.at<double>(i, 4) = grdpoints[i].pt.x * grdpoints[i].pt.x;
      A.at<double>(i, 5) = grdpoints[i].pt.y * grdpoints[i].pt.y;
      bx.at<double>(i, 0) = imgpoints[i].pt.x;
      by.at<double>(i, 0) = imgpoints[i].pt.y;
    }
    //solve for the coefficients
    AtA = A.t()*A;
    Atb = A.t()*bx;
    coeff_x = AtA.inv()*Atb;
    Atb = A.t()*by;
    coeff_y = AtA.inv()*Atb;

    //copy over the coefficients
    for (int_t i_coeff = 0; i_coeff < 6; i_coeff++) {
      grd_to_imgx_[i_coeff] = coeff_x.at<double>(i_coeff, 0);
      grd_to_imgy_[i_coeff] = coeff_y.at<double>(i_coeff, 0);
    }
  }
  //DEBUG_MSG("Calibration::calc_trans_coeff(): end");
}

//create a box around the valid points
void
Calibration::create_bounding_box(std::vector<float> & box_x,
  std::vector<float> & box_y) {
  assert(box_x.size()==box_y.size());
  assert(box_x.size()==5);
  float xgrid, ygrid;
  //xmin, ymin point
  xgrid = - 0.5;
  ygrid = - 0.5;
  grid_to_image(xgrid, ygrid, box_x[0], box_y[0]);
  //xmax, ymin point
  xgrid = num_fiducials_x_ -1 + 0.5;
  ygrid =  - 0.5;
  grid_to_image(xgrid, ygrid, box_x[1], box_y[1]);
  //xmax, ymax point
  xgrid = num_fiducials_x_  - 1 + 0.5;
  ygrid = num_fiducials_y_  - 1 + 0.5;
  grid_to_image(xgrid, ygrid, box_x[2], box_y[2]);
  //xmin, ymax point
  xgrid =  - 0.5;
  ygrid = num_fiducials_y_ - 1 + 0.5;
  grid_to_image(xgrid, ygrid, box_x[3], box_y[3]);
  //close the loop
  box_x[4] = box_x[0];
  box_y[4] = box_y[0];
}

//convert grid locations to image locations
void
Calibration::grid_to_image(const float & grid_x,
  const float & grid_y,
  float & img_x,
  float & img_y) {
  img_x = grd_to_imgx_[0] + grd_to_imgx_[1] * grid_x + grd_to_imgx_[2] * grid_y + grd_to_imgx_[3] * grid_x * grid_y + grd_to_imgx_[4] * grid_x * grid_x + grd_to_imgx_[5] * grid_y * grid_y;
  img_y = grd_to_imgy_[0] + grd_to_imgy_[1] * grid_x + grd_to_imgy_[2] * grid_y + grd_to_imgy_[3] * grid_x * grid_y + grd_to_imgy_[4] * grid_x * grid_x + grd_to_imgy_[5] * grid_y * grid_y;
  if(img_x<0) img_x = 0;
  if(img_x>image_size_.width-1) img_x = image_size_.width-1;
  if(img_y<0) img_y = 0;
  if(img_y>image_size_.height-1) img_y = image_size_.height-1;
}

//convert image locations to grid locations
void
Calibration::image_to_grid(const float & img_x,
  const float & img_y,
  float & grid_x,
  float & grid_y) {
  grid_x = img_to_grdx_[0] + img_to_grdx_[1] * img_x + img_to_grdx_[2] * img_y + img_to_grdx_[3] * img_x * img_y + img_to_grdx_[4] * img_x * img_x + img_to_grdx_[5] * img_y * img_y;
  grid_y = img_to_grdy_[0] + img_to_grdy_[1] * img_x + img_to_grdy_[2] * img_y + img_to_grdy_[3] * img_x * img_y + img_to_grdy_[4] * img_x * img_x + img_to_grdy_[5] * img_y * img_y;
}

//distance between two points
float
Calibration::dist2(KeyPoint pnt1, KeyPoint pnt2) {
  return (pnt1.pt.x - pnt2.pt.x)*(pnt1.pt.x - pnt2.pt.x) + (pnt1.pt.y - pnt2.pt.y)*(pnt1.pt.y - pnt2.pt.y);
}

//order three distances returns biggest to smallest
void
Calibration::order_dist3(std::vector<float> & dist,
  std::vector<int> & dist_order) {
  dist_order[0] = 0;
  dist_order[2] = 0;
  for (int_t i = 1; i < 3; i++) {
    if (dist[dist_order[0]] < dist[i]) dist_order[0] = i;
    if (dist[dist_order[2]] > dist[i]) dist_order[2] = i;
  }
  if (dist_order[0] == dist_order[2]) assert(false);
  dist_order[1] = 3 - (dist_order[0] + dist_order[2]);
}

//is a point contained within the quadrilateral
bool
Calibration::is_in_quadrilateral(const float & x,
  const float & y,
  const std::vector<float> & box_x,
  const std::vector<float> & box_y) {
  //is the point within the box
  float angle = 0.0;
  assert(box_x.size() == 5);
  assert(box_y.size() == 5);

  for (int_t i = 0; i < 4; i++) {
    // get the two end points of the polygon side and construct
    // a vector from the point to each one:
    const float dx1 = box_x[i] - x;
    const float dy1 = box_y[i] - y;
    const float dx2 = box_x[i + 1] - x;
    const float dy2 = box_y[i + 1] - y;
    angle += angle_2d(dx1, dy1, dx2, dy2); //angle_2d from DICe_shape
  }
  // if the angle is greater than PI, the point is in the polygon
  if (std::abs(angle) >= DICE_PI) {
    return true;
  }
  return false;
}

//filter the dot markers by size, bounding box and closeness to the expected grid location
void
Calibration::filter_dot_markers(int_t i_cam,
  int_t i_img,
  std::vector<cv::KeyPoint>  dots,
  std::vector<cv::KeyPoint> & img_points,
  std::vector<cv::KeyPoint> & grd_points,
  float dot_tol,
  cv::Mat out_img,
  bool draw) {

  //bounding box values
  std::vector<float> box_x(5, 0.0);
  std::vector<float> box_y(5, 0.0);
  //returned grid values and the interger grid locations
  float grid_x, grid_y;
  long grid_ix, grid_iy;
  //single point and keypoint for drawing and storage
  Point cvpoint;
  KeyPoint cvkeypoint;
  //return value
  //bool grid_dots_ok = true;

  //clear the grid and images points
  grd_points.clear();
  img_points.clear();

  //create the bounding box for the points
  create_bounding_box(box_x, box_y);
  if (draw) {
    std::vector<Point> contour;
    for (size_t n = 0; n < box_x.size(); ++n) {
      contour.push_back(Point(box_x[n],box_y[n]));
    }
    polylines(out_img, contour, true, Scalar(255, 255, 153),2);
  }
  // compute the average dot size:
  float avg_dot_size = 0.0;
  assert(dots.size() > 0);
  for (size_t i = 0; i < dots.size(); ++i) {
    avg_dot_size += dots[i].size;
  }
  avg_dot_size /= dots.size();

  //filter the points
  for (size_t n = 0; n < dots.size(); ++n) {
    //if requested draw all the found points
    if (draw) {
      //circle needs cvpoints not key points
      cvpoint.x = dots[n].pt.x;
      cvpoint.y = dots[n].pt.y;
      //draw the white (found) circle
      circle(out_img, cvpoint, 20, Scalar(0, 0, 255), 2);
    }

    //is the point in an acceptable size range
    if (dots[n].size < 0.8f*avg_dot_size || dots[n].size > 1.4f*avg_dot_size) continue;
    //draw the black (size ok) circle
    //if (draw) circle(out_img, cvpoint, 16, Scalar(0, 0, 0), -1);

    //is the point in the bounding box
    if (!is_in_quadrilateral(dots[n].pt.x, dots[n].pt.y, box_x, box_y)) continue;
    //if (draw) circle(out_img, cvpoint, 12, Scalar(255, 0, 255), -1);

    //get the corresponding grid location from the image location
    image_to_grid((float)dots[n].pt.x, (float)dots[n].pt.y, grid_x, grid_y);
    //get the nearest integer dot location
    grid_ix = std::lround(grid_x);
    grid_iy = std::lround(grid_y);
    //is it within the acceptable distance from the expected location and in the desired grid area
    if (abs(grid_x - round(grid_x)) <= dot_tol && abs(grid_y - round(grid_y)) <= dot_tol &&
        grid_ix>=0 && grid_ix < num_fiducials_x_ && grid_iy>=0 && grid_iy < num_fiducials_y_) {
      //save the point
      cvkeypoint.pt.x = grid_ix;
      cvkeypoint.pt.y = grid_iy;
      img_points.push_back(dots[n]);
      grd_points.push_back(cvkeypoint);
      if(draw) circle(out_img, cvpoint, 12, Scalar(0, 255, 0), 4);
    }
  }//end dots loop

  //draw the expected locations
  if (draw) {
    float imgx, imgy;
    for (float i_x = 0; i_x < num_fiducials_x_; i_x++) {
      for (float i_y = 0; i_y < num_fiducials_y_; i_y++) {
        grid_to_image(i_x, i_y, imgx, imgy);
        cvpoint.x = imgx;
        cvpoint.y = imgy;
        circle(out_img, cvpoint, 5, Scalar(255, 0, 255), -1);
      }
    }
  }
}

//write a file with the intersection information
void
Calibration::write_calibration_file(const std::string & filename) {

  TEUCHOS_TEST_FOR_EXCEPTION(filename.find("xml") == std::string::npos,std::runtime_error,"invalid file extension (should be .xml)");
  DEBUG_MSG("Calibration::write_calibration_file(): writing calibration file: " << filename);

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
  os.precision(0);
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
