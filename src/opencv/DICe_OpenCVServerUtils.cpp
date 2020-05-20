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

/*! \file  DICe_OpenCVServer.cpp
    \brief Utility server to perform OpenCV operations on an image, used by the GUI
*/

#include <DICe.h>
#include <DICe_OpenCVServerUtils.h>
#include <DICe_Parser.h>
#include <DICe_Calibration.h>
#include <DICe_CameraSystem.h>
#ifdef DICE_ENABLE_TRACKLIB
#include <TrackLib_Driver.h>
#endif

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <cassert>

using namespace cv;
namespace DICe{

// parse the input string and return a Teuchos ParameterList
DICE_LIB_DLL_EXPORT
Teuchos::ParameterList parse_filter_string(int argc, char *argv[]){
  DEBUG_MSG("opencv_server::parse_filter_string():");
  DEBUG_MSG("User specified " << argc << " arguments");
  //for(int_t i=0;i<argc;++i){
  //  DEBUG_MSG(argv[i]);
  //}

  Teuchos::ParameterList params;
  Teuchos::ParameterList io_files;

  int_t end_images = 0;
  bool i_o_flag = true;
  std::string temp_input_file;
  for(int_t i=1;i<argc;++i){ // the 0th entry is the executable name so start with 1
    std::string arg = argv[i];
    if(arg.find('.')!=std::string::npos&&!std::isdigit(arg[0])&&(arg.find(':')==std::string::npos||arg.find(':')==1)){ // on win there is a ':' in the file path!
      if(i_o_flag){
        temp_input_file = arg;
      }else{
        if(arg.find(".png")==std::string::npos){
          std::cout << "error, invalid output image format, (only .png is allowed)" << std::endl;
          params = Teuchos::ParameterList();
          return params;
        }
        io_files.set(temp_input_file,arg);
        DEBUG_MSG("Adding input image: " << temp_input_file << " output image: " << arg);
      }
      i_o_flag = !i_o_flag;
    }else{
      end_images = i;
      break;
    }
  } // end argc iterations search for image names

  TEUCHOS_TEST_FOR_EXCEPTION(io_files.numParams()<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!(end_images%2),std::runtime_error,"");
  params.set(opencv_server_io_files, io_files);

  Teuchos::ParameterList filters;
  Teuchos::ParameterList filter_params;
  bool name_value_flag = true;
  std::string temp_name_string;
  std::string filter_name;
  // read the list of filters:
  for(int_t i=end_images;i<argc;++i){
    std::string arg = argv[i];
    if(arg.find("filter:")!=std::string::npos){
      // remove the Filter: decorator
      arg.erase(0,7);
      if(filter_params.numParams()>0) // save off the previous filter's params
        filters.set(filter_name,filter_params);
      filter_params = Teuchos::ParameterList();
      filter_name = arg;
      if(filter_name=="none")  // save off an empty parameter list for filter:none
        filters.set(filter_name,filter_params);
      continue;
    } // end found filter keyword
    if(name_value_flag)
      temp_name_string = argv[i];
    else{
      std::string arg_upper = arg;
      std::transform(arg_upper.begin(),arg_upper.end(),arg_upper.begin(),::toupper);
      if(std::isdigit(arg[0])||arg[0]=='-'){ // is the string a number?
        if(arg.find('.')!=std::string::npos){ // is it a double
          filter_params.set(temp_name_string,std::strtod(arg.c_str(),NULL));
        }else{
          filter_params.set(temp_name_string,std::atoi(arg.c_str()));
        }
      }else if(arg_upper.find("TRUE")!=std::string::npos){
        filter_params.set(temp_name_string,true);
      }else if(arg_upper.find("FALSE")!=std::string::npos){ // test for bools
        filter_params.set(temp_name_string,false);
      }
      else{ // otherwise add a string parameter
        filter_params.set(temp_name_string,arg);
      }
    }
    name_value_flag = !name_value_flag;
    if(i==argc-1)
      filters.set(filter_name,filter_params);
  } // end arg iteration
  params.set(opencv_server_filters,filters);

  return params;
}

DICE_LIB_DLL_EXPORT
int_t opencv_server(int argc, char *argv[]){
  DEBUG_MSG("opencv_server(): begin");

  int_t error_code = 0;

  // parse the string and return error code if it is an empty parameterlist
  Teuchos::ParameterList input_params = parse_filter_string(argc,argv);
#ifdef DICE_DEBUG_MSG
  input_params.print(std::cout);
#endif
  Teuchos::ParameterList io_files = input_params.get<Teuchos::ParameterList>(DICe::opencv_server_io_files,Teuchos::ParameterList());
  Teuchos::ParameterList filters = input_params.get<Teuchos::ParameterList>(DICe::opencv_server_filters,Teuchos::ParameterList());

  // if the background filter is active, create a background image to pass to subsequent filters:
  Mat background_img; // empty if no background image is available
  for(Teuchos::ParameterList::ConstIterator filter_it=filters.begin();filter_it!=filters.end();++filter_it){
    if(filter_it->first == "background"){
      Teuchos::ParameterList options = filters.get<Teuchos::ParameterList>(filter_it->first,Teuchos::ParameterList());
      if(options.get<int>(opencv_server_background_num_frames,0)==0) break; // zero means don't do background subtraction
      opencv_create_cine_background_image(options);
      const std::string background_file = options.get<std::string>(opencv_server_background_file_name); // the check that this param exists happens in function above
      background_img = imread(background_file, IMREAD_GRAYSCALE);
      break;
    }
  }

  // iterate the selected images
  for(Teuchos::ParameterList::ConstIterator file_it=io_files.begin();file_it!=io_files.end();++file_it){
    std::string image_in_filename = file_it->first;
    std::string image_out_filename = io_files.get<std::string>(file_it->first);
    DEBUG_MSG("Processing image: " << image_in_filename << " output image " << image_out_filename);
    // load the image as an openCV mat
    Mat img;
    if(image_in_filename.find("filter")!=std::string::npos)
      img = imread(image_in_filename, IMREAD_COLOR); // if it's a filtered image it might have color annotations
    else
      img = imread(image_in_filename, IMREAD_GRAYSCALE);
    if(img.empty()){
      std::cout << "*** error, the image is empty" << std::endl;
      return 4;
    }
    if(!img.data){
      std::cout << "*** error, the image failed to load" << std::endl;
      return 4;
    }
    // iterate the selected filters
    for(Teuchos::ParameterList::ConstIterator filter_it=filters.begin();filter_it!=filters.end();++filter_it){
      std::string filter = filter_it->first;
      DEBUG_MSG("Applying filter: " << filter);
      Teuchos::ParameterList options = filters.get<Teuchos::ParameterList>(filter,Teuchos::ParameterList());
      // switch statement on the filters, pass the options as an argument and the opencv image mat
      if(filter==opencv_server_filter_none||filter==opencv_server_filter_background){
        // no op
      }
      else if(filter==opencv_server_filter_binary_threshold){
        error_code = opencv_binary_threshold(img,options);
      }else if(filter==opencv_server_filter_adaptive_threshold){
        error_code = opencv_adaptive_threshold(img,options);
      }else if(filter==opencv_server_filter_checkerboard_targets){
        error_code = opencv_checkerboard_targets(img,options);
      }else if(filter==opencv_server_filter_dot_targets){
        int_t return_thresh = 0.0;
        error_code = opencv_dot_targets(img,options,return_thresh);
      }else if(filter==opencv_server_filter_epipolar_line){
        error_code = opencv_epipolar_line(img,options,file_it==io_files.begin());
      }else if(filter==opencv_server_filter_tracklib){
        DEBUG_MSG("Applying TrackLib filter");
#ifdef DICE_ENABLE_TRACKLIB
        error_code = TrackLib::tracklib_preview(img,background_img,options);
#else
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Filter tracklib is only available when tracklib is available");
#endif
      }else{
        std::cout << "error, unknown filter: " << filter << std::endl;
        error_code = 5;
      }
    } // end filter iteration
    //if(error_code!=-1){
    DEBUG_MSG("Writing output image: " << image_out_filename);
    imwrite(image_out_filename, img);
    //}
    //if(error_code) // exit on the first error
    //  return error_code;
  } // end file iteration
  DEBUG_MSG("opencv_server(): end");
  return error_code;
}

DICE_LIB_DLL_EXPORT
int_t opencv_adaptive_threshold(Mat & img, Teuchos::ParameterList & options){
  // mean is 0 gaussian is 1
  int filter_mode = options.get<int_t>(opencv_server_filter_mode,1);
  DEBUG_MSG("option, filter mode:     " << filter_mode);
  //  cv::THRESH_BINARY = 0,
  //  cv::THRESH_BINARY_INV = 1,
  //  cv::THRESH_TRUNC = 2,
  //  cv::THRESH_TOZERO = 3,
  //  cv::THRESH_TOZERO_INV = 4,
  //  cv::THRESH_MASK = 7,
  //  cv::THRESH_OTSU = 8,
  //  cv::THRESH_TRIANGLE = 16
  int threshold_mode = options.get<int_t>(opencv_server_threshold_mode,0);
  DEBUG_MSG("option, threshold mode:   " << threshold_mode);
  int block_size = options.get<int_t>(opencv_server_block_size,75);
  DEBUG_MSG("option, block size:      " << block_size);
  double binary_constant = options.get<double>(opencv_server_binary_constant,100.0);
  DEBUG_MSG("option, binary constant: " << binary_constant);
  adaptiveThreshold(img,img,255.0,filter_mode,threshold_mode,block_size,binary_constant);
  return 0;
}

DICE_LIB_DLL_EXPORT
int_t opencv_create_cine_background_image(Teuchos::ParameterList & options){
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(opencv_server_cine_file_name),std::runtime_error,"");
  const std::string cine_file = options.get<std::string>(opencv_server_cine_file_name);
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(opencv_server_background_file_name),std::runtime_error,"");
  const std::string background_file = options.get<std::string>(opencv_server_background_file_name);
  DEBUG_MSG("opencv_create_cine_background_image(): cine file:          " << cine_file);
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(opencv_server_background_ref_frame),std::runtime_error,"");
  const int_t ref_frame = options.get<int>(opencv_server_background_ref_frame);
  DEBUG_MSG("opencv_create_cine_background_image(): cine ref frame:     " << ref_frame);
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(opencv_server_background_num_frames),std::runtime_error,"");
  const int_t num_frames_to_avg = options.get<int>(opencv_server_background_num_frames);
  DEBUG_MSG("opencv_create_cine_background_image(): num frames to avg:  " << num_frames_to_avg);

  // read the first cine frame in the file
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  params->set(filter_failed_cine_pixels,true);
  params->set(convert_cine_to_8_bit,true);
  params->set(DICe::reinitialize_cine_reader_conversion_factor,true);
  // insert the frame number before the extension
  TEUCHOS_TEST_FOR_EXCEPTION(cine_file.find(".cine")==std::string::npos,std::runtime_error,"invalid cine file name");
  size_t found_ext = cine_file.find(".cine");
  std::stringstream zero_frame_ss; // decorate the cine file name to indicate averaging frames ref_frame to ref_frame + num_frames_to_avg
  zero_frame_ss << "_avg" << ref_frame << "to" << ref_frame+num_frames_to_avg;
  std::string cine_file_zero_frame = cine_file;
  cine_file_zero_frame.insert(found_ext,zero_frame_ss.str());
  Teuchos::RCP<Image> bg_img = Teuchos::rcp(new Image(cine_file_zero_frame.c_str(),params));
  // write an output .tiff image
  DEBUG_MSG("opencv_create_cine_background_image(): writing background file: " << background_file);
  bg_img->write(background_file);

  DEBUG_MSG("opencv_create_cine_background_image(): complete " << background_file);

  return 0;
}

DICE_LIB_DLL_EXPORT
int_t opencv_epipolar_line(Mat & img,
  Teuchos::ParameterList & options,
  const bool first_image){
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(opencv_server_epipolar_is_left),std::runtime_error,"");
  const bool is_left = options.get<bool>(opencv_server_epipolar_is_left);
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(opencv_server_epipolar_dot_x),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(opencv_server_epipolar_dot_y),std::runtime_error,"");
  int dot_x = options.get<int_t>(opencv_server_epipolar_dot_x,0);
  int dot_y = options.get<int_t>(opencv_server_epipolar_dot_y,0);

  int frame_number = options.get<int>("frame_number");
  std::stringstream yml_file_left;
  std::stringstream yml_file_right;
  yml_file_left << ".dice/.keypoints_left_" << frame_number << ".yml";
  yml_file_right << ".dice/.keypoints_right_" << frame_number << ".yml";

  cv::Scalar color = cv::Scalar(255, 0, 255);
  cv::KeyPoint kpt(cv::Point2f(dot_x,dot_y),1.0);
  bool snap_point = false;
#ifdef DICE_ENABLE_TRACKLIB
  // find the nearest keypoint to the clicked point
  const float distance_threshold = 250.0; //(squared_distance)
  const std::string keypoint_filename = is_left ? yml_file_left.str() : yml_file_right.str();
  DEBUG_MSG("keypoint filename: " << keypoint_filename);
  cv::KeyPoint snap_kpt = TrackLib::snap_to_keypoint(kpt,keypoint_filename,distance_threshold);
  if(snap_kpt.pt.x >0){
    DEBUG_MSG("original point: " << kpt.pt.x << " " << kpt.pt.y << " snapped point " << snap_kpt.pt.x << " " << snap_kpt.pt.y);
    kpt = snap_kpt;
    snap_point = true;
  }
#endif

  // if this is the dot image, draw a dot
  if((first_image&&is_left)||((!first_image)&&(!is_left))){
    DEBUG_MSG("drawing dot on image");
    // get the location of the dot
    DEBUG_MSG("option, dot location: " << dot_x << " " << dot_y);
    if(img.channels() == 1)
      cvtColor(img, img, cv::COLOR_GRAY2RGB);
    circle(img, kpt.pt, 5, Scalar(0, 0, 255),-1);
    if(snap_point){
      int radius = std::max((int)kpt.size/2, 15);
      cv::circle(img, kpt.pt, radius, color, 2, 8, 0);
      cv::circle(img, kpt.pt, 2, color, -1, 8, 0);
    }
  }else{
    DEBUG_MSG("drawing line on image");
    TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(opencv_server_cal_file),std::runtime_error,"");
    Teuchos::RCP<DICe::Camera_System> camera_system = Teuchos::rcp(new DICe::Camera_System(options.get<std::string>(opencv_server_cal_file)));
    Matrix<scalar_t,3> F = camera_system->fundamental_matrix();
    // TODO move this to a matrix method?
    cv::Mat matF =  cv::Mat(3, 3, CV_32F);
    for(int_t i=0;i<matF.rows;++i){
      for(int_t j=0;j<matF.cols;++j){
        matF.at<float>(i,j) = F(i,j);
      }
    }
    Mat lines;
    int whichImage = options.get<bool>(opencv_server_epipolar_is_left) ? 1:2;
    std::vector<Point2f> points;
    points.push_back(kpt.pt);
    cv::computeCorrespondEpilines(points,whichImage,matF,lines);
    scalar_t a = lines.at<float>(0);
    scalar_t b = lines.at<float>(1);
    scalar_t c = lines.at<float>(2);
    TEUCHOS_TEST_FOR_EXCEPTION(b==0.0,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(a==0.0,std::runtime_error,"");
    Point2f ptX(0.0,-1.0*c/b);
    Point2f ptY(img.cols,-1.0*(c + a*img.cols)/b);
    if(img.channels() == 1)
      cvtColor(img, img, cv::COLOR_GRAY2RGB);
    cv::line(img,ptX,ptY,Scalar(0,0,255),2);

//#ifdef DICE_ENABLE_TRACKLIB
//    const std::string stereo_keypoint_filename = is_left ? yml_file_right.str() : yml_file_left.str();
//    const float stereo_distance_threshold = 1.0;
//    DEBUG_MSG("stereo keypoint filename: " << stereo_keypoint_filename);
//    cv::KeyPoint corr_kpt = TrackLib::find_corresponding_keypoint(kpt,stereo_keypoint_filename,lines,stereo_distance_threshold);
//    if(corr_kpt.pt.x>0){
//      int radius = std::max((int)corr_kpt.size/2, 15);
//      cv::circle(img, corr_kpt.pt, radius, color, 2, 8, 0);
//      cv::circle(img, corr_kpt.pt, 2, color, -1, 8, 0);
//      // triangulate the keypoints
//      cv::Point3f pt_3d;
//      if(is_left){
//        pt_3d = TrackLib::trangulate_keypoints(kpt,corr_kpt,options.get<std::string>(opencv_server_cal_file));
//      }else{
//        pt_3d = TrackLib::trangulate_keypoints(corr_kpt,kpt,options.get<std::string>(opencv_server_cal_file));
//      }
//      std::stringstream point_text;
//      point_text << "pos: (" << pt_3d.x << "," << pt_3d.y << "," << pt_3d.z << ") size: " << corr_kpt.size;
//      cv::putText(img,point_text.str(),cv::Point2f(25,25), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255), 1, CV_AA);
//    }
//#endif
  }
  return 0;
}

DICE_LIB_DLL_EXPORT
int_t opencv_binary_threshold(Mat & img, Teuchos::ParameterList & options){
  //  cv::THRESH_BINARY = 0,
  //  cv::THRESH_BINARY_INV = 1,
  //  cv::THRESH_TRUNC = 2,
  //  cv::THRESH_TOZERO = 3,
  //  cv::THRESH_TOZERO_INV = 4,
  //  cv::THRESH_MASK = 7,
  //  cv::THRESH_OTSU = 8,
  //  cv::THRESH_TRIANGLE = 16
  int threshold_mode = options.get<int_t>(opencv_server_threshold_mode,0);
  DEBUG_MSG("option, threshold mode:   " << threshold_mode);
  double binary_constant = options.get<double>(opencv_server_binary_constant,100.0);
  DEBUG_MSG("option, binary constant: " << binary_constant);
  threshold(img, img, binary_constant, 255.0, threshold_mode);
  return 0;
}

DICE_LIB_DLL_EXPORT
int_t opencv_checkerboard_targets(Mat & img, Teuchos::ParameterList & options){
  std::vector<Point2f> corners;
  return opencv_checkerboard_targets(img,options,corners);
}

DICE_LIB_DLL_EXPORT
int_t opencv_checkerboard_targets(Mat & img, Teuchos::ParameterList & options,
  std::vector<Point2f> & corners){
  // establish the calibration plate properties
  if(!options.isParameter(DICe::num_cal_fiducials_x)||!options.isParameter(DICe::num_cal_fiducials_y)){
    std::cout << "*** error, missing checkerboard dimensions" << std::endl;
    return 2;
  }
  corners.clear();
  const int_t num_fiducials_x = options.get<int_t>(DICe::num_cal_fiducials_x);
  const int_t num_fiducials_y = options.get<int_t>(DICe::num_cal_fiducials_y);
  //std::cout << "opencv_checkerboard_targets(): option, board size x:   " << num_fiducials_x << std::endl;
  //std::cout << "opencv_checkerboard_targets(): option, board size y:   " << num_fiducials_y << std::endl;
  Size board_size; //used by openCV
  //set the height and width of the board in intersections
  board_size.width = num_fiducials_x;
  board_size.height = num_fiducials_y;

  // FIXME this method goes into an infinite loop if the number of corners in x and y is not correct
  // come up with a way to detect this...
  const bool found = findChessboardCorners(img, board_size, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

  if(found){
    std::cout << "opencv_checkerboard_targets(): found " << corners.size() << " checkerboard intersections" << std::endl;
    // improve the locations with cornerSubPixel
    // need to check out the window and zero zone parameters to see their effect
    cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
      TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
        30, 0.01));
    cvtColor(img, img, cv::COLOR_GRAY2RGB);
    int_t i_pnt = 0;
    for (int_t i_y = 0; i_y < num_fiducials_y; i_y++) {
      for (int_t i_x = 0; i_x < num_fiducials_x; i_x++) {
        circle(img, corners[i_pnt], 3, Scalar(0, 255, 0));
        i_pnt++;
      } // end corner loop x
    } // end corner loop y
  }else{
    std::cout << "opencv_checkerboard(): failed" << std::endl;
    return 3;
  }
  return 0;
}

DICE_LIB_DLL_EXPORT
int_t opencv_dot_targets(cv::Mat & img,
  Teuchos::ParameterList & options,
  int_t & return_thresh){
  std::vector<cv::KeyPoint> key_points;
  std::vector<cv::KeyPoint> img_points;
  std::vector<cv::KeyPoint> grd_points;
  return opencv_dot_targets(img,options,key_points,img_points,grd_points, return_thresh);
}

DICE_LIB_DLL_EXPORT
int_t opencv_dot_targets(Mat & img,
  Teuchos::ParameterList & options,
  std::vector<cv::KeyPoint> & key_points,
  std::vector<cv::KeyPoint> & img_points,
  std::vector<cv::KeyPoint> & grd_points,
  int_t & return_thresh){
  key_points.clear();
  img_points.clear();
  grd_points.clear();

  // clone the input image so that we have a copy of the original
  Mat img_cpy = img.clone();

  const int_t threshold_start = options.get<int_t>(opencv_server_threshold_start,20);
  //std::cout << "opencv_dot_targets(): option, threshold start:   " << threshold_start << std::endl;
  const int_t threshold_end = options.get<int_t>(opencv_server_threshold_end,250);
  //std::cout << "opencv_dot_targets(): option, threshold end:     " << threshold_end << std::endl;
  const int_t threshold_step = options.get<int_t>(opencv_server_threshold_step,5);
  //std::cout << "opencv_dot_targets(): option, threshold step:    " << threshold_step << std::endl;
  const bool preview_thresh = options.get<bool>(opencv_server_preview_threshold,false);
  //std::cout << "opencv_dot_targets(): option, preview thresh:    " << preview_thresh << std::endl;
  const int_t block_size = options.get<int_t>(opencv_server_block_size,75); // The old method had default set to 75
  //std::cout << "opencv_dot_targets(): option, block size:        " << block_size << std::endl;
  const bool use_adaptive = options.get<bool>(opencv_server_use_adaptive_threshold,false);
  //std::cout << "opencv_dot_targets(): option, use adaptive:      " << use_adaptive << std::endl;
  TEUCHOS_TEST_FOR_EXCEPTION(!use_adaptive&&threshold_start<=0,std::runtime_error,"");
  int filter_mode = options.get<int_t>(opencv_server_filter_mode,1);
  //std::cout << "opencv_dot_targets(): option, filter mode:       " << filter_mode << std::endl;
  //  cv::THRESH_BINARY = 0,
  //  cv::THRESH_BINARY_INV = 1,
  //  cv::THRESH_TRUNC = 2,
  //  cv::THRESH_TOZERO = 3,
  //  cv::THRESH_TOZERO_INV = 4,
  //  cv::THRESH_MASK = 7,
  //  cv::THRESH_OTSU = 8,
  //  cv::THRESH_TRIANGLE = 16
  int threshold_mode = options.get<int_t>(opencv_server_threshold_mode,0);
  //std::cout << "opencv_dot_targets(): option, threshold mode:   " << threshold_mode << std::endl;

  // establish the calibration plate properties
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(DICe::num_cal_fiducials_x),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(DICe::num_cal_fiducials_y),std::runtime_error,"");
  const int_t num_fiducials_x = options.get<int_t>(DICe::num_cal_fiducials_x);
  const int_t num_fiducials_y = options.get<int_t>(DICe::num_cal_fiducials_y);
  TEUCHOS_TEST_FOR_EXCEPTION(num_fiducials_x<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(num_fiducials_y<=0,std::runtime_error,"");
  const int_t origin_loc_x = options.get<int_t>(DICe::cal_origin_x,0);
  const int_t origin_loc_y = options.get<int_t>(DICe::cal_origin_y,0);
  const int_t num_fiducials_origin_to_x_marker = options.get<int_t>(DICe::num_cal_fiducials_origin_to_x_marker,num_fiducials_x);
  const int_t num_fiducials_origin_to_y_marker = options.get<int_t>(DICe::num_cal_fiducials_origin_to_y_marker,num_fiducials_y);
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter(DICe::cal_target_type),std::runtime_error,"");
  Calibration::Target_Type target_type = Calibration::string_to_target_type(options.get<std::string>(DICe::cal_target_type));
  const bool invert = target_type==Calibration::BLACK_ON_WHITE_W_DONUT_DOTS;
  const double dot_tol = options.get<double>(opencv_server_dot_tol,0.25);

  std::vector<KeyPoint> marker_grid_locs;
  marker_grid_locs.resize(3);
  marker_grid_locs[0].pt.x = origin_loc_x;
  marker_grid_locs[0].pt.y = origin_loc_y;
  marker_grid_locs[1].pt.x = origin_loc_x + num_fiducials_origin_to_x_marker - 1;
  marker_grid_locs[1].pt.y = origin_loc_y;
  marker_grid_locs[2].pt.x = origin_loc_x;
  marker_grid_locs[2].pt.y = origin_loc_y + num_fiducials_origin_to_y_marker - 1;

  // find the keypoints in the image
  bool keypoints_found = true;
  //try to find the keypoints at different thresholds
  int_t i_thresh_first = 0;
  int_t i_thresh_last = 0;
  int_t i_thresh = threshold_start;
  if(threshold_start!=threshold_end){
    for (; i_thresh <= threshold_end; i_thresh += threshold_step) {
      // get the dots using an inverted image to get the donut holes
      get_dot_markers(img_cpy, key_points, i_thresh, invert,options);
      // were three keypoints found?
      if (key_points.size() != 3) {
        keypoints_found = false;
        if (i_thresh_first != 0) {
          keypoints_found = true; // keypoints found in a previous pass
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
    // calculate the average threshold value
    i_thresh = (i_thresh_first + i_thresh_last) / 2;
  }

  // get the key points at the average threshold value
  get_dot_markers(img_cpy, key_points, i_thresh, invert,options);

  // it is possible that this threshold does not have 3 points.
  // chances are that this indicates p thresholding problem to begin with
  if (key_points.size() != 3) {
    std::cout << "*** warning: unable to identify three keypoints, other points will not be extracted" << std::endl;
    keypoints_found = false;
  }
  Point cvpoint;
  if(preview_thresh){
    if(use_adaptive){
      adaptiveThreshold(img,img,255,filter_mode,threshold_mode,block_size,i_thresh);
    }else{
      // apply thresholding
      threshold(img,img,i_thresh,255,threshold_mode);
    }
  }
  if(!keypoints_found) return 1;

  // now that we have the keypoints try to get the rest of the dots

  // reorder the keypoints into an origin, xaxis, yaxis order
  reorder_keypoints(key_points);

  // report the results
  std::cout << "opencv_dot_targets():     using threshold: " << i_thresh << std::endl;
  return_thresh = i_thresh;
  DEBUG_MSG("    ordered keypoints: ");
  for (size_t i = 0; i < key_points.size(); ++i) //save and display the keypoints
    DEBUG_MSG("      keypoint: " << key_points[i].pt.x << " " << key_points[i].pt.y);

  // copy the image into the output image
  cvtColor(img, img, cv::COLOR_GRAY2RGB);
  for (int_t n = 0; n < 3; n++) {
    cvpoint.x = key_points[n].pt.x;
    cvpoint.y = key_points[n].pt.y;
    if(img.size().height>800){
      circle(img, cvpoint, 20, Scalar(0, 255, 255), 4);
    }else{
      circle(img, cvpoint, 10, Scalar(0, 255, 255), 4);
    }
  }

  std::vector<scalar_t> img_to_grdx(6,0.0);
  std::vector<scalar_t> img_to_grdy(6,0.0);
  std::vector<scalar_t> grd_to_imgx(6,0.0);
  std::vector<scalar_t> grd_to_imgy(6,0.0);
  // from the keypoints calculate the image to grid and grid to image transforms (no keystoning)
  calc_trans_coeff(key_points, marker_grid_locs,img_to_grdx,img_to_grdy,grd_to_imgx,grd_to_imgy);

  // determine a threshold from the gray levels between the keypoints
  int_t xstart, xend, ystart, yend;
  float maxgray, mingray;
  xstart = key_points[0].pt.x;
  xend = key_points[1].pt.x;
  if (xend < xstart) {
    xstart = key_points[1].pt.x;
    xend = key_points[0].pt.x;
  }
  ystart = key_points[0].pt.y;
  yend = key_points[1].pt.y;
  if (yend < ystart) {
    ystart = key_points[1].pt.y;
    yend = key_points[0].pt.y;
  }

  maxgray = img_cpy.at<uchar>(ystart, xstart);
  mingray = maxgray;
  int_t curgray;
  for (int_t ix = xstart; ix <= xend; ix++) {
    for (int_t iy = ystart; iy <= yend; iy++) {
      curgray = img_cpy.at<uchar>(iy, ix);
      if (maxgray < curgray) maxgray = curgray;
      if (mingray > curgray) mingray = curgray;
    }
  }
  i_thresh = (maxgray + mingray) / 2;
  DEBUG_MSG("  min gray value (inside target keypoints): " << mingray << " max gray value: " << maxgray);
  DEBUG_MSG("  getting the rest of the dots using average gray intensity value as threshold");
  DEBUG_MSG("    threshold to get dots: " << i_thresh);

  // get the rest of the dots
  std::vector<KeyPoint> dots;
  get_dot_markers(img_cpy, dots, i_thresh, !invert,options);
  DEBUG_MSG("    prospective grid points found: " << dots.size());

  // filter dots based on avg size and whether the dots fall in the central box
  filter_dot_markers(dots, img_points, grd_points, grd_to_imgx, grd_to_imgy, img_to_grdx, img_to_grdy,
    num_fiducials_x, num_fiducials_y, dot_tol, img, false);

  // initialize the process variables
  int_t filter_passes = 1;
  int_t old_dot_num = 3;
  int_t new_dot_num = img_points.size();
  int_t max_dots = num_fiducials_x * num_fiducials_x - 3;

  // if the number of dots has not changed
  while ((old_dot_num != new_dot_num && new_dot_num != max_dots && filter_passes < 20) || filter_passes < 3) {
    // update the old dot count
    old_dot_num = new_dot_num;
    // xsfrom the good points that were found improve the mapping parameters
    calc_trans_coeff(img_points, grd_points, img_to_grdx, img_to_grdy, grd_to_imgx, grd_to_imgy);
    // filter dots based on avg size and whether the dots fall in the central box with the new parameters
    // the transformation now includes keystoning
    filter_dot_markers(dots, img_points, grd_points, grd_to_imgx, grd_to_imgy, img_to_grdx, img_to_grdy,
      num_fiducials_x, num_fiducials_y, dot_tol, img, false);
    filter_passes++;
    new_dot_num = img_points.size();
  }

  // if drawing the images run filter one more time and draw the intersection locations
  filter_dot_markers(dots, img_points, grd_points, grd_to_imgx, grd_to_imgy, img_to_grdx, img_to_grdy,
    num_fiducials_x, num_fiducials_y,  dot_tol, img, true);

  // save the information about the found dots
  std::cout << "opencv_dot_targets():     good dots identified: " << new_dot_num << std::endl;
  DEBUG_MSG("    filter passes: " << filter_passes);
  if(new_dot_num < num_fiducials_x*num_fiducials_y*0.75){ // TODO fix this hard coded tolerance
    std::cout << "*** warning: not enough (non-keypoint) dots found" << std::endl;
    return 2; // not an issue with the thresholding (which would have resulted in error code 1)
  }
  else
    return 0;
}

// get all the possible dot markers
void get_dot_markers(cv::Mat img,
  std::vector<KeyPoint> & keypoints,
  int_t thresh,
  bool invert,
  Teuchos::ParameterList & options) {
  DEBUG_MSG("get_dot_markers(): thresh " << thresh << " invert " << invert);
  const int_t block_size = options.get<int_t>(opencv_server_block_size,75); // The old method had default set to 75
  DEBUG_MSG("option, block size:        " << block_size);
  const bool use_adaptive = options.get<bool>(opencv_server_use_adaptive_threshold,false);
  DEBUG_MSG("option, use adaptive:      " << use_adaptive);
  int filter_mode = options.get<int_t>(opencv_server_filter_mode,1);
  DEBUG_MSG("option, filter mode:       " << filter_mode);
  //  cv::THRESH_BINARY = 0,
  //  cv::THRESH_BINARY_INV = 1,
  //  cv::THRESH_TRUNC = 2,
  //  cv::THRESH_TOZERO = 3,
  //  cv::THRESH_TOZERO_INV = 4,
  //  cv::THRESH_MASK = 7,
  //  cv::THRESH_OTSU = 8,
  //  cv::THRESH_TRIANGLE = 16
  int threshold_mode = options.get<int_t>(opencv_server_threshold_mode,0);
  DEBUG_MSG("option, threshold mode:    " << threshold_mode);

  // setup the blob detector
  SimpleBlobDetector::Params params;
  params.maxArea = 10e4;
  params.minArea = 100;
  cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

  // clear the points vector
  keypoints.clear();

  // create a temporary image
  Mat timg;
  img.copyTo(timg);
  // setup the binary image
  Mat bi_src(timg.size(), timg.type());

  // apply thresholding
  if(use_adaptive){
    adaptiveThreshold(img,img,255,filter_mode,threshold_mode,block_size,thresh);
  }else{
    threshold(timg, bi_src, thresh, 255, threshold_mode);
  }
  // invert the source image
  Mat not_src(bi_src.size(), bi_src.type());
  bitwise_not(bi_src, not_src);

  // detect dots on the appropriately inverted image
  if (invert) detector->detect(not_src, keypoints);
  else detector->detect(bi_src, keypoints);
  DEBUG_MSG("get_dot_markers(): num keypoints " << keypoints.size());
}

//calculate the transformation coefficients
void calc_trans_coeff(std::vector<cv::KeyPoint> & imgpoints,
  std::vector<cv::KeyPoint> & grdpoints,
  std::vector<scalar_t> & img_to_grdx,
  std::vector<scalar_t> & img_to_grdy,
  std::vector<scalar_t> & grd_to_imgx,
  std::vector<scalar_t> & grd_to_imgy) {
  TEUCHOS_TEST_FOR_EXCEPTION(grd_to_imgx.size()!=6,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(grd_to_imgy.size()!=6,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(img_to_grdx.size()!=6,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(img_to_grdy.size()!=6,std::runtime_error,"");

  //DEBUG_MSG("calc_trans_coeff(): begin");
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
      img_to_grdx[i_coeff] = coeff_x.at<double>(i_coeff, 0);
      img_to_grdy[i_coeff] = coeff_y.at<double>(i_coeff, 0);
    }
    //set the higher order terms to 0
    img_to_grdx[3] = 0.0;
    img_to_grdy[3] = 0.0;
    img_to_grdx[4] = 0.0;
    img_to_grdy[4] = 0.0;
    img_to_grdx[5] = 0.0;
    img_to_grdy[5] = 0.0;

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
      grd_to_imgx[i_coeff] = coeff_x.at<double>(i_coeff, 0);
      grd_to_imgy[i_coeff] = coeff_y.at<double>(i_coeff, 0);
    }
    //set the higher order terms to 0
    grd_to_imgx[3] = 0.0;
    grd_to_imgy[3] = 0.0;
    grd_to_imgx[4] = 0.0;
    grd_to_imgy[4] = 0.0;
    grd_to_imgx[5] = 0.0;
    grd_to_imgy[5] = 0.0;
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
      img_to_grdx[i_coeff] = coeff_x.at<double>(i_coeff, 0);
      img_to_grdy[i_coeff] = coeff_y.at<double>(i_coeff, 0);
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
      grd_to_imgx[i_coeff] = coeff_x.at<double>(i_coeff, 0);
      grd_to_imgy[i_coeff] = coeff_y.at<double>(i_coeff, 0);
    }
  }
}


//filter the dot markers by size, bounding box and closeness to the expected grid location
void filter_dot_markers(std::vector<cv::KeyPoint>  dots,
  std::vector<cv::KeyPoint> & img_points,
  std::vector<cv::KeyPoint> & grd_points,
  const std::vector<scalar_t> & grd_to_imgx,
  const std::vector<scalar_t> & grd_to_imgy,
  const std::vector<scalar_t> & img_to_grdx,
  const std::vector<scalar_t> & img_to_grdy,
  const int_t num_fiducials_x,
  const int_t num_fiducials_y,
  float dot_tol,
  cv::Mat img,
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
  create_bounding_box(box_x, box_y,num_fiducials_x,num_fiducials_y,
    grd_to_imgx,grd_to_imgy,img.size().width,img.size().height);

  if (draw) {
    std::vector<Point> contour;
    for (size_t n = 0; n < box_x.size(); ++n) {
      contour.push_back(Point(box_x[n],box_y[n]));
    }
    polylines(img, contour, true, Scalar(255, 255, 153),2);
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
      if(img.size().height>800){
        circle(img, cvpoint, 20, Scalar(0, 0, 255), 2);
      }else{
        circle(img, cvpoint, 10, Scalar(0, 0, 255), 2);
      }
    }

    //is the point in an acceptable size range
    if (dots[n].size < 0.8f*avg_dot_size || dots[n].size > 1.4f*avg_dot_size) continue;
    //draw the black (size ok) circle
    //if (draw) circle(img, cvpoint, 16, Scalar(0, 0, 0), -1);

    //is the point in the bounding box
    if (!is_in_quadrilateral(dots[n].pt.x, dots[n].pt.y, box_x, box_y)) continue;
    //if (draw) circle(img, cvpoint, 12, Scalar(255, 0, 255), -1);

    //get the corresponding grid location from the image location
    image_to_grid((float)dots[n].pt.x, (float)dots[n].pt.y, grid_x, grid_y,img_to_grdx,img_to_grdy);
    //get the nearest integer dot location
    grid_ix = std::lround(grid_x);
    grid_iy = std::lround(grid_y);
    //is it within the acceptable distance from the expected location and in the desired grid area
    if (abs(grid_x - round(grid_x)) <= dot_tol && abs(grid_y - round(grid_y)) <= dot_tol &&
        grid_ix>=0 && grid_ix < num_fiducials_x && grid_iy>=0 && grid_iy < num_fiducials_y) {
      //save the point
      cvkeypoint.pt.x = grid_ix;
      cvkeypoint.pt.y = grid_iy;
      img_points.push_back(dots[n]);
      grd_points.push_back(cvkeypoint);
      if(draw){
        if(img.size().height>800){
          circle(img, cvpoint, 12, Scalar(0, 255, 0), 4);
        }else{
          circle(img, cvpoint, 6, Scalar(0, 255, 0), 2);
        }
      }
    }
  }//end dots loop

  //draw the expected locations
  if (draw) {
    float imgx, imgy;
    for (float i_x = 0; i_x < num_fiducials_x; i_x++) {
      for (float i_y = 0; i_y < num_fiducials_y; i_y++) {
        grid_to_image(i_x, i_y, imgx, imgy,grd_to_imgx,grd_to_imgy,img.size().width,img.size().height);
        cvpoint.x = imgx;
        cvpoint.y = imgy;
        std::stringstream dot_text;
        dot_text << "(" << (int)i_x << "," << (int)i_y << ")";
        if(img.size().height>800){
          putText(img, dot_text.str(), cvpoint + Point(20,20),
            FONT_HERSHEY_COMPLEX_SMALL, 1.5, Scalar(255,0,255), 1, cv::LINE_AA);
          circle(img, cvpoint, 5, Scalar(255, 0, 255), -1);
        }else{
          putText(img, dot_text.str(), cvpoint + Point(8,20),
            FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255), 1, cv::LINE_AA);
          circle(img, cvpoint, 3, Scalar(255, 0, 255), -1);
        }
      }
    }
  }
}

void create_bounding_box(std::vector<float> & box_x,
  std::vector<float> & box_y,
  const int_t num_fiducials_x,
  const int_t num_fiducials_y,
  const std::vector<scalar_t> & grd_to_imgx,
  const std::vector<scalar_t> & grd_to_imgy,
  const int_t img_w,
  const int_t img_h) {
  assert(box_x.size()==box_y.size());
  assert(box_x.size()==5);
  float xgrid, ygrid;
  //xmin, ymin point
  xgrid = - 0.5;
  ygrid = - 0.5;
  grid_to_image(xgrid, ygrid, box_x[0], box_y[0],grd_to_imgx,grd_to_imgy,img_w,img_h);
  //xmax, ymin point
  xgrid = num_fiducials_x -1 + 0.5;
  ygrid =  - 0.5;
  grid_to_image(xgrid, ygrid, box_x[1], box_y[1],grd_to_imgx,grd_to_imgy,img_w,img_h);
  //xmax, ymax point
  xgrid = num_fiducials_x  - 1 + 0.5;
  ygrid = num_fiducials_y  - 1 + 0.5;
  grid_to_image(xgrid, ygrid, box_x[2], box_y[2],grd_to_imgx,grd_to_imgy,img_w,img_h);
  //xmin, ymax point
  xgrid =  - 0.5;
  ygrid = num_fiducials_y - 1 + 0.5;
  grid_to_image(xgrid, ygrid, box_x[3], box_y[3],grd_to_imgx,grd_to_imgy,img_w,img_h);
  //close the loop
  box_x[4] = box_x[0];
  box_y[4] = box_y[0];
  DEBUG_MSG("create_bounding_box(): (" << box_x[0] << "," << box_y[0] << ") (" << box_x[1] << "," << box_y[1] << ") (" <<
    box_x[2] << "," << box_y[2] << ") (" << box_x[3] << "," << box_y[3] << ")");
}

void grid_to_image(const float & grid_x,
  const float & grid_y,
  float & img_x,
  float & img_y,
  const std::vector<scalar_t> & grd_to_imgx,
  const std::vector<scalar_t> & grd_to_imgy,
  const int_t img_w,
  const int_t img_h) {
  TEUCHOS_TEST_FOR_EXCEPTION(grd_to_imgx.size()!=6,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(grd_to_imgy.size()!=6,std::runtime_error,"");
  img_x = grd_to_imgx[0] + grd_to_imgx[1] * grid_x + grd_to_imgx[2] * grid_y + grd_to_imgx[3] * grid_x * grid_y + grd_to_imgx[4] * grid_x * grid_x + grd_to_imgx[5] * grid_y * grid_y;
  img_y = grd_to_imgy[0] + grd_to_imgy[1] * grid_x + grd_to_imgy[2] * grid_y + grd_to_imgy[3] * grid_x * grid_y + grd_to_imgy[4] * grid_x * grid_x + grd_to_imgy[5] * grid_y * grid_y;
  if(img_x<0) img_x = 0;
  if(img_x>img_w-1) img_x = img_w-1;
  if(img_y<0) img_y = 0;
  if(img_y>img_h-1) img_y = img_h-1;
}

//convert image locations to grid locations
void image_to_grid(const float & img_x,
  const float & img_y,
  float & grid_x,
  float & grid_y,
  const std::vector<scalar_t> & img_to_grdx,
  const std::vector<scalar_t> & img_to_grdy) {
  grid_x = img_to_grdx[0] + img_to_grdx[1] * img_x + img_to_grdx[2] * img_y + img_to_grdx[3] * img_x * img_y + img_to_grdx[4] * img_x * img_x + img_to_grdx[5] * img_y * img_y;
  grid_y = img_to_grdy[0] + img_to_grdy[1] * img_x + img_to_grdy[2] * img_y + img_to_grdy[3] * img_x * img_y + img_to_grdy[4] * img_x * img_x + img_to_grdy[5] * img_y * img_y;
}


//distance between two points
float dist2(KeyPoint pnt1, KeyPoint pnt2) {
  return (pnt1.pt.x - pnt2.pt.x)*(pnt1.pt.x - pnt2.pt.x) + (pnt1.pt.y - pnt2.pt.y)*(pnt1.pt.y - pnt2.pt.y);
}

//order three distances returns biggest to smallest
void order_dist3(std::vector<float> & dist,
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
bool is_in_quadrilateral(const float & x,
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


//reorder the keypoints into origin, xaxis, yaxis order
void reorder_keypoints(std::vector<KeyPoint> & keypoints) {
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

}
