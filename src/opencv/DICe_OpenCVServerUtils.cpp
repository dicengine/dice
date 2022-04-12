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

/*! \file  DICe_OpenCVServer.cpp
    \brief Utility server to perform OpenCV operations on an image, used by the GUI
*/

#include <DICe.h>
#include <DICe_ImageIO.h>
#include <DICe_OpenCVServerUtils.h>
#include <DICe_Parser.h>
#include <DICe_Calibration.h>
#include <DICe_CameraSystem.h>
#include <DICe_PointCloud.h>
//#include "opencv2/flann/miniflann.hpp"
#ifdef DICE_ENABLE_TRACKLIB
#include <tracklib.h>
#endif

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>


#include <cassert>
#include <numeric>

using namespace cv;
namespace DICe{

// parse the input string and return a Teuchos ParameterList
DICE_LIB_DLL_EXPORT
Teuchos::ParameterList parse_filter_string(int argc, char *argv[]){
  DEBUG_MSG("opencv_server::parse_filter_string():");
//  DEBUG_MSG("User specified " << argc << " arguments");
//  for(int_t i=0;i<argc;++i){
//    DEBUG_MSG(argv[i]);
//  }

  // test if these are params for tracklib
  std::string first_arg = argv[1];
  if(first_arg==opencv_server_filter_tracklib){
    TEUCHOS_TEST_FOR_EXCEPTION(argc<3,std::runtime_error,"");
    Teuchos::ParameterList tracklib_params;
    tracklib_params.set("use_tracklib",true);
    // assume the rest is key value pairs
    for(int i=2;i<argc;++i){
      std::string arg = argv[i++];
//      std::cout << " found parameter " << arg << std::endl;
      std::string value = argv[i];
//      std::cout << " value " << value << std::endl;
      if(std::isdigit(value[0])||value[0]=='-'){ // is the string a number?
        if(value.find('.')!=std::string::npos){ // is it a double
          tracklib_params.set(arg,std::strtod(value.c_str(),NULL));
        }else{ // it must be an integer
          tracklib_params.set(arg,std::atoi(value.c_str()));
        }
      }else if(value.find("true")!=std::string::npos){
        tracklib_params.set(arg,true);
      }else if(value.find("false")!=std::string::npos){ // test for bools
        tracklib_params.set(arg,false);
      }else{ // otherwise add a string parameter
        tracklib_params.set(arg,value);
      }
    }
    return tracklib_params;
  }

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
      if(filter_name=="none"||filter_name=="equalize_hist")  // save off an empty parameter list for filter:none
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

  // check for is_tracklib
  if(input_params.get<bool>("use_tracklib",false)){

    // split the parameters up into input and tracklib and call tracklib_driver
    Teuchos::RCP<Teuchos::ParameterList> file_params = Teuchos::rcp( new Teuchos::ParameterList());
    Teuchos::RCP<Teuchos::ParameterList> tracking_params = Teuchos::rcp( new Teuchos::ParameterList());
    if(input_params.isParameter("cine_file")){
      file_params->set("cine_file",input_params.get<std::string>("cine_file"));
      file_params->set("stereo_cine_file",input_params.get<std::string>("stereo_cine_file"));
      file_params->set("cine_ref_index",input_params.get<int>("cine_ref_index"));
      file_params->set("cine_start_index",input_params.get<int>("cine_start_index"));
      file_params->set("cine_preview_index",input_params.get<int>("cine_preview_index"));
      file_params->set("cine_skip_index",input_params.get<int>("cine_skip_index"));
      file_params->set("cine_end_index",input_params.get<int>("cine_end_index"));
    }else{
      file_params->set("video_file",input_params.get<std::string>("video_file"));
      file_params->set("stereo_video_file",input_params.get<std::string>("stereo_video_file"));
      file_params->set("video_ref_index",input_params.get<int>("video_ref_index"));
      file_params->set("video_start_index",input_params.get<int>("video_start_index"));
      file_params->set("video_preview_index",input_params.get<int>("video_preview_index"));
      file_params->set("video_skip_index",input_params.get<int>("video_skip_index"));
      file_params->set("video_end_index",input_params.get<int>("video_end_index"));
    }
    file_params->set("camera_system_file",input_params.get<std::string>("camera_system_file"));
//    file_params->set("display_file_left",input_params.get<std::string>("display_file_left"));
//    file_params->set("display_file_right",input_params.get<std::string>("display_file_right"));

    tracking_params->set("threshold_method",input_params.get<std::string>("threshold_method"));
    tracking_params->set("min_thresh_left",input_params.get<int>("min_thresh_left"));
    tracking_params->set("max_thresh_left",input_params.get<int>("max_thresh_left"));
    tracking_params->set("steps_thresh_left",input_params.get<int>("steps_thresh_left"));
    tracking_params->set("min_thresh_right",input_params.get<int>("min_thresh_right"));
    tracking_params->set("max_thresh_right",input_params.get<int>("max_thresh_right"));
    tracking_params->set("steps_thresh_right",input_params.get<int>("steps_thresh_right"));
    tracking_params->set("max_pt_density",input_params.get<double>("max_pt_density"));
    tracking_params->set("min_area",input_params.get<int>("min_area"));
    tracking_params->set("max_area",input_params.get<int>("max_area"));
    tracking_params->set("colocation_tol",input_params.get<double>("colocation_tol"));
    tracking_params->set("neighbor_radius",input_params.get<double>("neighbor_radius"));
    tracking_params->set("num_search_frames",input_params.get<int>("num_search_frames"));
    tracking_params->set("min_pts_per_track",input_params.get<int>("min_pts_per_track"));
    tracking_params->set("area_tol",input_params.get<double>("area_tol"));
    tracking_params->set("area_weight",input_params.get<double>("area_weight"));
    tracking_params->set("gray_tol",input_params.get<int>("gray_tol"));
    tracking_params->set("gray_weight",input_params.get<double>("gray_weight"));
    tracking_params->set("dist_weight",input_params.get<double>("dist_weight"));
    tracking_params->set("angle_tol",input_params.get<double>("angle_tol"));
    tracking_params->set("angle_weight",input_params.get<double>("angle_weight"));
    tracking_params->set("stereo_area_tol",input_params.get<double>("stereo_area_tol"));
    tracking_params->set("stereo_area_weight",input_params.get<double>("stereo_area_weight"));
    tracking_params->set("dist_from_epi_tol",input_params.get<double>("dist_from_epi_tol"));
    tracking_params->set("dist_from_epi_weight",input_params.get<double>("dist_from_epi_weight"));
    tracking_params->set("num_background_frames",input_params.get<int>("num_background_frames"));
    tracking_params->set("show_segmentation",input_params.get<bool>("show_segmentation",false));
    tracking_params->set("write_results",input_params.get<bool>("write_results",true));
    tracking_params->set("preview_mode",true);

#ifdef DICE_ENABLE_TRACKLIB
    error_code = TrackLib::tracklib_driver(file_params,tracking_params);
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"tracklib_driver only available when DICe is built with tracklib on");
#endif
    return error_code;
  }

  Teuchos::ParameterList io_files = input_params.get<Teuchos::ParameterList>(DICe::opencv_server_io_files,Teuchos::ParameterList());
  Teuchos::ParameterList filters = input_params.get<Teuchos::ParameterList>(DICe::opencv_server_filters,Teuchos::ParameterList());

  // if the input and output files are the same path then get the dimensions and exit
  std::string first_image_in = io_files.begin()->first;
  std::string first_image_out = io_files.get<std::string>(io_files.begin()->first);
  DEBUG_MSG("opencv_server(): first image in " << first_image_in << " first_image_out " << first_image_out);
  if(first_image_in==first_image_out){
    DEBUG_MSG("opencv_server(): input and out image files are the same, reading dimensions and exiting");
    int width=0, height=0;
    DICe::utils::read_image_dimensions(first_image_in.c_str(),width,height);
    BUFFER_MSG("IMAGE_WIDTH",width);
    BUFFER_MSG("IMAGE_HEIGHT",height);
    return error_code;
  }

  // if the background filter is active, create a background image to pass to subsequent filters:
  Mat background_img; // empty if no background image is available
  for(Teuchos::ParameterList::ConstIterator filter_it=filters.begin();filter_it!=filters.end();++filter_it){
    if(filter_it->first == "background"){
      Teuchos::ParameterList options = filters.get<Teuchos::ParameterList>(filter_it->first,Teuchos::ParameterList());
      //if(options.get<int>(opencv_server_background_num_frames,1)<=1) break; // zero means don't do background subtraction
      opencv_create_cine_background_image(options);
      const std::string background_file = options.get<std::string>(opencv_server_background_file_name); // the check that this param exists happens in function above
//      background_img = imread(background_file, IMREAD_GRAYSCALE);
      background_img = DICe::utils::read_image(background_file.c_str());
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
      img = DICe::utils::read_image(image_in_filename.c_str()); // TODO if it's a filtered image it might have color annotations
//    img = imread(image_in_filename, IMREAD_COLOR); // if it's a filtered image it might have color annotations
    else
      img = DICe::utils::read_image(image_in_filename.c_str());
//      img = imread(image_in_filename, IMREAD_GRAYSCALE);
    if(img.empty()){
      std::cout << "*** error, the image is empty" << std::endl;
      return 4;
    }
    if(!img.data){
      std::cout << "*** error, the image failed to load" << std::endl;
      return 4;
    }
    if(file_it==io_files.begin()){
      BUFFER_MSG("IMAGE_WIDTH",img.cols);
      BUFFER_MSG("IMAGE_HEIGHT",img.rows);
      // if the input and the output images are the same, return without writing an output image
      if(image_in_filename.compare(image_out_filename) == 0){
        return error_code;
      }
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
      else if(filter==opencv_server_filter_equalize_hist){
        cv::equalizeHist(img,img);
      }
      else if(filter==opencv_server_filter_brightness){
        int brightness = options.get<int_t>("brightness",0);
        DEBUG_MSG("opencv_server(): adjsting brightness using value " << brightness);
        img.convertTo(img,-1,1,brightness);
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
      }
      else{
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
  if(block_size%2==0) block_size++; // block size has to be odd
  DEBUG_MSG("option, block size:      " << block_size);
  double binary_constant = options.get<double>(opencv_server_binary_constant,100.0);
  DEBUG_MSG("option, binary constant: " << binary_constant);
  adaptiveThreshold(img,img,255.0,filter_mode,threshold_mode,block_size,binary_constant);
  return 0;
}

DICE_LIB_DLL_EXPORT
int_t opencv_create_cine_background_image(Teuchos::ParameterList & options){
  TEUCHOS_TEST_FOR_EXCEPTION(!options.isParameter("video_file"),std::runtime_error,"");
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
  if(num_frames_to_avg < 1){
    std::cout << "invalid number of background frames (must be >=1)" << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"invalid num background frames");
  }

  // read the first cine frame in the file
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  params->set(filter_failed_cine_pixels,true);
  params->set(convert_cine_to_8_bit,true);
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
    cv::line(img,ptX,ptY,Scalar(0,0,255),1);
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
  if(!options.isParameter(opencv_server_min_blob_size))
    options.set<int_t>(opencv_server_min_blob_size,100);
  Mat img_cpy = img.clone();
  int_t error_code = opencv_dot_targets(img,options,key_points,img_points,grd_points, return_thresh);
  if(error_code!=0){
    DEBUG_MSG("opencv_dot_targets(): resetting the min_blob_size to 10 and trying again.");
    options.set<int_t>(opencv_server_min_blob_size,10);
    // reset the image in case it got annotated with dots, etc in the last step
    img = img_cpy.clone();
    error_code = opencv_dot_targets(img,options,key_points,img_points,grd_points, return_thresh);
//    if(error_code!=0){
//      DEBUG_MSG("opencv_dot_targets(): resetting the min_blob_size to 500 and trying again.");
//      options.set<int_t>(opencv_server_min_blob_size,500);
//      error_code = opencv_dot_targets(img,options,key_points,img_points,grd_points, return_thresh);
//    }
  }
  return error_code;
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

#ifdef DICE_DEBUG_MSG
  std::cout << "--- opencv_dot_targets(): options:" << std::endl;
  options.print(std::cout);
#endif

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
  int_t block_size = options.get<int_t>(opencv_server_block_size,75); // The old method had default set to 75
  if(block_size%2==0) block_size++; // block size has to be odd
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
  int min_blob_size = options.get<int_t>(opencv_server_min_blob_size,100);
  DEBUG_MSG("using min_blob_size: " << min_blob_size);
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
  //const double dot_tol = options.get<double>(opencv_server_dot_tol,0.25);

  // find the keypoints in the image
  options.set<bool>("donut_test",true);
  bool keypoints_found = true;
  //try to find the keypoints at different thresholds
  int_t i_thresh_first = 0;
  int_t i_thresh_last = 0;
  int_t i_thresh = threshold_start;
  if(threshold_start!=threshold_end){
    for (; i_thresh <= threshold_end; i_thresh += threshold_step) {
      // get the dots using an inverted image to get the donut holes
      get_dot_markers(img_cpy, key_points, i_thresh, invert,options,min_blob_size);
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
    i_thresh -= i_thresh%threshold_step; // truncate to the nearest thresh step to ensure that the final thresh is one that we know produced 3 keypoints
  }

  // get the key points at the average threshold value
  get_dot_markers(img_cpy, key_points, i_thresh, invert,options,min_blob_size);

  // it is possible that this threshold does not have 3 points.
  // chances are that this indicates p thresholding problem to begin with
  if (key_points.size() != 3) {
    //std::cout << "*** warning: unable to identify three keypoints, resetting the threshold" << std::endl;
    keypoints_found = false;
  }
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

  // copy the image into the output image
  cvtColor(img, img, cv::COLOR_GRAY2RGB);
  for (int_t n = 0; n < 3; n++) {
    if(img.size().height>800){
      circle(img, key_points[n].pt, 15, Scalar(0, 255, 255), 4);
    }else{
      circle(img, key_points[n].pt, 5, Scalar(0, 255, 255), 4);
    }
  }
  // report the results
  //  std::cout << "opencv_dot_targets():     using threshold: " << i_thresh << std::endl;
  return_thresh = i_thresh;

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
  const int_t slope = (yend - ystart)/(xend - xstart);
  maxgray = img_cpy.at<uchar>(ystart, xstart);
  mingray = maxgray;
  for (int_t ix = xstart; ix <= xend; ix++) {
    const int_t iy = ystart + (ix-xstart)*slope;
    const int_t curgray = img_cpy.at<uchar>(iy, ix);
    if (maxgray < curgray) maxgray = curgray;
    if (mingray > curgray) mingray = curgray;
  }
  i_thresh = (maxgray + mingray) / 2;
  DEBUG_MSG("  min gray value (inside target keypoints): " << mingray << " max gray value: " << maxgray);
  DEBUG_MSG("  getting the rest of the dots using average gray intensity value as threshold");
  DEBUG_MSG("    threshold to get dots: " << i_thresh);


  // get the rest of the dots
  options.set<bool>("donut_test",false);
  std::vector<KeyPoint> dots;
  get_dot_markers(img_cpy, dots, i_thresh, !invert,options,min_blob_size);
  const int_t num_dots = dots.size();
  DEBUG_MSG("    prospective grid points found: " << num_dots);
  if(num_dots <= 0){
    std::cout << "opencv_dot_targets(): zero dots found" << std::endl;
    return 2;
  }
  for (int_t n = 0; n < num_dots; n++) {
    if(img.size().height>800){
      circle(img, dots[n].pt, 5, Scalar(0, 0, 255), -1);
    }else{
      circle(img, dots[n].pt, 5, Scalar(0, 0, 255), -1);
    }
  }
  // TODO filter the dot markers (for example by size)

  // reorder the keypoints into an origin, xaxis, yaxis order
  reorder_keypoints(key_points,dots);
  DEBUG_MSG("    ordered keypoints: ");
  for (size_t i = 0; i < key_points.size(); ++i){ //save and display the keypoints
    DEBUG_MSG("      keypoint: " << key_points[i].pt.x << " " << key_points[i].pt.y);
  }
  img_points.clear();
  grd_points.clear();

  Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
  point_cloud->pts.resize(num_dots + 3); // add three for the keypoints

  std::vector<int_t> dot_use_count(num_dots + 3,0); // keep track of how many times a dot is used to ensure no repeats
  for(int_t i=0;i<num_dots;++i){
    point_cloud->pts[i].x = dots[i].pt.x;
    point_cloud->pts[i].y = dots[i].pt.y;
  }
  for(int_t i=0;i<3;++i){
    point_cloud->pts[num_dots + i].x = key_points[i].pt.x;
    point_cloud->pts[num_dots + i].y = key_points[i].pt.y;
    dots.push_back(key_points[i]);
  }
  std::vector<scalar_t> query(2,0.0);
  std::vector<size_t> indices(1,0);
  std::vector<scalar_t> dists(1,0.0);
  DEBUG_MSG("building the kd-tree");
  Teuchos::RCP<kd_tree_2d_t> kd_tree = Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  kd_tree->buildIndex();
  DEBUG_MSG("kd-tree completed");

  // create a grid for storing dots linked to the grid location

  const int_t num_grid_points = num_fiducials_x * num_fiducials_y;
  std::vector<KeyPoint> dot_grid(num_grid_points,cv::KeyPoint(cv::Point2f(-1.0,-1.0),0));

  // compute the preliminary search vectors
  assert(num_fiducials_origin_to_x_marker>1);
  assert(num_fiducials_origin_to_y_marker>1);
  float udx = (key_points[1].pt.x - key_points[0].pt.x)/(num_fiducials_origin_to_x_marker - 1);
  float udy = (key_points[1].pt.y - key_points[0].pt.y)/(num_fiducials_origin_to_x_marker - 1);
  const float vdx = (key_points[2].pt.x - key_points[0].pt.x)/(num_fiducials_origin_to_y_marker - 1);
  const float vdy = (key_points[2].pt.y - key_points[0].pt.y)/(num_fiducials_origin_to_y_marker - 1);
  const float dist_threshold = (0.25*udx*udx + 0.25*udy*udy);
//  float dist_threshold = (0.9*udx*0.9*udx + 0.9*udy*0.9*udy);

  // traverse x-axis, prior to and including the origin, points may not be in the F.O.V
  for(int_t i=0;i<=origin_loc_x;++i){
    query[0] = (i - origin_loc_x)*udx + key_points[0].pt.x;
    query[1] = (i - origin_loc_x)*udy + key_points[0].pt.y;
    kd_tree->knnSearch(&query[0],1,&indices[0],&dists[0]);
    if(query[0]<20.0||query[0]>img.size().width-20.0||dists[0] >= dist_threshold){
      if(query[0]<20.0||query[0]>img.size().width-20.0){
        DEBUG_MSG("opencv_dot_targets(): query point (on x-axis) " << query[0] << " " << query[1] << " distance " << dists[0] << " is too close to the boundary so skipping it.");
      }else{
        DEBUG_MSG("opencv_dot_targets(): query point (on x-axis) " << query[0] << " " << query[1] << " distance " << dists[0] << " is too far from any dots found so skipping it.");
      }
      continue;
    }
    DEBUG_MSG("opencv_dot_targets(): query point (on x-axis) " << query[0] << " " << query[1] << " found x-axis dot (" << point_cloud->pts[indices[0]].x <<"," << point_cloud->pts[indices[0]].y << ") distance " << dists[0] << " threshold " << dist_threshold);
    if(dot_grid[origin_loc_y*num_fiducials_x + i].pt.x>0.0){
      std::cout << "*** warning: attempting to associate dot with grid point that already has a dot associated with it" << std::endl;
      std::cout << "             dot at " << dots[indices[0]].pt.x << "  " << dots[indices[0]].pt.y << std::endl;
      return 2;
    }
    dot_grid[origin_loc_y*num_fiducials_x + i] = dots[indices[0]];
    dot_use_count[indices[0]]++;
    if(img.size().height>800){
      circle(img, dots[indices[0]].pt, 20, Scalar(0, 255, 0), 4);
    }else{
      circle(img, dots[indices[0]].pt, 10, Scalar(0, 255, 0), 4);
    }
  }
  // traverse from the origin along the x-axis using the the trajector from the last point as an initial guess
  if(dot_grid[origin_loc_y*num_fiducials_x + origin_loc_x].pt.x<0){
    DEBUG_MSG("opencv_dot_targets(): ***could not initialize x-axis dots (no dots found at origin or to the left of origin)");
    return 2;
  }
  for(int_t i=origin_loc_x+1;i<num_fiducials_x;++i){
    if(dot_grid[origin_loc_y*num_fiducials_x + i - 1].pt.x <=0){
      DEBUG_MSG("opencv_dot_targets(): query point (on x-axis) preivous point failed so skipping this one");
      break;
    }
    if(i==origin_loc_x+1){
      query[0] = dot_grid[origin_loc_y*num_fiducials_x + origin_loc_x].pt.x + udx;
      query[1] = dot_grid[origin_loc_y*num_fiducials_x + origin_loc_x].pt.y + udy;
    }else{
      query[0] = 2.0*dot_grid[origin_loc_y*num_fiducials_x + i-1].pt.x - dot_grid[origin_loc_y*num_fiducials_x + i-2].pt.x;
      query[1] = 2.0*dot_grid[origin_loc_y*num_fiducials_x + i-1].pt.y - dot_grid[origin_loc_y*num_fiducials_x + i-2].pt.y;
    }
    // traverse the x-axis locating points
    kd_tree->knnSearch(&query[0],1,&indices[0],&dists[0]);
    if(dists[0] >= dist_threshold){
      DEBUG_MSG("opencv_dot_targets(): query point (on x-axis) " << query[0] << " " << query[1] << " distance " << dists[0] << " is too far from any dots found so skipping it.");
      continue;
    }
    DEBUG_MSG("opencv_dot_targets(): query point (on x-axis) " << query[0] << " " << query[1] << " found x-axis dot (" << point_cloud->pts[indices[0]].x <<"," << point_cloud->pts[indices[0]].y << ") distance " << dists[0] << " threshold " << dist_threshold);
    if(dot_grid[origin_loc_y*num_fiducials_x + i].pt.x>0.0){
      std::cout << "*** warning: attempting to associate dot with grid point that already has a dot associated with it" << std::endl;
      std::cout << "             dot at " << dots[indices[0]].pt.x << "  " << dots[indices[0]].pt.y << std::endl;
      return 2;
    }
    dot_grid[origin_loc_y*num_fiducials_x + i] = dots[indices[0]];
    dot_use_count[indices[0]]++;
    if(img.size().height>800){
      circle(img, dots[indices[0]].pt, 20, Scalar(0, 255, 0), 4);
    }else{
      circle(img, dots[indices[0]].pt, 10, Scalar(0, 255, 0), 4);
    }
  }
  // traverse one level up in y from the x-axis using vdx
  for(int_t i=0;i<num_fiducials_x;++i){
    // use the starting point in the previous row and add v
    if(dot_grid[origin_loc_y*num_fiducials_x + i].pt.x<0){ // ensure the point is valid
      DEBUG_MSG("opencv_dot_targets(): preivous point failed so skipping this one");
      continue;
    }
    query[0] = dot_grid[origin_loc_y*num_fiducials_x + i].pt.x + vdx;
    query[1] = dot_grid[origin_loc_y*num_fiducials_x + i].pt.y + vdy;
    // find the closest dot to this point
    kd_tree->knnSearch(&query[0],1,&indices[0],&dists[0]);
    DEBUG_MSG("opencv_dot_targets(): query point (one row up from x-axis) " << query[0] << " " << query[1] << " found x-axis dot (" << point_cloud->pts[indices[0]].x <<"," << point_cloud->pts[indices[0]].y << ") distance " << dists[0] << " threshold " << dist_threshold);
    if(dists[0] < dist_threshold){
      if(dot_grid[(origin_loc_y+1)*num_fiducials_x + i].pt.x>0.0){
        std::cout << "*** warning: attempting to associate dot with grid point that already has a dot associated with it" << std::endl;
        std::cout << "             dot at " << dots[indices[0]].pt.x << "  " << dots[indices[0]].pt.y << std::endl;
        return 2;
      }
      dot_grid[(origin_loc_y+1)*num_fiducials_x + i] = dots[indices[0]];
      dot_use_count[indices[0]]++;
      if(img.size().height>800){
        circle(img, dots[indices[0]].pt, 20, Scalar(255, 0, 0), 4);
      }else{
        circle(img, dots[indices[0]].pt, 10, Scalar(255, 0, 0), 4);
      }
    }
    else
      DEBUG_MSG("opencv_dot_targets(): skipping point due to distance being too large");
  }
  // now that there are two rows to work with, use these rows sequentially to predict the next several rows
  for(int_t j=origin_loc_y+2;j<num_fiducials_y;++j){
    for(int_t i=0;i<num_fiducials_x;++i){
      // use the starting point in the previous row and add v
      if(dot_grid[(j-1)*num_fiducials_x + i].pt.x<0||dot_grid[(j-2)*num_fiducials_x + i].pt.x<0){ // ensure the point is valid
        DEBUG_MSG("opencv_dot_targets(): preivous point failed so skipping this one");
        continue;
      }
      // compute the localized projection to the next point
      query[0] = 2.0*dot_grid[(j-1)*num_fiducials_x + i].pt.x - dot_grid[(j-2)*num_fiducials_x + i].pt.x;
      query[1] = 2.0*dot_grid[(j-1)*num_fiducials_x + i].pt.y - dot_grid[(j-2)*num_fiducials_x + i].pt.y;

      // find the closest dot to this point
      kd_tree->knnSearch(&query[0],1,&indices[0],&dists[0]);
      DEBUG_MSG("opencv_dot_targets(): query point (above x-axis) " << query[0] << " " << query[1] << " found x-axis dot (" << point_cloud->pts[indices[0]].x <<"," << point_cloud->pts[indices[0]].y << ") distance " << dists[0] << " threshold " << dist_threshold);
      if(dists[0] < dist_threshold){
        if(dot_grid[j*num_fiducials_x + i].pt.x>0.0){
          std::cout << "*** warning: attempting to associate dot with grid point that already has a dot associated with it" << std::endl;
          std::cout << "             dot at " << dots[indices[0]].pt.x << "  " << dots[indices[0]].pt.y << std::endl;
          return 2;
        }
        dot_grid[j*num_fiducials_x + i] = dots[indices[0]];
        dot_use_count[indices[0]]++;
        if(img.size().height>800){
          circle(img, dots[indices[0]].pt, 20, Scalar(225, 225, 50), 4);
        }else{
          circle(img, dots[indices[0]].pt, 10, Scalar(225, 225, 50), 4);
        }
      }
      else
        DEBUG_MSG("opencv_dot_targets(): skipping point due to distance being too large");
    }
  }

  // traverse down from the x axis
  for(int_t j=origin_loc_y-1;j>=0;--j){
    for(int_t i=0;i<num_fiducials_x;++i){
      // use the starting point in the previous row and add v
      if(dot_grid[(j+1)*num_fiducials_x + i].pt.x<0||dot_grid[(j+2)*num_fiducials_x + i].pt.x<0){ // ensure the point is valid
        DEBUG_MSG("opencv_dot_targets(): preivous point failed so skipping this one");
        continue;
      }
      // compute the localized projection to the next point
      query[0] = 2.0*dot_grid[(j+1)*num_fiducials_x + i].pt.x - dot_grid[(j+2)*num_fiducials_x + i].pt.x;
      query[1] = 2.0*dot_grid[(j+1)*num_fiducials_x + i].pt.y - dot_grid[(j+2)*num_fiducials_x + i].pt.y;
      // find the closest dot to this point
      kd_tree->knnSearch(&query[0],1,&indices[0],&dists[0]);
      DEBUG_MSG("opencv_dot_targets(): query point (below x-axis) " << query[0] << " " << query[1] << " found x-axis dot (" << point_cloud->pts[indices[0]].x <<"," << point_cloud->pts[indices[0]].y << ") distance " << dists[0] << " threshold " << dist_threshold);
      if(dists[0] < dist_threshold){
        if(dot_grid[j*num_fiducials_x + i].pt.x>0.0){
          std::cout << "*** warning: attempting to associate dot with grid point that already has a dot associated with it" << std::endl;
          std::cout << "             dot at " << dots[indices[0]].pt.x << "  " << dots[indices[0]].pt.y << std::endl;
          return 2;
        }
        dot_grid[j*num_fiducials_x + i] = dots[indices[0]];
        dot_use_count[indices[0]]++;
        if(img.size().height>800){
          circle(img, dots[indices[0]].pt, 20, Scalar(255, 0, 255), 4);
        }else{
          circle(img, dots[indices[0]].pt, 10, Scalar(255, 0, 255), 4);
        }
      }
      else
        DEBUG_MSG("opencv_dot_targets(): skipping point due to distance being too large");
    }
  }

//  static int img_counter = 0;
//  img_counter++;
//  std::stringstream filename;
//  filename << "debug_box_" << img_counter << ".png";
//  cv::imwrite(filename.str(),img);

  // assemble the img_points and grd_points vectors
  for(int_t j=0;j<num_fiducials_y;++j){
    for(int_t i=0;i<num_fiducials_x;++i){
      // don't include the failed points
      if(dot_grid[j*num_fiducials_x+i].pt.x<0) continue;
      // don't include the corner key points (marker dots)
      if((j==origin_loc_y&&i==origin_loc_x)||
          (j==origin_loc_y&&i==origin_loc_x+num_fiducials_origin_to_x_marker-1)||
          (j==origin_loc_y+num_fiducials_origin_to_y_marker-1&&i==origin_loc_x)) continue;
      img_points.push_back(dot_grid[j*num_fiducials_x+i]);
      grd_points.push_back(cv::KeyPoint(cv::Point2f(i,j),0));
      std::stringstream dot_text;
      dot_text << "(" << i << "," << j << ")";
      if(img.size().height>800){
        putText(img, dot_text.str(), dot_grid[j*num_fiducials_x+i].pt + Point2f(20.0,20.0),
          FONT_HERSHEY_COMPLEX_SMALL, 1.5, Scalar(255,0,255), 1, cv::LINE_AA);
      }else{
        putText(img, dot_text.str(), dot_grid[j*num_fiducials_x+i].pt + Point2f(20.0,20.0),
          FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255), 1, cv::LINE_AA);
      }
    }
  }

//  static int img_counter = 0;
//  img_counter++;
//  std::stringstream filename;
//  filename << "debug_box_" << img_counter << ".png";
//  cv::imwrite(filename.str(),img);

  // save the information about the found dots
  for(int_t i=0;i<num_dots+3;++i){
    if(dot_use_count[i]>1){
      std::cout << "*** warning: detected dot being associated with multiple grid points, dot extraction failed." << std::endl;
      std::cout << "             dot at " << dots[i].pt.x << " " << dots[i].pt.y << std::endl;
      return 2;
    }
  }
  std::cout << "opencv_dot_targets():     good dots identified: " << img_points.size() << std::endl;
  if(img_points.size() < num_fiducials_x*num_fiducials_y*0.70){ // TODO fix this hard coded tolerance
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
  Teuchos::ParameterList & options,
  const int min_size) {
  DEBUG_MSG("get_dot_markers(): thresh " << thresh << " invert " << invert);
  int_t block_size = options.get<int_t>(opencv_server_block_size,75); // The old method had default set to 75
  if(block_size%2==0) block_size++; // block size has to be odd
  DEBUG_MSG("option, block size:        " << block_size);
  const bool use_adaptive = options.get<bool>(opencv_server_use_adaptive_threshold,false);
  DEBUG_MSG("option, use adaptive:      " << use_adaptive);
  int filter_mode = options.get<int_t>(opencv_server_filter_mode,1);
  DEBUG_MSG("option, filter mode:       " << filter_mode);
  const bool donut_test = options.get<bool>("donut_test",false);
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
  DEBUG_MSG("min blob size:             " << min_size);

  // setup the blob detector
  SimpleBlobDetector::Params params;
  params.filterByArea = true;
  params.maxArea = 10e4;
  params.minArea = min_size;
//  params.filterByInertia = true;
//  params.minInertiaRatio = 0.75;
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
    img.copyTo(bi_src);
  }else{
    threshold(timg, bi_src, thresh, 255, threshold_mode);
  }

  // invert the source image
  Mat not_src(bi_src.size(), bi_src.type());
  bitwise_not(bi_src, not_src);

  // detect dots on the appropriately inverted image
  Mat labelImage(img.size(),CV_32S);
  Mat stats, centroids;
  int nLabels = 0;

  // detect dots on the appropriately inverted image
  if (invert){
    detector->detect(not_src, keypoints);
    nLabels = connectedComponentsWithStats(bi_src, labelImage, stats, centroids, 8, CV_32S);
  }else{
    detector->detect(bi_src, keypoints);
    nLabels = connectedComponentsWithStats(not_src, labelImage, stats, centroids, 8, CV_32S);
  }
  DEBUG_MSG("get_dot_markers(): num keypoints " << keypoints.size());
  float avg_size = 0.0f;
  for(size_t i=0;i<keypoints.size();++i){
    DEBUG_MSG("Keypoint: " << i << " " << keypoints[i].pt.x << " " << keypoints[i].pt.y << " blob size " << keypoints[i].size);
    avg_size += keypoints[i].size;
  }
  if(keypoints.size()>0)
    avg_size /= keypoints.size();

  // filter out blobs by size if not searching for keypoints
  if(!donut_test){
    size_t i = keypoints.size();
    while(i--){
      if(std::abs(keypoints[i].size-avg_size)/avg_size > 0.5){
        DEBUG_MSG("get_dot_markers(): erasing keypoing at " << keypoints[i].pt.x << " " << keypoints[i].pt.y << " due to keypoint size >50% difference in diameter from avg");
        keypoints.erase(keypoints.begin() + i);
      }
    }
  }

//  cvtColor(not_src, not_src, cv::COLOR_GRAY2RGB);
//  for(size_t i=0;i<keypoints.size();++i)
//    circle(not_src, keypoints[i].pt, 20, Scalar(0,0,255), 4);
//  static int img_counter = 0;
//  img_counter++;
//  std::stringstream filename;
//  filename << "keypoints_" << img_counter << ".png";
//  cv::imwrite(filename.str(),not_src);

  if(keypoints.size()==0) return;

  // test for donut shape of the keypoints by looking at the labels of the connected components
  // if there are more than 2 connected components then it's probably a donut
  if(donut_test){
    const int_t donut_span = 2 * avg_size;
    DEBUG_MSG("get_dot_markers(): donut span: " << donut_span);
    size_t i = keypoints.size();
    while(i--){
      // check that the end points have the same zone
      if(labelImage.at<int>(keypoints[i].pt.y,keypoints[i].pt.x - donut_span)!=labelImage.at<int>(keypoints[i].pt.y,keypoints[i].pt.x + donut_span)){
        DEBUG_MSG("get_dot_markers(): erasing keypoing at " << keypoints[i].pt.x << " " << keypoints[i].pt.y << " because the endpoints of the donut span are not in the same connected component");
        keypoints.erase(keypoints.begin() + i);
        continue;
      }
      std::set<int_t> zones;
      for(int_t j=keypoints[i].pt.x - donut_span;j<keypoints[i].pt.x+donut_span;++j){
        if(j>=0&&j<img.size().width){
          zones.insert(labelImage.at<int>(keypoints[i].pt.y,j));
          //std::cout << " keypoint " << i << " has zone " << labelImage.at<int>(keypoints[i].pt.y,j) << " size " << zones.size() << std::endl;
        }
      }
      if(zones.size()!=3){
        DEBUG_MSG("get_dot_markers(): erasing keypoing at " << keypoints[i].pt.x << " " << keypoints[i].pt.y << " because num zones is " << zones.size() << " not 3");
        keypoints.erase(keypoints.begin() + i);
      }
    }
  }

  if(keypoints.size()==3&&nLabels>=3){
    float avg_diameter = 0.0f;
    for(size_t i=0;i<keypoints.size();++i){
      int x = keypoints[i].pt.x;
      int y = keypoints[i].pt.y;
      DEBUG_MSG("Keypoint: " << i << " " << keypoints[i].pt.x << " " << keypoints[i].pt.y << " blob size " << keypoints[i].size);
      int label = labelImage.at<int>(y,x);
      DEBUG_MSG("label " << label);
      int psize = stats.at<int>(label,CC_STAT_AREA);
      DEBUG_MSG("new size " << psize);
      keypoints[i].size = psize;
      DEBUG_MSG("Keypoint: " << i << " updated size " << keypoints[i].size);
      avg_diameter += keypoints[i].size;
    }
    avg_diameter /= keypoints.size();
    DEBUG_MSG("get_dot_markers(): avg keypoint diameter " << avg_diameter);
    size_t i = keypoints.size();
    while (i--) {
      // remove the keypoint from the vector
      if(keypoints[i].size<=0.0){
        DEBUG_MSG("get_dot_markers(): removing keypoint " << i << " due to keypoint size == 0.0");
        keypoints.erase(keypoints.begin() + i);
        continue;
      }
      if(std::abs(keypoints[i].size-avg_diameter)/avg_diameter>2.0){
        DEBUG_MSG("get_dot_markers(): removing keypoint " << i << " due to keypoint size >200% difference in diameter from avg");
        keypoints.erase(keypoints.begin() + i);
      }
    }
    DEBUG_MSG("get_dot_markers(): num keypoints " << keypoints.size());
  }
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
  assert(dist_order.size()==dist.size());
  std::iota(dist_order.begin(),dist_order.end(),0); //Initializing
  std::sort(dist_order.begin(),dist_order.end(), [&](int i,int j){return dist[i]>dist[j];} );
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

void reorder_keypoints(std::vector<KeyPoint> & keypoints, const std::vector<KeyPoint> & dots) {

  // compute the line coefficients for each side
  assert(keypoints.size()==3);
  // m = (y1 - y2)/(x1-x2), y = mx + b -> b = y - mx
  std::vector<float> m(3,0.0f);
  std::vector<float> b(3,0.0f);
  std::vector<float> count(3,0.0f); //holds the distances between the points
  std::vector<int_t> count_order(3,0); //index order of the distances max to min
  m[0] = (keypoints[1].pt.y - keypoints[2].pt.y)/(keypoints[1].pt.x - keypoints[2].pt.x);
  b[0] = keypoints[1].pt.y - m[0] * keypoints[1].pt.x;
  m[1] = (keypoints[0].pt.y - keypoints[2].pt.y)/(keypoints[0].pt.x - keypoints[2].pt.x);
  b[1] = keypoints[0].pt.y - m[1] * keypoints[0].pt.x;
  m[2] = (keypoints[0].pt.y - keypoints[1].pt.y)/(keypoints[0].pt.x - keypoints[1].pt.x);
  b[2] = keypoints[0].pt.y - m[2] * keypoints[0].pt.x;
  // compute the distances from each img_point to one of those lines
  const float dist_tol = 5.0;
  for(size_t i=0;i<3;++i){
    for(size_t j=0;j<dots.size();++j){
      const float dist = std::abs(-1.0*m[i]*dots[j].pt.x + dots[j].pt.y - b[i])/std::sqrt(m[i]*m[i]+1);
      //std::cout << "dot " << dots[j].pt.x << " " << dots[j].pt.y << " dist " << dist << std::endl;
      if(dist<dist_tol)
        count[i]+=1.0;
    }
  }
  order_dist3(count,count_order);
//  for(size_t i=0;i<3;++i){
//    std::cout << " count " << i << " " << count[i] << std::endl;
//  }
//  for(size_t i=0;i<3;++i){
//    std::cout << " count order " << i << " " << count_order[i] << std::endl;
//  }

  std::set<int_t> side0;
  side0.insert(1);
  side0.insert(2);
  std::set<int_t> side1;
  side1.insert(0);
  side1.insert(2);
  std::set<int_t> side2;
  side2.insert(0);
  side2.insert(1);

  std::vector<std::set<int_t> > sides;
  sides.push_back(side0);
  sides.push_back(side1);
  sides.push_back(side2);

  int_t i0 = 0; //common point between set one and two must be the origin

  for (std::set<int_t>::iterator it0 = sides[count_order[0]].begin(); it0 != sides[count_order[0]].end(); ++it0) {
    for (std::set<int_t>::iterator it1 = sides[count_order[1]].begin(); it1 != sides[count_order[1]].end(); ++it1) {
      if(*it0==*it1)
        i0 = *it0;
    }
  }

  DEBUG_MSG("reorder_keypoints(): origin is keypoint " << i0);

  // now remove the origin from the other two points to get the x and y axis
  sides[count_order[0]].erase(i0);
  sides[count_order[1]].erase(i0);
  // the leftover points are the end points
  int_t i1 = *sides[count_order[0]].begin();
  DEBUG_MSG("reorder_keypoints(): end of x-axis is " << i1);
  // small side is y-axis
  int_t i2 = *sides[count_order[1]].begin();
  DEBUG_MSG("reorder_keypoints(): end of y-axis is " << i2);
  assert(i1!=i2);
  assert(i0!=i1);
  assert(i0!=i2);
  //reorder the points and return
  std::vector<cv::KeyPoint> temp_points = keypoints;
  keypoints[0] = temp_points[i0];
  keypoints[1] = temp_points[i1];
  keypoints[2] = temp_points[i2];
}

}
