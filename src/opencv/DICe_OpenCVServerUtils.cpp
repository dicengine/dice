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
      if(filter_params.numParams()>0)
        filters.set(filter_name,filter_params);
      filter_params = Teuchos::ParameterList();
      filter_name = arg;
      continue;
    } // end found filter keyword
    if(name_value_flag)
      temp_name_string = argv[i];
    else{
      std::string arg_upper = arg;
      std::transform(arg_upper.begin(),arg_upper.end(),arg_upper.begin(),::toupper);
      if(std::isdigit(arg[0])){ // is the string a number?
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

  int_t error_code = 0;

  // parse the string and return error code if it is an empty parameterlist
  Teuchos::ParameterList input_params = parse_filter_string(argc,argv);
#ifdef DICE_DEBUG_MSG
  input_params.print(std::cout);
#endif
  Teuchos::ParameterList io_files = input_params.get<Teuchos::ParameterList>(DICe::opencv_server_io_files,Teuchos::ParameterList());
  Teuchos::ParameterList filters = input_params.get<Teuchos::ParameterList>(DICe::opencv_server_filters,Teuchos::ParameterList());
  // iterate the selected images
  for(Teuchos::ParameterList::ConstIterator file_it=io_files.begin();file_it!=io_files.end();++file_it){
    std::string image_in_filename = file_it->first;
    std::string image_out_filename = io_files.get<std::string>(file_it->first);
    DEBUG_MSG("Processing image: " << image_in_filename << " output image " << image_out_filename);
    // load the image as an openCV mat
    Mat img = imread(image_in_filename, IMREAD_GRAYSCALE);
    if(img.empty()){
      std::cout << "error, the image is empty" << std::endl;
      return -1;
    }
    if(!img.data){
      std::cout << "error, the image failed to load" << std::endl;
      return -1;
    }
    // iterate the selected filters
    for(Teuchos::ParameterList::ConstIterator filter_it=filters.begin();filter_it!=filters.end();++filter_it){
      std::string filter = filter_it->first;
      DEBUG_MSG("Applying filter: " << filter);
      Teuchos::ParameterList options = filters.get<Teuchos::ParameterList>(filter,Teuchos::ParameterList());
      // switch statement on the filters, pass the options as an argument and the opencv image mat
      if(filter==opencv_server_filter_binary_threshold){
        error_code = opencv_binary_threshold(img,options);
      }else if(filter==opencv_server_filter_adaptive_threshold){
        error_code = opencv_adaptive_threshold(img,options);
      }else if(filter==opencv_server_filter_checkerboard_targets){
        error_code = opencv_checkerboard_targets(img,options);
      }else if(filter==opencv_server_filter_dot_targets){
        error_code = opencv_dot_targets(img,options);
      }else{
        std::cout << "error, unknown filter: " << filter << std::endl;
        error_code = -1;
      }
    } // end filter iteration
    if(error_code!=-1){
      DEBUG_MSG("Writing output image: " << image_out_filename);
      imwrite(image_out_filename, img);
    }
  } // end file iteration

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
  int block_size = options.get<int_t>(opencv_server_block_size,3);
  DEBUG_MSG("option, block size:      " << block_size);
  double binary_constant = options.get<double>(opencv_server_binary_constant,100.0);
  DEBUG_MSG("option, binary constant: " << binary_constant);
  adaptiveThreshold(img,img,255.0,filter_mode,threshold_mode,block_size,binary_constant);
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
    std::cout << "error, missing checkerboard dimensions" << std::endl;
    return -1;
  }
  corners.clear();
  const int_t num_fiducials_x = options.get<int_t>(DICe::num_cal_fiducials_x);
  const int_t num_fiducials_y = options.get<int_t>(DICe::num_cal_fiducials_y);
  DEBUG_MSG("option, board size x:   " << num_fiducials_x);
  DEBUG_MSG("option, board size y:   " << num_fiducials_y);
  Size board_size; //used by openCV
  //set the height and width of the board in intersections
  board_size.width = num_fiducials_x;
  board_size.height = num_fiducials_y;
  // convert the image to color
  const bool found = findChessboardCorners(img, board_size, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
  if(found){
    DEBUG_MSG("found " << corners.size() << " checkerboard intersections");
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
    return -1;
  }
  return 0;
}

DICE_LIB_DLL_EXPORT
int_t opencv_dot_targets(cv::Mat & img, Teuchos::ParameterList & options){
  std::vector<cv::KeyPoint> key_points;
  std::vector<cv::KeyPoint> img_points;
  std::vector<cv::KeyPoint> grd_points;
  return opencv_dot_targets(img,options,key_points,img_points,grd_points);
}

DICE_LIB_DLL_EXPORT
int_t opencv_dot_targets(Mat & img, Teuchos::ParameterList & options,
  std::vector<cv::KeyPoint> & key_points,
  std::vector<cv::KeyPoint> & img_points,
  std::vector<cv::KeyPoint> & grd_points){

  DEBUG_MSG("opencv_dot_targets()");

  key_points.clear();
  img_points.clear();
  grd_points.clear();

  // clone the input image so that we have a copy of the original
  Mat img_cpy = img.clone();

  const int_t threshold_start = options.get<int_t>(opencv_server_threshold_start,20);
  DEBUG_MSG("option, threshold start:   " << threshold_start);
  const int_t threshold_end = options.get<int_t>(opencv_server_threshold_end,250);
  DEBUG_MSG("option, threshold end:     " << threshold_end);
  const int_t threshold_step = options.get<int_t>(opencv_server_threshold_step,5);
  DEBUG_MSG("option, threshold step:    " << threshold_step);
  const bool preview_thresh = options.get<bool>(opencv_server_preview_threshold,false);
  DEBUG_MSG("option, preview thresh:    " << preview_thresh);
  const int_t block_size = options.get<int_t>(opencv_server_block_size,3); // The old method had default set to 75
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
  DEBUG_MSG("option, threshold mode:   " << threshold_mode);

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
  int_t i_thresh = 0;
  for (i_thresh = threshold_start; i_thresh <= threshold_end; i_thresh += threshold_step) {
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

  // get the key points at the average threshold value
  get_dot_markers(img_cpy, key_points, i_thresh, invert,options);
  // it is possible that this threshold does not have 3 points.
  // chances are that this indicates p thresholding problem to begin with
  if (key_points.size() != 3) {
    DEBUG_MSG("warning: unable to identify three keypoints");
    DEBUG_MSG("         other points will not be extracted");
    keypoints_found = false;
  }
  if(!keypoints_found) return -1;

  // now that we have the keypoints try to get the rest of the dots

  // reorder the keypoints into an origin, xaxis, yaxis order
  reorder_keypoints(key_points);

  // report the results
  DEBUG_MSG("    threshold: " << i_thresh);
  DEBUG_MSG("    ordered keypoints: ");
  for (size_t i = 0; i < key_points.size(); ++i) //save and display the keypoints
    DEBUG_MSG("      keypoint: " << key_points[i].pt.x << " " << key_points[i].pt.y);

  Point cvpoint;
  if(preview_thresh){
    if(use_adaptive){
      adaptiveThreshold(img,img,255,filter_mode,threshold_mode,block_size,i_thresh);
    }else{
      // apply thresholding
      threshold(img,img,i_thresh,255,cv::THRESH_BINARY);
    }
  }
  // copy the image into the output image
  cvtColor(img, img, cv::COLOR_GRAY2RGB);
  for (int_t n = 0; n < 3; n++) {
    cvpoint.x = key_points[n].pt.x;
    cvpoint.y = key_points[n].pt.y;
    circle(img, cvpoint, 20, Scalar(0, 255, 255), 4);
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
  DEBUG_MSG("  getting the rest of the dots");
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
  DEBUG_MSG("    good dots identified: " << new_dot_num);
  DEBUG_MSG("    filter passes: " << filter_passes);

  return 0;
}

// get all the possible dot markers
void get_dot_markers(cv::Mat img,
  std::vector<KeyPoint> & keypoints,
  int_t thresh,
  bool invert,
  Teuchos::ParameterList & options) {
  DEBUG_MSG("get_dot_markers(): thresh " << thresh << " invert " << invert);
  const int_t block_size = options.get<int_t>(opencv_server_block_size,3); // The old method had default set to 75
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
    threshold(timg, bi_src, thresh, 255, cv::THRESH_BINARY);
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
      circle(img, cvpoint, 20, Scalar(0, 0, 255), 2);
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
        circle(img, cvpoint, 12, Scalar(0, 255, 0), 4);
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
        putText(img, dot_text.str(), cvpoint + Point(20,20),
          FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255,0,255), 1, cv::LINE_AA);
        circle(img, cvpoint, 5, Scalar(255, 0, 255), -1);
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


//
//  // filter parameters:
//  int filterMode = -1;
//  int invertedMode = -1;
//  bool cal_target_is_inverted = false;
//  //int antiInvertedMode = -1;
//  double blockSize = -1.0;
//  double binaryConstant = -1.0;
//  int board_width = 0;
//  int board_height = 0;
//  double pattern_spacing = 0.0;
//  SimpleBlobDetector::Params params;
//
//  // check if this is a calibration preview and also the filters applied:
//  bool is_cal = false;
//  //bool has_binary = false;
//  //bool has_blob = false;
//  bool preview_thresh = false;
//  bool has_adaptive = false;
//
//  // set up the filters
//  DEBUG_MSG("number of filters: " << filters.size());
//  if(filters.size()!=filter_params.size()){
//    std::cout << "error, filters vec and filter params vec are not the same size" << std::endl;
//    return -1;
//  }
//  for(size_t i=0;i<filters.size();++i){
//    DEBUG_MSG("filter: " << filters[i]);
//    for(size_t j=0;j<filter_params[i].size();++j)
//      DEBUG_MSG("    parameter: " << filter_params[i][j]);
//    if(filters[i]=="Filter:CalPreview"){
//      DEBUG_MSG("CalPreview filter is active");
//      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"not implemented yet");
//      is_cal = true;
//    }
//    if(filters[i]=="Filter:PreviewThreshold"){
//      DEBUG_MSG("Preview images will use thresholded image for background");
//      preview_thresh = true;
//    }
//    if(filters[i]=="Filter:AdaptiveThreshold"){
//      DEBUG_MSG("Preview images will use adaptive thresholding");
//      has_adaptive = true;
//    }
//    if(filters[i]=="Filter:BinaryThreshold"){
//      //has_binary = true;
//      if(filter_params[i].size()!=4){
//        std::cout << "error, the number of parameters for a binary filter should be 4, not " << filter_params[i].size() << std::endl;
//        return -1;
//      }
//      filterMode = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
//      if(filter_params[i][0]==1.0) filterMode = cv::ADAPTIVE_THRESH_MEAN_C;
//      invertedMode = cv::THRESH_BINARY;
//      //antiInvertedMode = cv::THRESH_BINARY_INV;
//      if(filter_params[i][3]==1.0){
//        invertedMode = cv::THRESH_BINARY_INV;
//        cal_target_is_inverted = true;
//        //antiInvertedMode = cv::THRESH_BINARY;
//      }
//      blockSize = filter_params[i][1];
//      binaryConstant = filter_params[i][2];
//    } // end binary filter
//    if(filters[i]=="Filter:BoardSize"){
//      if(filter_params[i].size()!=2){
//        std::cout << "error, the number of parameters for a board size should be 2, not " << filter_params[i].size() << std::endl;
//        return -1;
//      }
//      board_width = filter_params[i][0];
//      board_height = filter_params[i][1];
//      std::cout << "board size " << board_width << " x " << board_height << std::endl;
//    }
//    if(filters[i]=="Filter:PatternSpacing"){
//      if(filter_params[i].size()!=1){
//        std::cout << "error, the number of parameters for pattern spacing should be 1, not " << filter_params[i].size() << std::endl;
//        return -1;
//      }
//      pattern_spacing = filter_params[i][0];
//      //std::cout << "pattern spacing " << pattern_spacing << std::endl;
//    }
//    if(filters[i]=="Filter:Blob"){
//      //has_blob = true;
//      if(filter_params[i].size()!=12){
//        std::cout << "error, the number of parameters for a blob filter should be 12, not " << filter_params[i].size() << std::endl;
//        return -1;
//      }
//      // find the dots with holes as blobs
//      // area
//      if(filter_params[i][0]!=0.0){
//        params.filterByArea = true;
//        params.minArea = filter_params[i][1];
//        params.maxArea = filter_params[i][2];
//      }
//      // circularity
//      if(filter_params[i][3]!=0.0){
//        params.filterByCircularity = true;
//        params.minCircularity = filter_params[i][4];
//        params.maxCircularity = filter_params[i][5];
//      }
//      // eccentricity
//      if(filter_params[i][6]!=0.0){
//        params.filterByInertia = true;
//        params.minInertiaRatio = filter_params[i][7];
//        params.maxInertiaRatio = filter_params[i][8];
//      }
//      // convexity
//      if(filter_params[i][9]!=0.0){
//        params.filterByConvexity = true;
//        params.minConvexity = filter_params[i][10];
//        params.maxConvexity = filter_params[i][11];
//      }
//    } // end blob filter
//  } // end loop over filters
//
//  Teuchos::RCP<Teuchos::ParameterList> preview_params = rcp(new Teuchos::ParameterList());
//  preview_params->set("preview_thresh",preview_thresh);
//  preview_params->set("cal_target_has_adaptive",has_adaptive);
//  //preview_params->set("filterMode",filterMode);
//  preview_params->set("cal_target_is_inverted",cal_target_is_inverted);
//  //preview_params->set("antiInvertedMode",antiInvertedMode);
//  preview_params->set("cal_target_block_size",blockSize);
//  preview_params->set("cal_target_binary_constant",binaryConstant);
//  preview_params->set("cal_target_spacing_size",pattern_spacing);
//
//  for(size_t image_it=0;image_it<input_images.size();++image_it){
//    DEBUG_MSG("processing image " << input_images[image_it]);
//
//    if(is_cal){
////      if(!has_blob || !has_binary){
////        std::cout << "error, cal preview requires blob and binary filters to be active" << std::endl;
////        return -1;
////      }
////      std::vector<Point2f> image_points;
////      std::vector<Point3f> object_points;
////      Size imageSize;
////      // find the cal dot points and check show a preview of their locations
////      const int pre_code = pre_process_cal_image(input_images[image_it],output_images[image_it],preview_params,
////        image_points,object_points,imageSize);
////      DEBUG_MSG("pre_process_cal_image return value: " << pre_code);
////      if(pre_code==-1)
////        return -1;
////      else if(pre_code==-4) // image load failure
////        return 9;
////      else if(pre_code!=0){
////        if(image_it==0)
////          error_code = 2; // left alone
////        else if(image_it==1&&error_code==0)
////          error_code = 3; // right alone
////        else if(image_it==1&&error_code!=0)
////          error_code = 4; // left and right failed
////        else if(image_it==2&&error_code==0)
////          error_code = 5; // middle alone
////        else if(image_it==2&&error_code==2)
////          error_code = 6; // middle and left
////        else if(image_it==2&&error_code==3)
////          error_code = 7; // middle and right
////        else if(image_it==2&&error_code==4)
////          error_code = 8; // all
////        continue;
////      } // pre_code != 0
//    } // end is cal
//    else{
//      // load the image as an openCV mat
//      Mat img = imread(input_images[image_it], IMREAD_GRAYSCALE);
////      Mat binary_img(img.size(),CV_8UC3);
////      //Mat contour_img(img.size(), CV_8UC3,Scalar::all(255));
////      Mat out_img(img.size(), CV_8UC3);
////      Mat bi_copy_img(img.size(), CV_8UC3);
////      cvtColor(img, out_img, CV_GRAY2RGB);
//
//      //Mat out_img = imread(input_images[image_it], IMREAD_GRAYSCALE);
//      if(img.empty()){
//        std::cout << "error, the image is empty" << std::endl;
//        return -1;
//      }
//      if(!img.data){
//        std::cout << "error, the image failed to load" << std::endl;
//        return -1;
//      }
//      for(size_t filter_it=0;filter_it<filters.size();++filter_it){
//        DEBUG_MSG("processing filter " << filter_it);
//
//        if(filters[filter_it]=="Filter:BinaryThreshold"){
//          DEBUG_MSG("applying a binary threshold");
//          adaptiveThreshold(img,img,255,filterMode,invertedMode,blockSize,binaryConstant);
//          DEBUG_MSG("binary threshold successful");
//        } // end binary filter
//
//        if(filters[filter_it]=="Filter:Blob"){
//          DEBUG_MSG("applying a blob detector");
//          cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
//          std::vector<KeyPoint> keypoints;
//          detector->detect( img, keypoints );
//          drawKeypoints( img, keypoints, img, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
//          DEBUG_MSG("blob detector successful");
//        } // end blob filter
//
//      } // end filters
//      DEBUG_MSG("writing output image");
//      imwrite(output_images[image_it], img);
//    }
//
////    // pre-processing for CalPreview
////    if(is_cal){
////      if(!has_blob || !has_binary){
////        std::cout << "error, cal preview requires blob and binary filters to be active" << std::endl;
////        return -1;
////      }
////      // blur the image to remove noise
////      GaussianBlur(img, binary_img, Size(9, 9), 2, 2 );
////
////      // binary threshold to create black and white image
////      if(has_adaptive)
////        adaptiveThreshold(binary_img,binary_img,255,filterMode,invertedMode,blockSize,binaryConstant);
////      else
////        threshold(binary_img, binary_img, binaryConstant, 255, CV_THRESH_BINARY);
////      //threshold(binary_img, binary_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
////
////      // output the preview images using the thresholded image as the background
////      if(preview_thresh){
////        cvtColor(binary_img, out_img, CV_GRAY2RGB);
////      }
////
////      //medianBlur ( median_img, median_img, 7 );
////
////      // find the contours in the image with parents and children to locate the doughnut looking dots
////      std::vector<std::vector<Point> > contours;
////      std::vector<std::vector<Point> > trimmed_contours;
////      std::vector<Vec4i> hierarchy;
////      findContours(binary_img.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );
////      // make sure the vectors are the same size:
////      if(hierarchy.size()!=contours.size()){
////        std::cout << "error, hierarchy and contours vectors are not the same size" << std::endl;
////        return -1;
////      }
////
////      //std::vector<RotatedRect> minEllipse( contours.size() );
////
////      // count up the number of potential candidates for marker blobs
////      size_t min_marker_blob_size = 75;
////      std::vector<KeyPoint> potential_markers;
////      // global centroid of the potential markers
////      //float centroid_x = (float)binary_img.cols/2;//0.0;
////      //float centroid_y = (float)binary_img.rows/2;//0.0;
////      for(size_t idx=0; idx<hierarchy.size(); ++idx){
////          if(hierarchy[idx][3]!=-1&&hierarchy[idx][2]!=-1&&contours[idx].size()>min_marker_blob_size){
////            // compute the centroid of this marker
////            float cx = 0.0;
////            float cy = 0.0;
////            for(size_t i=0;i<contours[idx].size();++i){
////              cx += contours[idx][i].x;
////              cy += contours[idx][i].y;
////            }
////            float denom = (float)contours[idx].size();
////            cx /= denom;
////            cy /= denom;
////            potential_markers.push_back(KeyPoint(cx,cy,contours[idx].size()));
////            trimmed_contours.push_back(contours[idx]);
////            //centroid_x += cx;
////            //centroid_y += cy;
////          }
////      }
////      if(potential_markers.size() < 3){
////        // output the image with markers located
////        for(size_t i=0;i<potential_markers.size();++i)
////          circle(out_img,potential_markers[i].pt,20,Scalar(255,255,100),-1);
////        imwrite(output_images[image_it], out_img);
////        // error codes: 2 left only 3 right only 4 left and right 5 middle only 6 middle and left 7 middle and right 8 all three
////        if(image_it==0)
////          error_code = 2; // left alone
////        else if(image_it==1&&error_code==0)
////          error_code = 3; // right alone
////        else if(image_it==1&&error_code!=0)
////          error_code = 4; // left and right failed
////        else if(image_it==2&&error_code==0)
////          error_code = 5; // middle alone
////        else if(image_it==2&&error_code==2)
////          error_code = 6; // middle and left
////        else if(image_it==2&&error_code==3)
////          error_code = 7; // middle and right
////        else if(image_it==2&&error_code==4)
////          error_code = 8; // all
////        continue;
////      }
////      if(potential_markers.size()==0){
////        std::cout << "error, no marker dots could be located" << std::endl;
////        return -1;
////      }
////      if(potential_markers.size()!=trimmed_contours.size()){
////        std::cout << "error, potential markers and trimmed contours should be the same size" << std::endl;
////        return -1;
////      }
////      //centroid_x /= (float)potential_markers.size();
////      //centroid_y /= (float)potential_markers.size();
////
////      // final set of marker dots
////      std::vector<KeyPoint> marker_dots(3);
////      for(size_t i=0;i<3;++i)
////        marker_dots[i] = potential_markers[i];
////
////      // downselect to the three dots closest to the centroid if there are more than three
////      if(potential_markers.size()>3){
////
////        std::vector<std::pair<int,float> > id_quant;
////
////        // create a copy of the binary image with the contours turned white:
////        bi_copy_img = binary_img.clone();
////        drawContours(bi_copy_img, trimmed_contours, -1, Scalar::all(255), CV_FILLED, 8);
////        // invert the image
////        bitwise_not(bi_copy_img, bi_copy_img);
////
////        for(size_t idx=0;idx<trimmed_contours.size();++idx){
////          // determine the bounding box for each contour:
////          Rect boundingBox = boundingRect(trimmed_contours[idx]);
////          // find the center of the box
////          int center_x = (boundingBox.br().x + boundingBox.tl().x)/2;
////          int center_y = (boundingBox.br().y + boundingBox.tl().y)/2;
////          int x_span = (boundingBox.br().x - center_x)*1.5; // expand 1.5 times the original width and height
////          int y_span = (boundingBox.br().y - center_y)*1.5;
////
////          //rectangle( bi_copy_img, boundingBox.tl(), boundingBox.br(), Scalar(0,255,0), 2, 8, 0 );
////          // iterate over the box and count up the intensity values from the copied binary image
////          float total_intensity = 0.0;
////          for(int j=center_y-y_span;j<center_y+y_span;++j){
////            if(j<0||j>bi_copy_img.rows)continue;
////            for(int i=center_x-x_span;i<center_x+x_span;++i){
////              if(i<0||i>bi_copy_img.cols)continue;
////              total_intensity += (float)bi_copy_img.at<uchar>(j,i);
////            }
////          }
////          //std::cout << " QUANTITY FOR CONTOUR: " << total_intensity << std::endl;
////          id_quant.push_back(std::pair<int,float>(idx,total_intensity));
////        }
////        // order the vector by smallest quantity
////        std::sort(id_quant.begin(), id_quant.end(), pair_compare);
////        // take the top three as the markers:
////        for(size_t i=0;i<3;++i){
////          marker_dots[i] = potential_markers[id_quant[i].first];
////        }
////      }
//////      for(size_t i=0;i<3;++i){
//////        circle(out_img,marker_dots[i].pt,20,Scalar(255,255,100),-1);
//////      }
////
////
////      // compute the median dot size:
////      std::vector<std::pair<int,float> > id_size;
////      for(size_t idx=0; idx<hierarchy.size(); ++idx){
////          if(hierarchy[idx][3]!=-1&&hierarchy[idx][2]==-1&&contours[idx].size()>min_marker_blob_size){
////            id_size.push_back(std::pair<int,float>(idx,contours[idx].size()));
////          }
////      }
////      int median_id = id_size.size()/2;
////      std::sort(id_size.begin(), id_size.end(), pair_compare);
////      const float median_size = id_size[median_id].second;
////
////      // now that the corner marker dots are found, look for the rest of the dots:
////      // create a copy of the binary image with the contours turned white:
////      bi_copy_img = binary_img.clone();
////      std::vector<std::vector<Point> > trimmed_dot_contours;
////      std::vector<float> trimmed_dot_cx;
////      std::vector<float> trimmed_dot_cy;
////
////      for(size_t idx=0; idx<hierarchy.size(); ++idx){
////          if(hierarchy[idx][3]!=-1&&hierarchy[idx][2]==-1&&contours[idx].size()>0.8*median_size&&contours[idx].size()<1.2*median_size){
////            trimmed_dot_contours.push_back(contours[idx]);
////            //std::cout << " found contour size " << contours[idx].size() << std::endl;
////            float cx = 0.0;
////            float cy = 0.0;
////            for(size_t i=0;i<contours[idx].size();++i){
////              cx += contours[idx][i].x;
////              cy += contours[idx][i].y;
////            }
////            float denom = (float)contours[idx].size();
////            cx /= denom;
////            cy /= denom;
////            trimmed_dot_cx.push_back(cx);
////            trimmed_dot_cy.push_back(cy);
////          }
////      }
////      //drawContours(out_img, trimmed_dot_contours, -1, Scalar(255,255,100), CV_FILLED, 8);
////      if(trimmed_dot_cy.size()!=trimmed_dot_cx.size()||trimmed_dot_cy.size()!=trimmed_dot_contours.size()){
////        std::cout << "error, centroid vectors are the wrong size" << std::endl;
////        return -1;
////      }
////
////      // iterate the marker dots and gather the closest two non-colinear points
////      Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
////      point_cloud->pts.resize(trimmed_dot_contours.size());
////      for(size_t i=0;i<trimmed_dot_contours.size();++i){
////        point_cloud->pts[i].x = trimmed_dot_cx[i];
////        point_cloud->pts[i].y = trimmed_dot_cy[i];
////      }
////
////      DEBUG_MSG("building the kd-tree");
////      Teuchos::RCP<kd_tree_2d_t> kd_tree =
////          Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
////      kd_tree->buildIndex();
////      DEBUG_MSG("kd-tree completed");
////      scalar_t query_pt[2];
////      int_t num_neigh = 4;
////      std::vector<size_t> ret_index(num_neigh);
////      std::vector<scalar_t> out_dist_sqr(num_neigh);
////
////      std::vector<std::vector<Point> > marker_line_dots(3);
////      //const float dist_tol = 200.0;
////      for(size_t i=0;i<marker_dots.size();++i){
////        query_pt[0] = marker_dots[i].pt.x;
////        query_pt[1] = marker_dots[i].pt.y;
////        // find the closest dots to this point
////        kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
////        // let the first closest dot define one of the cardinal directions for this marker dot
////        marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[0]],trimmed_dot_cy[ret_index[0]]));
////        // get another dot that is not colinear to define the second cardinal direction for this marker
////        const float dist_1 = std::abs(dist_from_line(trimmed_dot_cx[ret_index[1]],trimmed_dot_cy[ret_index[1]],marker_dots[i].pt.x,marker_dots[i].pt.y,trimmed_dot_cx[ret_index[0]],trimmed_dot_cy[ret_index[0]]));
////        const float dist_2 = std::abs(dist_from_line(trimmed_dot_cx[ret_index[2]],trimmed_dot_cy[ret_index[2]],marker_dots[i].pt.x,marker_dots[i].pt.y,trimmed_dot_cx[ret_index[0]],trimmed_dot_cy[ret_index[0]]));
////        const float dist_3 = std::abs(dist_from_line(trimmed_dot_cx[ret_index[3]],trimmed_dot_cy[ret_index[3]],marker_dots[i].pt.x,marker_dots[i].pt.y,trimmed_dot_cx[ret_index[0]],trimmed_dot_cy[ret_index[0]]));
////        if(dist_1>=dist_2&&dist_1>=dist_3)
////          marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[1]],trimmed_dot_cy[ret_index[1]]));
////        if(dist_2>=dist_1&&dist_2>=dist_3)
////          marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[2]],trimmed_dot_cy[ret_index[2]]));
////        else
////          marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[3]],trimmed_dot_cy[ret_index[3]]));
////        if(marker_line_dots[i].size()!=2){
////          std::cout << "error, marker line dots vector is the wrong size" << std::endl;
////          return -1;
////        }
////        //circle(out_img,marker_line_dots[i][0],20,Scalar(255,0,0),-1);
////        //circle(out_img,marker_line_dots[i][1],20,Scalar(255,0,0),-1);
////        //drawContours(out_img, marker_line_dots[i], -1, Scalar(255,0,0), CV_FILLED, 8);
////      } // end marker dot loop
////
////      // at this point, it's not known which marker dot is the origin. The marker line dots define two dots for each marker dot, one dot in each cardinal direction.
////      // if a line is extended from each marker dot to the cardinal direction dots, the origin will be the marker that lies closes to the four lines that extend from
////      // the other marker dots
////      float min_dist = 0.0;
////      int origin_id = 0;
////      for(size_t i=0;i<marker_dots.size();++i){
////        float total_dist = 0.0;
////        for(size_t j=0;j<marker_dots.size();++j){
////          if(i==j)continue;
////          float dist_1 = std::abs(dist_from_line(marker_dots[i].pt.x,marker_dots[i].pt.y,marker_dots[j].pt.x,marker_dots[j].pt.y,marker_line_dots[j][0].x,marker_line_dots[j][0].y));
////          float dist_2 = std::abs(dist_from_line(marker_dots[i].pt.x,marker_dots[i].pt.y,marker_dots[j].pt.x,marker_dots[j].pt.y,marker_line_dots[j][1].x,marker_line_dots[j][1].y));
////          total_dist += dist_1 + dist_2;
////        }
////        if(total_dist < min_dist || i==0){
////          min_dist = total_dist;
////          origin_id = i;
////        }
////      }
////      // draw the origin marker dot
////      circle(out_img,marker_dots[origin_id].pt,20,Scalar(0,255,255),-1);
////      // compute the distance between the origin and the other marker dots and compare with the board size to get x and y axes
////      float max_dist = 0.0;
////      int xaxis_id = 0;
////      int yaxis_id = 0;
////      for(size_t i=0;i<marker_dots.size();++i){
////        if((int)i==origin_id) continue;
////        float dist = (marker_dots[origin_id].pt.x - marker_dots[i].pt.x)*(marker_dots[origin_id].pt.x - marker_dots[i].pt.x) +
////            (marker_dots[origin_id].pt.y - marker_dots[i].pt.y)*(marker_dots[origin_id].pt.y - marker_dots[i].pt.y);
////        if(dist>max_dist){
////          max_dist = dist;
////          xaxis_id = i;
////        }else{
////          yaxis_id = i;
////        }
////      }
////      // note: shortest distance is always y
//////      if(board_width < board_height){
//////        int temp_id = xaxis_id;
//////        xaxis_id = yaxis_id;
//////        yaxis_id = temp_id;
//////      }
////      circle(out_img,marker_dots[xaxis_id].pt,20,Scalar(255,255,100),-1);
////      circle(out_img,marker_dots[yaxis_id].pt,20,Scalar(0,0,255),-1);
////
////      // find the opposite corner dot...
////      // it will be the closest combined distance from x/y axis that is not the origin dot
////      float opp_dist = 1.0E10;
////      int opp_id = 0;
////      for(size_t i=0;i<trimmed_dot_contours.size();++i){
////        float dist_11 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y,marker_line_dots[xaxis_id][0].x,marker_line_dots[xaxis_id][0].y));
////        float dist_12 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y,marker_line_dots[xaxis_id][1].x,marker_line_dots[xaxis_id][1].y));
////        // take the smaller of the two
////        float dist_1 = dist_11 < dist_12 ? dist_11 : dist_12;
////        float dist_21 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_line_dots[yaxis_id][0].x,marker_line_dots[yaxis_id][0].y));
////        float dist_22 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_line_dots[yaxis_id][1].x,marker_line_dots[yaxis_id][1].y));
////        // take the smaller of the two
////        float dist_2 = dist_21 < dist_22 ? dist_21 : dist_22;
////        float total_dist = dist_1 + dist_2;
////        //std::stringstream dot_text;
////        //dot_text << total_dist;
////        //putText(out_img, dot_text.str(), Point2f(trimmed_dot_cx[i],trimmed_dot_cy[i]) + Point2f(20,20),
////        //  FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255,0,0), 1.5, CV_AA);
////        if(total_dist <= opp_dist){
////          opp_dist = total_dist;
////          opp_id = i;
////        }
////      }
////      // TODO test for tol from line...
////      circle(out_img,Point(trimmed_dot_cx[opp_id],trimmed_dot_cy[opp_id]),20,Scalar(100,255,100),-1);
////
////      // determine the pattern size
////      float axis_tol = 10.0;
////      int pattern_width = 0;
////      int pattern_height = 0;
////      const float dist_xaxis_dot = dist_from_line(marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y,marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y);
////      const float dist_yaxis_dot = dist_from_line(marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y,marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y);
////      for(size_t i=0;i<trimmed_dot_contours.size();++i){
////        float dist_x = dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y,marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y);
////        float dist_y = dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y);
////        if(std::abs(dist_x) < axis_tol && dist_y > 0.0 && dist_y < dist_xaxis_dot){
////          circle(out_img,Point(trimmed_dot_cx[i],trimmed_dot_cy[i]),20,Scalar(0,255,100),-1);
////          pattern_width++;
////        }
////        else if(std::abs(dist_y) < axis_tol && dist_x > 0.0 && dist_x < dist_yaxis_dot){
////          circle(out_img,Point(trimmed_dot_cx[i],trimmed_dot_cy[i]),20,Scalar(255,255,255),-1);
////          pattern_height++;
////        }
////      }
////      if(pattern_width<=1||pattern_height<=1){
////        std::cout << "error, could not determine the pattern size" << std::endl;
////        return -1;
////      }
////      // add the marker dots
////      pattern_width += 2;
////      pattern_height += 2;
////      std::cout << "pattern size: " << pattern_width << " x " << pattern_height << std::endl;
////      const float rough_grid_spacing = dist_xaxis_dot / (pattern_width-1);
////
////
////      // now determine the 6 parameter image warp using the four corners
////      std::vector<scalar_t> proj_xl(4,0.0);
////      std::vector<scalar_t> proj_yl(4,0.0);
////      std::vector<scalar_t> proj_xr(4,0.0);
////      std::vector<scalar_t> proj_yr(4,0.0);
////      // board space coords
////      proj_xl[0] = 0.0;               proj_yl[0] = 0.0;
////      proj_xl[1] = pattern_width-1.0; proj_yl[1] = 0.0;
////      proj_xl[2] = pattern_width-1.0; proj_yl[2] = pattern_height-1.0;
////      proj_xl[3] = 0.0;               proj_yl[3] = pattern_height-1.0;
////      // image coords
////      proj_xr[0] = marker_dots[origin_id].pt.x; proj_yr[0] = marker_dots[origin_id].pt.y;
////      proj_xr[1] = marker_dots[xaxis_id].pt.x;  proj_yr[1] = marker_dots[xaxis_id].pt.y;
////      proj_xr[2] = trimmed_dot_cx[opp_id]; proj_yr[2] = trimmed_dot_cy[opp_id];
////      proj_xr[3] = marker_dots[yaxis_id].pt.x;  proj_yr[3] = marker_dots[yaxis_id].pt.y;
////
////      Teuchos::SerialDenseMatrix<int_t,double> proj_matrix = compute_affine_matrix(proj_xl,proj_yl,proj_xr,proj_yr);
////      std::cout << "PROJECTION MATRIX " << std::endl;
////      std::cout << proj_matrix << std::endl;
////
////      // convert all the points in the grid to the image and draw a dot in the image:
////      std::vector<size_t> closest_neigh_id(1);
////      std::vector<scalar_t> neigh_dist_2(1);
////
////      for(int j=-3;j<pattern_height+3;++j){
////        for(int i=-3;i<pattern_width+3;++i){
////          float grid_x = i;
////          float grid_y = j;
////          float image_x = 0.0;
////          float image_y = 0.0;
////          project_grid_to_image(proj_matrix,grid_x,grid_y,image_x,image_y);
////          if(image_x<=5||image_y<=5||image_x>=img.cols-5||image_y>=img.rows-5) continue;
////          // TODO add the axis and origin points
////          if(i==0&&j==0) continue;
////          if(i==pattern_width-1&&j==0) continue;
////          if(i==0&&j==pattern_height-1) continue;
////
////          // find the closest contour near this grid point and
////          query_pt[0] = image_x;
////          query_pt[1] = image_y;
////          // find the closest dots to this point
////          kd_tree->knnSearch(&query_pt[0], 1, &closest_neigh_id[0], &neigh_dist_2[0]);
////          int neigh_id = closest_neigh_id[0];
////          if(neigh_dist_2[0]<=(0.5*rough_grid_spacing)*(0.5*rough_grid_spacing)){
////            circle(out_img,Point(trimmed_dot_cx[neigh_id],trimmed_dot_cy[neigh_id]),10,Scalar(255,0,255),-1);
////          }
//////          std::stringstream dot_text;
//////          dot_text << i <<"," <<j;
//////          putText(out_img, dot_text.str(), Point2f(image_x,image_y) + Point2f(20,20),
//////            FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255,0,0), 1.5, CV_AA);
//////          circle(out_img,Point(image_x,image_y),10,Scalar(0,255,255),-1);
////        }
////      }
////
//////      if((int)cal_dots.size() != board_height*board_width - 3){
//////        imwrite(output_images[image_it], out_img);
//////        std::cout << "error, invalid number of cal dots found" << std::endl;
//////        if(image_it==0)
//////          error_code = 9; // left alone
//////        else if(image_it==1&&error_code==0)
//////          error_code = 10; // right alone
//////        else if(image_it==1&&error_code!=0)
//////          error_code = 11; // left and right failed
//////        else if(image_it==2&&error_code==0)
//////          error_code = 12; // middle alone
//////        else if(image_it==2&&error_code==9)
//////          error_code = 13; // middle and left
//////        else if(image_it==2&&error_code==10)
//////          error_code = 14; // middle and right
//////        else if(image_it==2&&error_code==11)
//////          error_code = 15; // all
//////        continue;
//////      }
////
////    } // end is_cal
//  } // end images

//  return error_code;
//}

}
