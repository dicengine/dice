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

#include <DICe_StereoCalib.h>
#include <DICe_Shape.h>
#include <DICe_Parser.h>
#include <DICe_PointCloud.h>
#include <DICe_Triangulation.h>

#include <Teuchos_SerialDenseMatrix.hpp>

#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

namespace DICe {


DICE_LIB_DLL_EXPORT
bool
is_in_quadrilateral(const float & x,
  const float & y,
  const std::vector<float> & box_x,
  const std::vector<float> & box_y){
  float angle=0.0;
  assert(box_x.size()==5);
  assert(box_y.size()==5);
  for (int_t i=0;i<4;i++) {
    // get the two end points of the polygon side and construct
    // a vector from the point to each one:
    const float dx1 = box_x[i] - x;
    const float dy1 = box_y[i] - y;
    const float dx2 = box_x[i+1] - x;
    const float dy2 = box_y[i+1] - y;
    angle += angle_2d(dx1,dy1,dx2,dy2);
  }
  // if the angle is greater than PI, the point is in the polygon
  if(std::abs(angle) >= DICE_PI){
    return true;
  }
  return false;
}

/// free function to compute the distance from a point (x,y) to a line
/// defined by two points (x1,y1), (x2,y2). The sign determines which side
/// of the line the point is on.
DICE_LIB_DLL_EXPORT
float dist_from_line(const float & x,
  const float & y,
  const float & x1,
  const float & y1,
  const float & x2,
  const float & y2){
  float s = (y2 - y1) * x + (x1 - x2) * y + (x2 * y1 - x1 * y2);
  assert((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1)!=0.0);
  return s/std::sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));
};

/// free function to compare std::pair to enable vector sorting
DICE_LIB_DLL_EXPORT
bool pair_compare(const std::pair<int,float>& left, const std::pair<int,float>& right) {
  return left.second < right.second;
}

///// free function to compare std::pair to enable vector sorting
//DICE_LIB_DLL_EXPORT
//bool pair_compare_inv(const std::pair<int,float>& left, const std::pair<int,float>& right) {
//  return left.second > right.second;
//}

/// free function to compare std::tuple to enable vector sorting
DICE_LIB_DLL_EXPORT
bool tuple_compare(const std::tuple<int,float,float>& left, const std::tuple<int,float,float>& right) {
  if (std::get<1>(left) < std::get<1>(right))
    return true;
  else if (std::get<1>(left) > std::get<1>(right))
    return false;
  else
    return std::get<2>(left) < std::get<2>(right);
}

/// free function to compare std::tuple to enable vector sorting by second element (index 1)
DICE_LIB_DLL_EXPORT
bool tuple_second_compare(const std::tuple<int,float,float>& left, const std::tuple<int,float,float>& right) {
  return std::get<1>(left) < std::get<1>(right);
}

/// free function to compare std::tuple to enable vector sorting by third element (index 2)
DICE_LIB_DLL_EXPORT
bool tuple_third_compare(const std::tuple<int,float,float>& left, const std::tuple<int,float,float>& right) {
  return std::get<2>(left) < std::get<2>(right);
}

/// free function to project a point from a grid to the affine transformed point in an image
DICE_LIB_DLL_EXPORT
void project_grid_to_image(const Teuchos::SerialDenseMatrix<int_t,double> & proj_matrix,
  const float & xl,
  const float & yl,
  float & xr,
  float & yr){
  // first apply the quadratic warp transform
  assert(proj_matrix.numRows()==3&&proj_matrix.numCols()==3);
  xr = (proj_matrix(0,0)*xl + proj_matrix(0,1)*yl + proj_matrix(0,2))/(proj_matrix(2,0)*xl + proj_matrix(2,1)*yl + proj_matrix(2,2));
  yr = (proj_matrix(1,0)*xl + proj_matrix(1,1)*yl + proj_matrix(1,2))/(proj_matrix(2,0)*xl + proj_matrix(2,1)*yl + proj_matrix(2,2));
}

DICE_LIB_DLL_EXPORT
float
StereoCalibDotTarget(Teuchos::RCP<Teuchos::ParameterList> pre_params,
  const std::vector<std::string>& imagelist,
  const std::string & output_filename){

  DEBUG_MSG("StereoCalibDotTarget(): begin");
  assert(imagelist.size()>1&&imagelist.size()%2==0);

  // get the ids of images that have been manually turned off to get a better cal:
  Teuchos::ParameterList skip_sublist;
  std::set<int> skip_ids;
  if(pre_params->isParameter(DICe::cal_manual_skip_images)){
    skip_sublist = pre_params->sublist(DICe::cal_manual_skip_images);
    for(Teuchos::ParameterList::ConstIterator it=skip_sublist.begin();it!=skip_sublist.end();++it){
      std::string string_id = it->first;
      size_t found = string_id.find_last_of("_");
      string_id = string_id.substr(found+1,string_id.length());
      const int id = std::stoi(string_id);
      DEBUG_MSG("StereoCalibDotTarget: manually turning off image " << id);
      skip_ids.insert(id);
    }
  }
  const int_t num_images = imagelist.size()/2;
  std::vector<bool> image_success(num_images,true);

  std::vector<std::vector<Point2f> > image_points_left;
  std::vector<std::vector<Point2f> > image_points_right;
  std::vector<std::vector<Point3f> > object_points;
  Size imageSize;

  // iterate all the images in the set
  for(int_t i=0;i<num_images;++i){
    if(skip_ids.find(i)!=skip_ids.end()) continue;
    std::vector<Point2f> pre_image_points_left;
    std::vector<Point3f> pre_object_points_left;
    std::vector<Point2f> pre_image_points_right;
    std::vector<Point3f> pre_object_points_right;

    for(int_t k=0;k<2;k++){ // k represents 0 for the left image 1 for the right
      const string& filename = imagelist[i*2+k];
      std::cout << "processing cal image " << filename << std::endl;

      // find the cal dot points and check show a preview of their locations
      int pre_code = 0;

      if(k==0)
        pre_code = pre_process_cal_image(filename,"",pre_params,pre_image_points_left,pre_object_points_left,imageSize);
      else
        pre_code = pre_process_cal_image(filename,"",pre_params,pre_image_points_right,pre_object_points_right,imageSize);

      DEBUG_MSG("pre_process_cal_image return value: " << pre_code);
      if(pre_code!=0){
        std::cout << "error, pre-processing image for cal dot locations failed" << std::endl;
        image_success[i] = false;
        break;
      }
      if((k==0&&pre_object_points_left.size()<4)||(k==1&&pre_object_points_right.size()<4)){
        std::cout << "error, failed to locate enough dots" << std::endl;
        image_success[i] = false;
        break;
      }
      if(k==1){
        image_points_left.resize(image_points_left.size()+1);
        image_points_right.resize(image_points_right.size()+1);
        object_points.resize(object_points.size()+1);
        // check for matching points between the two images and push those back on the accumulated set of points
        for(size_t n=0;n<pre_object_points_left.size();++n){
          for(size_t m=0;m<pre_object_points_right.size();++m){
            if(pre_object_points_left[n]==pre_object_points_right[m]){
                image_points_left[image_points_left.size()-1].push_back(pre_image_points_left[n]);
                image_points_right[image_points_right.size()-1].push_back(pre_image_points_right[m]);
                object_points[object_points.size()-1].push_back(pre_object_points_right[m]);
                break;
            }
          } // loop over right image points
        } // loop over left image points
      } // end is left image
    } // loop over left and right
  } // loop over images

  std::cout << "successfully found points in " << object_points.size() << " images" << std::endl;
  for(size_t i=0;i<object_points.size();++i){
    std::cout << "image index " << i << " num points " << object_points[i].size() << std::endl;
//    for(size_t j=0;j<object_points[i].size();++j){
//      std::cout << "point " << j << " " << object_points[i][j].x << " " << object_points[i][j].y << " "
//          << image_points_left[i][j].x << " " << image_points_left[i][j].y << " "
//          << image_points_right[i][j].x << " " << image_points_right[i][j].y << std::endl;
//    }
  }
  std::vector<scalar_t> cal_qualities;
  const float rms = compute_cal_matrices(object_points,image_points_left,image_points_right,imageSize,cal_qualities,output_filename);

  // create an output file with corresponding errors for each image in the selectable list:
  // save intrinsic parameters
  std::FILE * filePtr = fopen("cal_errors.txt","w");
  DEBUG_MSG("StereoCalibDotTarget(): writing cal_errors.txt");
  size_t quality_index = 0;
  for(int i=0;i<num_images;++i){
    if(!image_success[i])
      fprintf(filePtr,"failed\n");
    else if(skip_ids.find(i)!=skip_ids.end())
      fprintf(filePtr,"skipped\n");
    else{
      assert(quality_index<cal_qualities.size());
      fprintf(filePtr,"%.4f\n",cal_qualities[quality_index]);
      quality_index++;
    }
  }
  fclose (filePtr);
  DEBUG_MSG("StereoCalibDotTarget(): end");
  return rms;
}

DICE_LIB_DLL_EXPORT
float compute_cal_matrices(std::vector<std::vector<Point3f> > & object_points,
  std::vector<std::vector<Point2f> > & image_points_left,
  std::vector<std::vector<Point2f> > & image_points_right,
  Size & imageSize,
  std::vector<scalar_t> & cal_qualities,
  const std::string & output_filename){
  DEBUG_MSG("compute_cal_matrices(): begin");

  cal_qualities.clear();
  cal_qualities.resize(object_points.size());
  const size_t nimages = object_points.size();
  Mat cameraMatrix[2], distCoeffs[2];
  cameraMatrix[0] = initCameraMatrix2D(object_points,image_points_left,imageSize,0);
  cameraMatrix[1] = initCameraMatrix2D(object_points,image_points_right,imageSize,0);
  Mat R, T, E, F;
  double rms = stereoCalibrate(object_points, image_points_left, image_points_right,
    cameraMatrix[0], distCoeffs[0],
    cameraMatrix[1], distCoeffs[1],
    imageSize, R, T, E, F,
    CALIB_FIX_ASPECT_RATIO +
    CALIB_USE_INTRINSIC_GUESS +
    CALIB_SAME_FOCAL_LENGTH +
    CALIB_ZERO_TANGENT_DIST +
    CALIB_RATIONAL_MODEL +
    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
  cout << "done with RMS error=" << rms << endl;

  std::cout << cameraMatrix[0] << std::endl;
  std::cout << distCoeffs[0] << std::endl;
  std::cout << cameraMatrix[1] << std::endl;
  std::cout << distCoeffs[1] << std::endl;

  // CALIBRATION QUALITY CHECK
  // because the output fundamental matrix implicitly
  // includes all the output information,
  // we can check the quality of calibration using the
  // epipolar geometry constraint: m2^t*F*m1=0
  double err = 0;
  int npoints = 0;
  std::vector<Vec3f> lines[2];
  for(size_t i = 0; i < nimages; i++ )
  {
    int npt = (int)image_points_left[i].size();
    assert(npt!=0);
    Mat imgpt[2];
    for( size_t k = 0; k < 2; k++ )
    {
      if(k==0)
        imgpt[k] = Mat(image_points_left[i]);
      else
        imgpt[k] = Mat(image_points_right[i]);
      undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
      computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
    }
    double imgErr = 0.0;
    for(int j = 0; j < npt; j++ )
    {
      double errij = fabs(image_points_left[i][j].x*lines[1][j][0] +
        image_points_left[i][j].y*lines[1][j][1] + lines[1][j][2]) +
            fabs(image_points_right[i][j].x*lines[0][j][0] +
              image_points_right[i][j].y*lines[0][j][1] + lines[0][j][2]);
      err += errij;
      imgErr += errij;
    }
    cal_qualities[i] = imgErr/npt;
    npoints += npt;
  }
  assert(npoints!=0);
  std::cout << "average epipolar err = " <<  err/npoints << endl;

  // save intrinsic parameters
  std::FILE * filePtr = fopen(output_filename.c_str(),"w");
  fprintf(filePtr,"%4.12E # cx\n",cameraMatrix[0].at<double>(0,2));
  fprintf(filePtr,"%4.12E # cy\n",cameraMatrix[0].at<double>(1,2));
  fprintf(filePtr,"%4.12E # fx\n",cameraMatrix[0].at<double>(0,0));
  fprintf(filePtr,"%4.12E # fy\n",cameraMatrix[0].at<double>(1,1));
  fprintf(filePtr,"%4.12E # fs\n",cameraMatrix[0].at<double>(0,1));
  fprintf(filePtr,"%4.12E # k1\n",distCoeffs[0].at<double>(0,0)); //
  fprintf(filePtr,"%4.12E # k2\n",distCoeffs[0].at<double>(0,1)); //
  fprintf(filePtr,"%4.12E # k6\n",distCoeffs[0].at<double>(0,7)); //
  fprintf(filePtr,"%4.12E # cx\n",cameraMatrix[1].at<double>(0,2));
  fprintf(filePtr,"%4.12E # cy\n",cameraMatrix[1].at<double>(1,2));
  fprintf(filePtr,"%4.12E # fx\n",cameraMatrix[1].at<double>(0,0));
  fprintf(filePtr,"%4.12E # fy\n",cameraMatrix[1].at<double>(1,1));
  fprintf(filePtr,"%4.12E # fs\n",cameraMatrix[1].at<double>(0,1));
  fprintf(filePtr,"%4.12E # k1\n",distCoeffs[1].at<double>(0,0)); //
  fprintf(filePtr,"%4.12E # k2\n",distCoeffs[1].at<double>(0,1)); //
  fprintf(filePtr,"%4.12E # k6\n",distCoeffs[1].at<double>(0,7)); //

  Mat R1, R2, P1, P2, Q;
  Rect validRoi[2];

  stereoRectify(cameraMatrix[0], distCoeffs[0],
    cameraMatrix[1], distCoeffs[1],
    imageSize, R, T, R1, R2, P1, P2, Q,
    CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
  // rotation matrix
  fprintf(filePtr,"%4.12E # R11\n",R.at<double>(0,0));
  fprintf(filePtr,"%4.12E # R12\n",R.at<double>(0,1));
  fprintf(filePtr,"%4.12E # R13\n",R.at<double>(0,2));
  fprintf(filePtr,"%4.12E # R21\n",R.at<double>(1,0));
  fprintf(filePtr,"%4.12E # R22\n",R.at<double>(1,1));
  fprintf(filePtr,"%4.12E # R23\n",R.at<double>(1,2));
  fprintf(filePtr,"%4.12E # R31\n",R.at<double>(2,0));
  fprintf(filePtr,"%4.12E # R32\n",R.at<double>(2,1));
  fprintf(filePtr,"%4.12E # R33\n",R.at<double>(2,2));
  // translations
  fprintf(filePtr,"%4.12E # tx\n",T.at<double>(0,0));
  fprintf(filePtr,"%4.12E # ty\n",T.at<double>(1,0));
  fprintf(filePtr,"%4.12E # tz\n",T.at<double>(2,0));

  fclose (filePtr);

  DEBUG_MSG("compute_cal_matrices(): end");
  return rms;
}

DICE_LIB_DLL_EXPORT
int pre_process_cal_image(const std::string & image_filename,
  const std::string & output_image_filename,
  Teuchos::RCP<Teuchos::ParameterList> pre_process_params,
  std::vector<Point2f> & image_points,
  std::vector<Point3f> & object_points,
  Size & imageSize){
  if(pre_process_params==Teuchos::null){
    std::cout << "error, pre_process_params is null" << std::endl;
    return -1;
  }
  image_points.clear();
  object_points.clear();
  const bool output_preview_images = output_image_filename != "";
  const bool preview_thresh = pre_process_params->get<bool>("preview_thresh",false);
  const bool has_adaptive = pre_process_params->get<bool>("cal_target_has_adaptive",false);
  const int filterMode = cv::ADAPTIVE_THRESH_MEAN_C;//pre_process_params->get<int>("filterMode",cv::ADAPTIVE_THRESH_GAUSSIAN_C);
  int invertedMode = cv::THRESH_BINARY;
  if(pre_process_params->get<bool>("cal_target_is_inverted",false))
    invertedMode = cv::THRESH_BINARY_INV;
  //const int invertedMode = pre_process_params->get<int>("invertedMode",cv::THRESH_BINARY);
  //const int antiInvertedMode = pre_process_params->get<int>("antiInvertedMode",cv::THRESH_BINARY_INV);
  if(!pre_process_params->isParameter("cal_target_spacing_size")){
    std::cout << "error, pattern_spacing is missing from parameters" << std::endl;
    return -1;
  }
  const float pattern_size = (float)pre_process_params->get<double>("cal_target_spacing_size");
  if(!pre_process_params->isParameter("cal_target_block_size")){
    std::cout << "error, cal_target_block_size is missing from parameters" << std::endl;
    return -1;
  }
  const double blockSize = pre_process_params->get<double>("cal_target_block_size");
  if(!pre_process_params->isParameter("cal_target_binary_constant")){
    std::cout << "error, cal_target_binary_constant is missing from parameters" << std::endl;
    return -1;
  }
  const double binaryConstant = pre_process_params->get<double>("cal_target_binary_constant");

  // load the image as an openCV mat
  Mat img = imread(image_filename, IMREAD_GRAYSCALE);
  Mat binary_img(img.size(),CV_8UC3);
  Mat out_img(img.size(), CV_8UC3);
  //Mat bi_copy_img(img.size(), CV_8UC3);
  cvtColor(img, out_img, cv::COLOR_GRAY2RGB);
  if(img.empty()){
    std::cout << "error, the image is empty" << std::endl;
    return -4;
  }
  if(!img.data){
    std::cout << "error, the image failed to load" << std::endl;
    return -4;
  }
  imageSize = img.size();
  // blur the image to remove noise
  GaussianBlur(img, binary_img, Size(9, 9), 2, 2 );
  //medianBlur ( median_img, median_img, 7 );
  // binary threshold to create black and white image
  if(has_adaptive)
    adaptiveThreshold(binary_img,binary_img,255,filterMode,invertedMode,blockSize,binaryConstant);
  else
    threshold(binary_img, binary_img, binaryConstant, 255, invertedMode);
  //threshold(binary_img, binary_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  // output the preview images using the thresholded image as the background
  if(preview_thresh){
    cvtColor(binary_img, out_img, cv::COLOR_GRAY2RGB);
  }

  // find the contours in the image with parents and children to locate the doughnut looking dots
  std::vector<std::vector<Point> > contours;
  std::vector<std::vector<Point> > trimmed_contours;
  std::vector<Vec4i> hierarchy;
  findContours(binary_img.clone(), contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE );

  // make sure the vectors are the same size:
  if(hierarchy.size()!=contours.size()){
    std::cout << "error, hierarchy and contours vectors are not the same size" << std::endl;
    return -1;
  }
  // count up the number of potential candidates for marker blobs
  std::vector<KeyPoint> potential_markers;
  // global centroids of the potential markers
  for(size_t idx=0; idx<hierarchy.size(); ++idx){
      if(hierarchy[idx][3]!=-1&&hierarchy[idx][2]!=-1){//&&contours[idx].size()>min_marker_blob_size){
        // compute the centroid of this marker
        float cx = 0.0;
        float cy = 0.0;
        for(size_t i=0;i<contours[idx].size();++i){
          cx += contours[idx][i].x;
          cy += contours[idx][i].y;
        }
        float denom = (float)contours[idx].size();
        assert(denom!=0.0);
        cx /= denom;
        cy /= denom;
        potential_markers.push_back(KeyPoint(cx,cy,contours[idx].size()));
        trimmed_contours.push_back(contours[idx]);
      }
  }
  //imwrite(output_image_filename, out_img);
  //return -2;
  if(potential_markers.size()!=trimmed_contours.size()){
    std::cout << "error, potential markers and trimmed contours should be the same size" << std::endl;
    return -1;
  }
  if(potential_markers.size()<3){
    // output the image with markers located
    for(size_t i=0;i<potential_markers.size();++i)
      circle(out_img,potential_markers[i].pt,10,Scalar(255,100,255),-1);
    if(output_preview_images)
      imwrite(output_image_filename, out_img);
    std::cout << "error, not enough marker dots could be located (need three)" << std::endl;
    return -2;
  }
  // final set of marker dots
  std::vector<KeyPoint> marker_dots(3);
  for(size_t i=0;i<3;++i)
    marker_dots[i] = potential_markers[i];
  // if there are more than three potential marker dots, use the fact that the markers are
  // surrounded by white space on the cal board to distinguish from background dots that may have been picked up
  if(potential_markers.size()>3){
    std::vector<std::pair<int,float> > id_quant;
    for(size_t idx=0;idx<trimmed_contours.size();++idx){
      // determine the ratio of the min enclosing circle size and the size of the contour
      Point2f center(0.0,0.0);
      float radius = 0.0;
      minEnclosingCircle(trimmed_contours[idx],center,radius);
      float circleArea = DICE_PI*radius*radius;
      float contArea = contourArea(trimmed_contours[idx]);
      if(contArea == 0.0)contArea = 1.0E-3;
      float areaRatio = circleArea / contArea;
      id_quant.push_back(std::pair<int,float>(idx,areaRatio));
    }
    // order the vector by smallest quantity
    std::sort(id_quant.begin(), id_quant.end(), pair_compare);
    // take the top three as the markers:
    for(size_t i=0;i<3;++i){
      marker_dots[i] = potential_markers[id_quant[i].first];
    }
  }

  // now that the marker dots are found, look for the rest of the regular dots.
  // the regular dots are filtered from the contours collected in the previous steps
  // by checking the size of the dot. It's a valid dot if the hierarchy is right and the size is between
  // 0.8 and 1.2 times the median dot size.
  // while iterating the dots, determine the centroid of the contours or dots.
  std::vector<float> trimmed_dot_cx;
  std::vector<float> trimmed_dot_cy;

  // limit the blob sizes based on the marker dot sizes
  size_t min_blob_size = (size_t)marker_dots[0].size * 0.5;

  // compute the median dot size:
  std::vector<std::pair<int,float> > id_size;
  for(size_t idx=0; idx<hierarchy.size(); ++idx){
    if(hierarchy[idx][3]!=-1&&hierarchy[idx][2]==-1&&contours[idx].size()>min_blob_size){
      id_size.push_back(std::pair<int,float>(idx,(float)contours[idx].size()));
    }
  }
  int median_id = id_size.size()/2;
  std::sort(id_size.begin(), id_size.end(), pair_compare);
  const float median_size = id_size[median_id].second;
  for(size_t idx=0; idx<hierarchy.size(); ++idx){
    if(hierarchy[idx][3]!=-1&&hierarchy[idx][2]==-1&&(float)contours[idx].size()>0.8*median_size&&(float)contours[idx].size()<1.2*median_size){
      float cx = 0.0;
      float cy = 0.0;
      for(size_t i=0;i<contours[idx].size();++i){
        cx += contours[idx][i].x;
        cy += contours[idx][i].y;
      }
      float denom = (float)contours[idx].size();
      assert(denom!=0.0);
      cx /= denom;
      cy /= denom;
      trimmed_dot_cx.push_back(cx);
      trimmed_dot_cy.push_back(cy);
      //drawContours(out_img, contours, idx, Scalar(255,0,0), CV_FILLED, 8);
    }
  }

  if(trimmed_dot_cy.size()!=trimmed_dot_cx.size()){
    std::cout << "error, dot centroid vectors are the wrong size" << std::endl;
    return -1;
  }

  // at this point, we take the three marker dots and determine which one is the origin, which is the xaxis
  // and which is the yaxis. We use subtle trick to determine the origin: we find the closest regular dots
  // to each marker dot and use them to define the cardinal directions for each marker dot. The origin will be the
  // dot that is closes to the axis lines of the other two marker dots

  // iterate the marker dots and gather the closest two non-colinear points
  Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
  point_cloud->pts.resize(trimmed_dot_cx.size());
  for(size_t i=0;i<trimmed_dot_cx.size();++i){
    point_cloud->pts[i].x = trimmed_dot_cx[i];
    point_cloud->pts[i].y = trimmed_dot_cy[i];
  }
  DEBUG_MSG("building the kd-tree");
  Teuchos::RCP<kd_tree_2d_t> kd_tree =
      Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  kd_tree->buildIndex();
  DEBUG_MSG("kd-tree completed");
  scalar_t query_pt[2];
  int_t num_neigh = 4;
  std::vector<size_t> ret_index(num_neigh);
  std::vector<scalar_t> out_dist_sqr(num_neigh);
  std::vector<std::vector<Point> > marker_line_dots(3);
  for(size_t i=0;i<marker_dots.size();++i){
    query_pt[0] = marker_dots[i].pt.x;
    query_pt[1] = marker_dots[i].pt.y;
    // find the closest dots to this point
    kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
    // let the first closest dot define one of the cardinal directions for this marker dot
    marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[0]],trimmed_dot_cy[ret_index[0]]));
    // get another dot that is not colinear to define the second cardinal direction for this marker

    // compute the slope of the first line:
    // prevent vertical slope
    const float slope_tol = 1.0E-3;
    float run = std::abs(marker_line_dots[i][0].x - marker_dots[i].pt.x)<slope_tol ? slope_tol : (marker_line_dots[i][0].x -  marker_dots[i].pt.x);
    const float slope_0 =  (marker_line_dots[i][0].y - marker_dots[i].pt.y)/run;
    // compute the slope of the other lines:
    run = std::abs(trimmed_dot_cx[ret_index[1]]-marker_dots[i].pt.x)<slope_tol ? slope_tol : (trimmed_dot_cx[ret_index[1]]-marker_dots[i].pt.x);
    const float slope_1 =  (trimmed_dot_cy[ret_index[1]]-marker_dots[i].pt.y)/run;
    run = std::abs(trimmed_dot_cx[ret_index[2]]-marker_dots[i].pt.x)<slope_tol ? slope_tol : (trimmed_dot_cx[ret_index[2]]-marker_dots[i].pt.x);
    const float slope_2 =  (trimmed_dot_cy[ret_index[2]]-marker_dots[i].pt.y)/run;
    run = std::abs(trimmed_dot_cx[ret_index[3]]-marker_dots[i].pt.x)<slope_tol ? slope_tol : (trimmed_dot_cx[ret_index[3]]-marker_dots[i].pt.x);
    const float slope_3 =  (trimmed_dot_cy[ret_index[3]]-marker_dots[i].pt.y)/run;
    const float angle_1 = std::abs(std::atan((slope_0 - slope_1)/(1.0 + slope_0*slope_1)));
    const float angle_2 = std::abs(std::atan((slope_0 - slope_2)/(1.0 + slope_0*slope_2)));
    const float angle_3 = std::abs(std::atan((slope_0 - slope_3)/(1.0 + slope_0*slope_3)));
    if(angle_1>=angle_2&&angle_1>=angle_3)
      marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[1]],trimmed_dot_cy[ret_index[1]]));
    else if(angle_2>=angle_1&&angle_2>=angle_3)
      marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[2]],trimmed_dot_cy[ret_index[2]]));
    else
      marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[3]],trimmed_dot_cy[ret_index[3]]));
    //circle(out_img,marker_line_dots[i][0],25,Scalar(255,0,0),2.0);
    //circle(out_img,marker_line_dots[i][1],25,Scalar(255,0,0),2.0);
    if(marker_line_dots[i].size()!=2){
      std::cout << "error, marker line dots vector is the wrong size" << std::endl;
      return -1;
    }
  } // end marker dot loop
  float min_dist = 0.0;
  int origin_id = 0;
  for(size_t i=0;i<marker_dots.size();++i){
    float total_dist = 0.0;
    for(size_t j=0;j<marker_dots.size();++j){
      if(i==j)continue;
      float dist_1 = std::abs(dist_from_line(marker_dots[i].pt.x,marker_dots[i].pt.y,marker_dots[j].pt.x,marker_dots[j].pt.y,marker_line_dots[j][0].x,marker_line_dots[j][0].y));
      float dist_2 = std::abs(dist_from_line(marker_dots[i].pt.x,marker_dots[i].pt.y,marker_dots[j].pt.x,marker_dots[j].pt.y,marker_line_dots[j][1].x,marker_line_dots[j][1].y));
      total_dist += dist_1 + dist_2;
    }
    if(total_dist < min_dist || i==0){
      min_dist = total_dist;
      origin_id = i;
    }
  }
  // draw the origin marker dot
  circle(out_img,marker_dots[origin_id].pt,20,Scalar(0,255,255),-1);
  // compute the distance between the origin and the other marker dots to get x and y axes
  // note: shortest distance is always y
  float max_dist = 0.0;
  int xaxis_id = 0;
  int yaxis_id = 0;
  std::set<int> ids_left_over;
  for(size_t i=0;i<3;++i) ids_left_over.insert(i);
  for(size_t i=0;i<marker_dots.size();++i){
    if((int)i==origin_id) continue;
    float dist = (marker_dots[origin_id].pt.x - marker_dots[i].pt.x)*(marker_dots[origin_id].pt.x - marker_dots[i].pt.x) +
        (marker_dots[origin_id].pt.y - marker_dots[i].pt.y)*(marker_dots[origin_id].pt.y - marker_dots[i].pt.y);
    if(dist>max_dist){
      max_dist = dist;
      xaxis_id = i;
    }
  }
  ids_left_over.erase(origin_id);
  ids_left_over.erase(xaxis_id);
  yaxis_id = *ids_left_over.begin();
  circle(out_img,marker_dots[xaxis_id].pt,20,Scalar(255,255,100),-1);
  circle(out_img,marker_dots[yaxis_id].pt,20,Scalar(0,0,255),-1);

  // find the opposite corner dot...
  // it will be the closest combined distance from the x/y axis that is not the origin dot
  float opp_dist = 1.0E10;
  int opp_id = 0;
  for(size_t i=0;i<trimmed_dot_cx.size();++i){
    float dist_11 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y,marker_line_dots[xaxis_id][0].x,marker_line_dots[xaxis_id][0].y));
    float dist_12 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y,marker_line_dots[xaxis_id][1].x,marker_line_dots[xaxis_id][1].y));
    // take the smaller of the two
    float dist_1 = dist_11 < dist_12 ? dist_11 : dist_12;
    float dist_21 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_line_dots[yaxis_id][0].x,marker_line_dots[yaxis_id][0].y));
    float dist_22 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_line_dots[yaxis_id][1].x,marker_line_dots[yaxis_id][1].y));
    // take the smaller of the two
    float dist_2 = dist_21 < dist_22 ? dist_21 : dist_22;
    float total_dist = dist_1 + dist_2;
    if(total_dist <= opp_dist){
      opp_dist = total_dist;
      opp_id = i;
    }
  }
  circle(out_img,Point(trimmed_dot_cx[opp_id],trimmed_dot_cy[opp_id]),20,Scalar(100,255,100),-1);

  // determine the pattern size
  float axis_tol = 10.0;
  int pattern_width = 0;
  int pattern_height = 0;
  const float dist_xaxis_dot = dist_from_line(marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y,marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y);
  const float dist_yaxis_dot = dist_from_line(marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y,marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y);
  for(size_t i=0;i<trimmed_dot_cx.size();++i){
    float dist_from_y_axis = dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y);
    float dist_from_x_axis = dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y,marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y);
//    std::stringstream dot_text;
//    dot_text << dist_from_x_axis;
//    putText(out_img, dot_text.str(), Point2f(trimmed_dot_cx[i],trimmed_dot_cy[i]) + Point2f(20,20),
//      FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(255,0,0), 1.5, CV_AA);
    const float min_x_range = dist_xaxis_dot < 0.0 ? dist_xaxis_dot : 0.0;
    const float max_x_range = dist_xaxis_dot < 0.0 ? 0.0 : dist_xaxis_dot;
    const float min_y_range = dist_yaxis_dot < 0.0 ? dist_yaxis_dot : 0.0;
    const float max_y_range = dist_yaxis_dot < 0.0 ? 0.0 : dist_yaxis_dot;
    if(std::abs(dist_from_x_axis) < axis_tol && dist_from_y_axis > min_x_range && dist_from_y_axis < max_x_range){
      circle(out_img,Point(trimmed_dot_cx[i],trimmed_dot_cy[i]),20,Scalar(200,200,200),-1);
      pattern_width++;
    }
    else if(std::abs(dist_from_y_axis) < axis_tol && dist_from_x_axis > min_y_range && dist_from_x_axis < max_y_range){
      circle(out_img,Point(trimmed_dot_cx[i],trimmed_dot_cy[i]),20,Scalar(200,200,200),-1);
      pattern_height++;
    }
  }
  if(pattern_width<=1||pattern_height<=1){
    std::cout << "error, could not determine the pattern size" << std::endl;
    if(output_preview_images)
      imwrite(output_image_filename, out_img);
    return -3;
  }
  // add the marker dots
  pattern_width += 2;
  pattern_height += 2;
  DEBUG_MSG("pattern size: " << pattern_width << " x " << pattern_height);
  const float rough_grid_spacing = dist_xaxis_dot / (pattern_width-1);

  // now determine the 6 parameter image warp using the four corners
  std::vector<scalar_t> proj_xl(4,0.0);
  std::vector<scalar_t> proj_yl(4,0.0);
  std::vector<scalar_t> proj_xr(4,0.0);
  std::vector<scalar_t> proj_yr(4,0.0);
  // board space coords for the three marker dots and the opposite corner dot
  proj_xl[0] = 0.0;               proj_yl[0] = 0.0;
  proj_xl[1] = pattern_width-1.0; proj_yl[1] = 0.0;
  proj_xl[2] = pattern_width-1.0; proj_yl[2] = pattern_height-1.0;
  proj_xl[3] = 0.0;               proj_yl[3] = pattern_height-1.0;
  // image coords for the marker dots
  proj_xr[0] = marker_dots[origin_id].pt.x; proj_yr[0] = marker_dots[origin_id].pt.y;
  proj_xr[1] = marker_dots[xaxis_id].pt.x;  proj_yr[1] = marker_dots[xaxis_id].pt.y;
  proj_xr[2] = trimmed_dot_cx[opp_id]; proj_yr[2] = trimmed_dot_cy[opp_id];
  proj_xr[3] = marker_dots[yaxis_id].pt.x;  proj_yr[3] = marker_dots[yaxis_id].pt.y;
  Teuchos::SerialDenseMatrix<int_t,double> proj_matrix = compute_affine_matrix(proj_xl,proj_yl,proj_xr,proj_yr);

  // convert all the points in the grid to the image and draw a dot in the image:
  std::vector<size_t> closest_neigh_id(1);
  std::vector<scalar_t> neigh_dist_2(1);

  // add the marker dots to the return data
  image_points.push_back(Point2f(marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y));
  object_points.push_back(Point3f(0,0,0));
  image_points.push_back(Point2f(marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y));
  object_points.push_back(Point3f((pattern_width-1)*pattern_size,0,0));
  image_points.push_back(Point2f(marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y));
  object_points.push_back(Point3f(0,(pattern_height-1)*pattern_size,0));

  // make sure that a dot is only found once
  std::set<int> found_list;
  // make sure all the dots inside the marker dots are found
  bool interior_dot_failed = false;
  for(int j=-2;j<pattern_height+2;++j){
    for(int i=-2;i<pattern_width+2;++i){
      float grid_x = i;
      float grid_y = j;
      float image_x = 0.0;
      float image_y = 0.0;
      project_grid_to_image(proj_matrix,grid_x,grid_y,image_x,image_y);
      if(image_x<=5||image_y<=5||image_x>=img.cols-5||image_y>=img.rows-5) continue;
      // skip the axis and origin points
      if(i==0&&j==0) continue;
      if(i==pattern_width-1&&j==0) continue;
      if(i==0&&j==pattern_height-1) continue;
      // find the closest contour near this grid point and
      query_pt[0] = image_x;
      query_pt[1] = image_y;
      // find the closest dots to this point
      kd_tree->knnSearch(&query_pt[0], 1, &closest_neigh_id[0], &neigh_dist_2[0]);
      int neigh_id = closest_neigh_id[0];
      if(neigh_dist_2[0]<=(0.5*rough_grid_spacing)*(0.5*rough_grid_spacing)&&found_list.find(neigh_id)==found_list.end()){
        found_list.insert(neigh_id);
        image_points.push_back(Point2f(trimmed_dot_cx[neigh_id],trimmed_dot_cy[neigh_id]));
        object_points.push_back(Point3f(i*pattern_size,j*pattern_size,0));
        std::stringstream dot_text;
        dot_text << i << "," << j;
        putText(out_img, dot_text.str(), Point2f(trimmed_dot_cx[neigh_id],trimmed_dot_cy[neigh_id]) + Point2f(20,20),
          FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(255,0,0), 1, cv::LINE_AA);
        circle(out_img,Point(trimmed_dot_cx[neigh_id],trimmed_dot_cy[neigh_id]),10,Scalar(255,0,255),-1);
      } else if (i>=0&&i<pattern_width&&j>=0&&j<pattern_height){
        interior_dot_failed = true;
      }
    }
  }
  DEBUG_MSG("Found " << image_points.size() << " cal dots");
//  for(size_t i=0;i<image_points.size();++i){
//    DEBUG_MSG(object_points[i].x << " " << object_points[i].y << " " << image_points[i].x << " " << image_points[i].y);
//  }
  if(output_preview_images)
    imwrite(output_image_filename, out_img);
  // check that there are the proper number of points:
  if(interior_dot_failed){
    std::cout << "error, all interior dots could not be located" << std::endl;
    return -3;
  }
  if((int)image_points.size()<pattern_width*pattern_height||(image_points.size()!=object_points.size())){
    std::cout << "error, not enough points located" << std::endl;
    return -3;
  }
  return 0;
}

DICE_LIB_DLL_EXPORT
float
StereoCalib(const int mode,
  const vector<string>& imagelist,
  const int board_width,
  const int board_height,
  const float & squareSize,
  const int binary_threshold,
  const bool useCalibrated,
  const bool showRectified,
  const std::string & output_filename)
{
  if( imagelist.size() % 2 != 0 )
  {
    cout << "Error: the image list contains odd (non-even) number of elements\n";
    return -1.0;
  }
  const int maxScale = 2;
  // ARRAY AND VECTOR STORAGE:
  cv::Size boardSize;
  boardSize.width = board_width;
  boardSize.height = board_height;

  vector<vector<Point2f> > imagePoints[2];
  vector<vector<Point3f> > objectPoints;
  Size imageSize;

  int i, j, k, nimages = (int)imagelist.size()/2;

  imagePoints[0].resize(nimages);
  imagePoints[1].resize(nimages);
  vector<string> goodImageList;

  for( i = j = 0; i < nimages; i++ )
  {
    for( k = 0; k < 2; k++ )
    {
      const string& filename = imagelist[i*2+k];
      std::cout << "processing cal image " << filename << std::endl;
      Mat img = imread(filename, IMREAD_GRAYSCALE);
      if(img.empty()){
        std::cout << "image mat is empty " << std::endl;
        break;
      }
      if( imageSize == Size() )
        imageSize = img.size();
      else if( img.size() != imageSize )
      {
        cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
        break;
      }
      bool found = false;
      vector<Point2f>& corners = imagePoints[k][j];
      for( int scale = 1; scale <= maxScale; scale++ )
      {
        Mat timg;
        if( scale == 1 )
          timg = img;
        else
          resize(img, timg, Size(), scale, scale);
        if(mode==1 || mode == 2){
          // binary image
          Mat bi_src(timg.size(), timg.type());
          // apply thresholding
          threshold(timg, bi_src, binary_threshold, 255, cv::THRESH_BINARY);
          // invert the source image
          Mat not_src(bi_src.size(), bi_src.type());
          bitwise_not(bi_src, not_src);
          // find the dots with holes as blobs
          SimpleBlobDetector::Params params;
          params.maxArea = 10e4;
          params.minArea = 100;
          cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
          std::vector<KeyPoint> keypoints;
          if(mode==1){ // base image is white on black so invert to make the white dots black
            detector->detect( not_src, keypoints );
          }else{
            detector->detect( bi_src, keypoints );
          }
          // draw result of dots with holes keypoints
          std::stringstream corner_filename;
          // strip off the extention from the original file name
          size_t lastindex = filename.find_last_of(".");
          string rawname = filename.substr(0, lastindex);
          corner_filename << rawname << "_markers.png";
          //Mat im_with_keypoints;
          //if(mode==1){
          //  drawKeypoints( not_src, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          //}else{
          //  drawKeypoints( bi_src, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          //}
          //imwrite(corner_filename.str().c_str(), im_with_keypoints);
          for(size_t i=0;i<keypoints.size();++i)
            std::cout << "keypoint " << keypoints[i].pt.x << " " << keypoints[i].pt.y << std::endl;
          if(keypoints.size()!=3){
            std::cout << "Error: only three marker dots should be found for circle grids, but found " << keypoints.size() << std::endl;
            found = false;
            break;
          }
          // squared distance between key point 0 and kp1:
          float dist_01 = (keypoints[1].pt.x - keypoints[0].pt.x)*(keypoints[1].pt.x - keypoints[0].pt.x) + (keypoints[1].pt.y - keypoints[0].pt.y)*(keypoints[1].pt.y - keypoints[0].pt.y);
          // squared distance between key point 1 and kp2:
          float dist_12 = (keypoints[2].pt.x - keypoints[1].pt.x)*(keypoints[2].pt.x - keypoints[1].pt.x) + (keypoints[2].pt.y - keypoints[1].pt.y)*(keypoints[2].pt.y - keypoints[1].pt.y);
          // squared distance between key point 0 and kp2:
          float dist_02 = (keypoints[2].pt.x - keypoints[0].pt.x)*(keypoints[2].pt.x - keypoints[0].pt.x) + (keypoints[2].pt.y - keypoints[0].pt.y)*(keypoints[2].pt.y - keypoints[0].pt.y);
          //std::cout << "distances: " << dist_01 << " " << dist_12 << " " << dist_02 << std::endl;
          // determine the corner opposite the hypot:
          float opposite_corner_x = 0.0;
          float opposite_corner_y = 0.0;
          int origin_id = 0;
          int y_axis_id = 0;
          int x_axis_id = 0;
          if(dist_01 > dist_12 && dist_01 > dist_02){
            float dx = keypoints[2].pt.x - keypoints[0].pt.x;
            float dy = keypoints[2].pt.y - keypoints[0].pt.y;
            opposite_corner_x = keypoints[1].pt.x - dx;
            opposite_corner_y = keypoints[1].pt.y - dy;
            origin_id = 2;
            x_axis_id = 0;
            y_axis_id = 1;
          }else if(dist_12 > dist_01 && dist_12 > dist_02){
            float dx = keypoints[0].pt.x - keypoints[1].pt.x;
            float dy = keypoints[0].pt.y - keypoints[1].pt.y;
            opposite_corner_x = keypoints[2].pt.x - dx;
            opposite_corner_y = keypoints[2].pt.y - dy;
            origin_id = 0;
            x_axis_id = 1;
            y_axis_id = 2;
          }else{
            float dx = keypoints[1].pt.x - keypoints[0].pt.x;
            float dy = keypoints[1].pt.y - keypoints[0].pt.y;
            opposite_corner_x = keypoints[2].pt.x - dx;
            opposite_corner_y = keypoints[2].pt.y - dy;
            origin_id = 1;
            x_axis_id = 0;
            y_axis_id = 2;
          }
          // compute the centroid of the box:
          float cx = (keypoints[0].pt.x + keypoints[1].pt.x + keypoints[2].pt.x + opposite_corner_x)/4.0;
          float cy = (keypoints[0].pt.y + keypoints[1].pt.y + keypoints[2].pt.y + opposite_corner_y)/4.0;
          // determine the blob spacing in pixels
          float blob_dim = boardSize.height > boardSize.width ? boardSize.height -1 : boardSize.width - 1;
          if(blob_dim==0.0){
            std::cout << "Error: invalid board size " << std::endl;
            found = false;
            break;
          }
          assert(blob_dim!=0.0);
          float factor = 1.0/blob_dim;
          //std::cout << "factor " << factor << std::endl;

          // determine the y axis point and the x axis point, now that the origin is known
          float cross = keypoints[x_axis_id].pt.x*keypoints[y_axis_id].pt.y - keypoints[x_axis_id].pt.y*keypoints[y_axis_id].pt.x;
          if(cross < 0.0){
            int x_temp = x_axis_id;
            x_axis_id = y_axis_id;
            y_axis_id = x_temp;
          }
          //std::cout << " origin: " << keypoints[origin_id].pt.x << " " << keypoints[origin_id].pt.y << std::endl;
          //std::cout << " x axis: " << keypoints[x_axis_id].pt.x << " " << keypoints[x_axis_id].pt.y << std::endl;
          //std::cout << " y axis: " << keypoints[y_axis_id].pt.x << " " << keypoints[y_axis_id].pt.y << std::endl;

          // expand the four corners by the factor
          std::vector<float> box_x;
          box_x.push_back(factor*(keypoints[y_axis_id].pt.x - cx) + keypoints[y_axis_id].pt.x);
          box_x.push_back(factor*(keypoints[origin_id].pt.x - cx) + keypoints[origin_id].pt.x);
          box_x.push_back(factor*(keypoints[x_axis_id].pt.x - cx) + keypoints[x_axis_id].pt.x);
          box_x.push_back(factor*(opposite_corner_x - cx) + opposite_corner_x);
          box_x.push_back(factor*(keypoints[y_axis_id].pt.x - cx) + keypoints[y_axis_id].pt.x);
          std::vector<float> box_y;
          box_y.push_back(factor*(keypoints[y_axis_id].pt.y - cy) + keypoints[y_axis_id].pt.y);
          box_y.push_back(factor*(keypoints[origin_id].pt.y - cy) + keypoints[origin_id].pt.y);
          box_y.push_back(factor*(keypoints[x_axis_id].pt.y - cy) + keypoints[x_axis_id].pt.y);
          box_y.push_back(factor*(opposite_corner_y - cy) + opposite_corner_y);
          box_y.push_back(factor*(keypoints[y_axis_id].pt.y - cy) + keypoints[y_axis_id].pt.y);
//          for(int_t i=0;i<5;++i){
//            std::cout << "box corner: " << box_x[i] << " " << box_y[i] << std::endl;
//          }

          // now detect the blobs in the original binary src image
          std::vector<KeyPoint> dots;
          if(mode==1){
            detector->detect( bi_src, dots );
          }
          else{
            detector->detect( not_src, dots );
          }
          //Mat im_with_dots;
          //drawKeypoints( bi_src, dots, im_with_dots, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          // Show blobs
          //std::stringstream dots_filename;
          //dots_filename << rawname << "_dots.png";
          //imwrite(dots_filename.str().c_str(), im_with_dots);

          // compute the average dot size:
          float avg_dot_size = 0.0;
          assert(dots.size()>0);
          for(size_t i=0;i<dots.size();++i){
            //std::cout << " dot " << i << " size " << dots[i].size << std::endl;
            avg_dot_size += dots[i].size;
          }
          avg_dot_size /= dots.size();
          //std::cout << " average dot size " << avg_dot_size << std::endl;

          // filter dots based on avg size and whether the dots fall in the central box
          // create a point cloud
          Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
          point_cloud->pts.resize(boardSize.width*boardSize.height);
          // add the axes keypoints:
          int_t pt_index = 0;
          for(;pt_index<3;++pt_index){
            point_cloud->pts[pt_index].x = keypoints[pt_index].pt.x;
            point_cloud->pts[pt_index].y = keypoints[pt_index].pt.y;
          }
          for(size_t i=0;i<dots.size();++i){
            if(dots[i].size < 0.8f*avg_dot_size) continue;
            if(!is_in_quadrilateral(dots[i].pt.x,dots[i].pt.y,box_x,box_y)) continue;
            point_cloud->pts[pt_index].x = dots[i].pt.x;
            point_cloud->pts[pt_index++].y = dots[i].pt.y;
          }
          if(pt_index!=boardSize.height*boardSize.width){
            std::cout << "Error: needed to find " << boardSize.height*boardSize.width << " dots, but found " << pt_index << " instead" << std::endl;
            found = false;
            break;
          }
          DEBUG_MSG("building the kd-tree");
          Teuchos::RCP<kd_tree_2d_t> kd_tree =
              Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
          kd_tree->buildIndex();
          DEBUG_MSG("kd-tree completed");
          scalar_t query_pt[2];
          const int_t num_neigh = 1;
          std::vector<size_t> ret_index(num_neigh);
          std::vector<scalar_t> out_dist_sqr(num_neigh);
          // iterate all points in the grid and find the closest dot
          float total_height = keypoints[y_axis_id].pt.y - keypoints[origin_id].pt.y;
          float offset_x = keypoints[y_axis_id].pt.x - keypoints[origin_id].pt.x;
          float delta_h = total_height / (boardSize.height - 1);
          float delta_ox = offset_x / (boardSize.height - 1);
          float total_width = keypoints[x_axis_id].pt.x - keypoints[origin_id].pt.x;
          float offset_y = keypoints[x_axis_id].pt.y - keypoints[origin_id].pt.y;
          float delta_w = total_width / (boardSize.width - 1);
          float delta_oy = offset_y / (boardSize.width - 1);
          std::vector<Point2f> keep_dots;
          for(int_t j=0;j<boardSize.height;++j){
            float row_origin_x = keypoints[x_axis_id].pt.x + delta_ox * j;
            float row_origin_y = keypoints[x_axis_id].pt.y + delta_h * j;
            for(int_t i=0;i<boardSize.width;++i){
              query_pt[0] = row_origin_x - delta_w * i;
              query_pt[1] = row_origin_y - delta_oy * i;
              //std::cout << " point " << j*boardSize.width + i << " " << ptx << " " << pty << std::endl;
              // find the closest dot to this point and add it to the ordered list:
              kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
              keep_dots.push_back(Point2f((float)point_cloud->pts[ret_index[0]].x,(float)point_cloud->pts[ret_index[0]].y));
            }
          }
          if((int_t)keep_dots.size()!=boardSize.height*boardSize.width){
            std::cout << "Error: needed to find " << boardSize.height*boardSize.width << " dots, but found " << keep_dots.size() << " instead" << std::endl;
            found = false;
            break;
          }
          //          Mat im_with_keep_dots;
          //          drawKeypoints( bi_src, keep_keypoints, im_with_keep_dots, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          //          // Show blobs
          //          std::stringstream keepdotname;
          //          keepdotname << "keepdots_" << id_num << ".png";
          //          imwrite(keepdotname.str().c_str(), im_with_keep_dots);

          Mat(keep_dots).copyTo(corners);
          found = true;
          //          SimpleBlobDetector::Params params;
          //          params.maxArea = 10e4;
          //          Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
          //          found = findCirclesGrid( timg, boardSize, corners, CALIB_CB_ASYMMETRIC_GRID, detector);//, blobDetector);//CALIB_CB_ASYMMETRIC_GRID + CALIB_CB_CLUSTERING);//,
          //          //  // Draw detected blobs as red circles.
          //          //  // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
          ////            Mat im_with_keypoints;
          ////            drawKeypoints( img, corners, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          ////          //
          ////            static int_t image_id = 0;
          ////            image_id++;
          ////            std::stringstream image_name;
          ////            image_name << "res_keypoints_" << image_id << ".png";
          ////          //  // Show blobs
          ////          //  //imshow("keypoints", im_with_keypoints );
          ////            imwrite(image_name.str().c_str(), im_with_keypoints);
        }else if(mode==0){
          found = findChessboardCorners(timg, boardSize, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        }
        else{
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid calibration mode: " << mode);
        }
        if( found )
        {
          if( scale > 1 )
          {
            Mat cornersMat(corners);
            cornersMat *= 1./scale;
          }
          break;
        }
      }
      if( !found ){
        std::cout << "corners not found" << std::endl;
        break;
      }
      if(mode==0){
        cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
          TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
            30, 0.01));
      }
    }
    if( k == 2 )
    {
      goodImageList.push_back(imagelist[i*2]);
      goodImageList.push_back(imagelist[i*2+1]);
      j++;
    }
  }
  cout << j << " pairs have been successfully detected.\n";
  nimages = j;
  if( nimages < 2 )
  {
    cout << "Error: too little pairs to run the calibration\n";
    return -1.0;
  }

  imagePoints[0].resize(nimages);
  imagePoints[1].resize(nimages);
  objectPoints.resize(nimages);

  for( i = 0; i < nimages; i++ )
  {
    for( j = 0; j < boardSize.height; j++ )
      for( k = 0; k < boardSize.width; k++ ){
        if(mode==0){
          objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
        } else {
          objectPoints[i].push_back(Point3f((boardSize.width-1-k)*squareSize,(boardSize.height-1-j)*squareSize, 0));
        }
      }
  }

  cout << "Running stereo calibration ...\n";

//  std::cout << "object points has num entries " << objectPoints[0].size() << std::endl;
//  for(int i=0;i<objectPoints[0].size(); ++i)
//    std::cout << objectPoints[0][i].x << " " << objectPoints[0][i].y << std::endl;
//  std::cout << "image points has num entries " << imagePoints[0][0].size() << std::endl;
//  for(int i=0;i<imagePoints[0][0].size(); ++i)
//    std::cout << imagePoints[0][0][i].x << " " << imagePoints[0][0][i].y << " " << imagePoints[1][0][i].x << " " << imagePoints[1][0][i].y << std::endl;

  Mat cameraMatrix[2], distCoeffs[2];
  cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePoints[0],imageSize,0);
  cameraMatrix[1] = initCameraMatrix2D(objectPoints,imagePoints[1],imageSize,0);

//  std::cout << cameraMatrix[0] << std::endl;
//  std::cout << cameraMatrix[1] << std::endl;

//  Mat leftCameraMatrix, leftDistCoeffs, rvecs, tvecs;
//  std::cout << " image size " << imageSize.width << " x " << imageSize.height << std::endl;
//  double rms_left = calibrateCamera(objectPoints, imagePoints[0], imageSize, leftCameraMatrix,
//                              leftDistCoeffs, rvecs, tvecs, CV_CALIB_ZERO_TANGENT_DIST|CV_CALIB_FIX_K2|CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5|CV_CALIB_FIX_K6);
//
//  std::cout << leftCameraMatrix << std::endl;
//  std::cout << leftDistCoeffs << std::endl;
//
//
//  Mat rightCameraMatrix, rightDistCoeffs, rvecs_r, tvecs_r;
//  double rms_right = calibrateCamera(objectPoints, imagePoints[1], imageSize, rightCameraMatrix,
//                              rightDistCoeffs, rvecs_r, tvecs_r, CV_CALIB_ZERO_TANGENT_DIST|CV_CALIB_FIX_K2|CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5|CV_CALIB_FIX_K6);
//
//  std::cout << rightCameraMatrix << std::endl;
//  std::cout << rightDistCoeffs << std::endl;

  //assert(false);
  //Mat cameraMatrix[2], distCoeffs[2];
  //cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePoints[0],imageSize,0);
  //cameraMatrix[1] = initCameraMatrix2D(objectPoints,imagePoints[1],imageSize,0);
  Mat R, T, E, F;

  double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
    cameraMatrix[0], distCoeffs[0],
    cameraMatrix[1], distCoeffs[1],
    imageSize, R, T, E, F,
    CALIB_FIX_ASPECT_RATIO +
    CALIB_USE_INTRINSIC_GUESS +
    CALIB_SAME_FOCAL_LENGTH +
    CALIB_ZERO_TANGENT_DIST +
    CALIB_RATIONAL_MODEL +
    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
  cout << "done with RMS error=" << rms << endl;

  std::cout << cameraMatrix[0] << std::endl;
  std::cout << distCoeffs[0] << std::endl;
  std::cout << cameraMatrix[1] << std::endl;
  std::cout << distCoeffs[1] << std::endl;

  // CALIBRATION QUALITY CHECK
  // because the output fundamental matrix implicitly
  // includes all the output information,
  // we can check the quality of calibration using the
  // epipolar geometry constraint: m2^t*F*m1=0
  double err = 0;
  int npoints = 0;
  vector<Vec3f> lines[2];
  for( i = 0; i < nimages; i++ )
  {
    int npt = (int)imagePoints[0][i].size();
    Mat imgpt[2];
    for( k = 0; k < 2; k++ )
    {
      imgpt[k] = Mat(imagePoints[k][i]);
      undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
      computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
    }
    double imgErr = 0.0;
    for( j = 0; j < npt; j++ )
    {
      double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
        imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
            fabs(imagePoints[1][i][j].x*lines[0][j][0] +
              imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
      err += errij;
      imgErr += errij;
    }
    cout << "image: " << goodImageList[i*2] << " error " << imgErr << endl;

    npoints += npt;
  }
  assert(npoints!=0);
  cout << "average epipolar err = " <<  err/npoints << endl;

  // save intrinsic parameters
  std::FILE * filePtr = fopen(output_filename.c_str(),"w");
  fprintf(filePtr,"%4.12E # cx\n",cameraMatrix[0].at<double>(0,2));
  fprintf(filePtr,"%4.12E # cy\n",cameraMatrix[0].at<double>(1,2));
  fprintf(filePtr,"%4.12E # fx\n",cameraMatrix[0].at<double>(0,0));
  fprintf(filePtr,"%4.12E # fy\n",cameraMatrix[0].at<double>(1,1));
  fprintf(filePtr,"%4.12E # fs\n",cameraMatrix[0].at<double>(0,1));
  fprintf(filePtr,"%4.12E # k1\n",distCoeffs[0].at<double>(0,0)); //
  fprintf(filePtr,"%4.12E # k2\n",distCoeffs[0].at<double>(0,1)); //
  fprintf(filePtr,"%4.12E # k6\n",distCoeffs[0].at<double>(0,7)); //
  fprintf(filePtr,"%4.12E # cx\n",cameraMatrix[1].at<double>(0,2));
  fprintf(filePtr,"%4.12E # cy\n",cameraMatrix[1].at<double>(1,2));
  fprintf(filePtr,"%4.12E # fx\n",cameraMatrix[1].at<double>(0,0));
  fprintf(filePtr,"%4.12E # fy\n",cameraMatrix[1].at<double>(1,1));
  fprintf(filePtr,"%4.12E # fs\n",cameraMatrix[1].at<double>(0,1));
  fprintf(filePtr,"%4.12E # k1\n",distCoeffs[1].at<double>(0,0)); //
  fprintf(filePtr,"%4.12E # k2\n",distCoeffs[1].at<double>(0,1)); //
  fprintf(filePtr,"%4.12E # k6\n",distCoeffs[1].at<double>(0,7)); //

//  FileStorage fs(intrinsic_filename.c_str(), FileStorage::WRITE);
//  if( fs.isOpened() )
//  {
//    fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
//        "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
//    fs.release();
//  }
//  else
//    cout << "Error: can not save the intrinsic parameters\n";

  Mat R1, R2, P1, P2, Q;
  Rect validRoi[2];

  stereoRectify(cameraMatrix[0], distCoeffs[0],
    cameraMatrix[1], distCoeffs[1],
    imageSize, R, T, R1, R2, P1, P2, Q,
    CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
  // rotation matrix
  fprintf(filePtr,"%4.12E # R11\n",R.at<double>(0,0));
  fprintf(filePtr,"%4.12E # R12\n",R.at<double>(0,1));
  fprintf(filePtr,"%4.12E # R13\n",R.at<double>(0,2));
  fprintf(filePtr,"%4.12E # R21\n",R.at<double>(1,0));
  fprintf(filePtr,"%4.12E # R22\n",R.at<double>(1,1));
  fprintf(filePtr,"%4.12E # R23\n",R.at<double>(1,2));
  fprintf(filePtr,"%4.12E # R31\n",R.at<double>(2,0));
  fprintf(filePtr,"%4.12E # R32\n",R.at<double>(2,1));
  fprintf(filePtr,"%4.12E # R33\n",R.at<double>(2,2));
  // translations
  fprintf(filePtr,"%4.12E # tx\n",T.at<double>(0,0));
  fprintf(filePtr,"%4.12E # ty\n",T.at<double>(1,0));
  fprintf(filePtr,"%4.12E # tz\n",T.at<double>(2,0));

//  fs.open(extrinsic_filename.c_str(), FileStorage::WRITE);
//  if( fs.isOpened() )
//  {
//    fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
//    fs.release();
//  }
//
//
//  else
//    cout << "Error: can not save the extrinsic parameters\n";

  fclose (filePtr);

  // OpenCV can handle left-right
  // or up-down camera arrangements
  bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

  // COMPUTE AND DISPLAY RECTIFICATION
  if( !showRectified )
    return rms;

  Mat rmap[2][2];
  // IF BY CALIBRATED (BOUGUET'S METHOD)
  if( useCalibrated )
  {
    // we already computed everything
  }
  // OR ELSE HARTLEY'S METHOD
  else
    // use intrinsic parameters of each camera, but
    // compute the rectification transformation directly
    // from the fundamental matrix
  {
    vector<Point2f> allimgpt[2];
    for( k = 0; k < 2; k++ )
    {
      for( i = 0; i < nimages; i++ )
        std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
    }
    F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
    Mat H1, H2;
    stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

    R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
    R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
    P1 = cameraMatrix[0];
    P2 = cameraMatrix[1];
  }

  //Precompute maps for cv::remap()
  initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
  initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

  Mat canvas;
  double sf;
  int w, h;
  if( !isVerticalStereo )
  {
    sf = 600./MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width*sf);
    h = cvRound(imageSize.height*sf);
    canvas.create(h, w*2, CV_8UC3);
  }
  else
  {
    sf = 300./MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width*sf);
    h = cvRound(imageSize.height*sf);
    canvas.create(h*2, w, CV_8UC3);
  }

  for( i = 0; i < nimages; i++ )
  {
    for( k = 0; k < 2; k++ )
    {
      Mat img = imread(goodImageList[i*2+k], 0), rimg, cimg;
      remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
      cvtColor(rimg, cimg, COLOR_GRAY2BGR);
      Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
      resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
      if( useCalibrated )
      {
        Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
        rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
      }
    }

    if( !isVerticalStereo )
      for( j = 0; j < canvas.rows; j += 16 )
        line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
    else
      for( j = 0; j < canvas.cols; j += 16 )
        line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);
    char c = (char)waitKey();
    if( c == 27 || c == 'q' || c == 'Q' )
      break;
  }
  return rms;
}

} // end DICe namespace
