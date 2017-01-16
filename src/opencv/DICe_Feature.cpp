// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 Sandia Corporation.
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#include <DICe_Feature.h>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <cstdio>

namespace DICe {

DICE_LIB_DLL_EXPORT
void match_features(const std::string & left_file_name,
  const std::string & right_file_name,
  std::vector<scalar_t> & left_x,
  std::vector<scalar_t> & left_y,
  std::vector<scalar_t> & right_x,
  std::vector<scalar_t> & right_y,
  const float & feature_tol,
  const std::string & result_image_name){

  left_x.clear();
  left_y.clear();
  right_x.clear();
  right_y.clear();

  DEBUG_MSG("match_features(): initializing OpenCV Mats");
  cv::Mat img1;
  cv::Mat img2;
  img1 = cv::imread(left_file_name.c_str(), cv::IMREAD_GRAYSCALE);
  img2 = cv::imread(right_file_name.c_str(), cv::IMREAD_GRAYSCALE);
  const float nn_match_ratio = 0.6f;   // Nearest neighbor matching ratio

  DEBUG_MSG("match_features(): detect and compute features");

  std::vector<cv::KeyPoint> kpts1, kpts2;
  cv::Mat desc1, desc2;
  cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,0,3,feature_tol,4,4,cv::KAZE::DIFF_PM_G2);
  akaze->detectAndCompute(img1, cv::noArray(), kpts1, desc1);
  akaze->detectAndCompute(img2, cv::noArray(), kpts2, desc2);

  DEBUG_MSG("match_features(): matching features");

  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector< std::vector<cv::DMatch> > nn_matches;
  matcher.knnMatch(desc1, desc2, nn_matches, 2);

  DEBUG_MSG("match_features(): removing outliers");

  std::vector<cv::KeyPoint> matched1, matched2, inliers1, inliers2;
  std::vector<cv::DMatch> good_matches;
  for(size_t i = 0; i < nn_matches.size(); i++) {
      cv::DMatch first = nn_matches[i][0];
      float dist1 = nn_matches[i][0].distance;
      float dist2 = nn_matches[i][1].distance;
      if(dist1 < nn_match_ratio * dist2) {
          matched1.push_back(kpts1[first.queryIdx]);
          matched2.push_back(kpts2[first.trainIdx]);
      }
  }

  for(unsigned i = 0; i < matched1.size(); i++) {
    int new_i = static_cast<int>(inliers1.size());
    inliers1.push_back(matched1[i]);
    inliers2.push_back(matched2[i]);
    good_matches.push_back(cv::DMatch(new_i, new_i, 0));
  }
  assert(inliers1.size()==inliers2.size());
  DEBUG_MSG("match_features(): number of features matched: " << inliers1.size());
  if(inliers1.size()==0)
    DEBUG_MSG("***Warning: no matching features matched");
  left_x.resize(inliers1.size(),0.0);
  left_y.resize(inliers1.size(),0.0);
  right_x.resize(inliers1.size(),0.0);
  right_y.resize(inliers1.size(),0.0);
  for(unsigned i = 0; i < left_x.size(); i++) {
    left_x[i] = inliers1[i].pt.x;
    left_y[i] = inliers1[i].pt.y;
    right_x[i] = inliers2[i].pt.x;
    right_y[i] = inliers2[i].pt.y;
  }
  // draw results image if requested
  if(result_image_name!=""){
    cv::Mat res;
    cv::drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    cv::imwrite(result_image_name.c_str(), res);
  }
}


DICE_LIB_DLL_EXPORT
void match_features(Teuchos::RCP<Image> left_image,
  Teuchos::RCP<Image> right_image,
  std::vector<scalar_t> & left_x,
  std::vector<scalar_t> & left_y,
  std::vector<scalar_t> & right_x,
  std::vector<scalar_t> & right_y,
  const float & feature_tol,
  const std::string & result_image_name){

  if(left_image->has_file_name()&&right_image->has_file_name()){
    // see if the image has a valid file name
    match_features(left_image->file_name(),right_image->file_name(),left_x,left_y,right_x,right_y,feature_tol,result_image_name);
  }else{
    left_image->write("left_image.tif");
    right_image->write("right_image.tif");
    match_features("left_image.tif","right_image.tif",left_x,left_y,right_x,right_y,feature_tol,result_image_name);
  }
    //TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Feature detection only enabled for images read from file, not array images");
//#if DICE_USE_DOUBLE
//    // need to convert the double array to a float array
//    Teuchos::ArrayRCP<intensity_t> left_intensities_dbl = left_image->intensities();
//    Teuchos::ArrayRCP<intensity_t> right_intensities_dbl = right_image->intensities();
//    Teuchos::ArrayRCP<float> left_intensities(left_image->width()*left_image->height(),0.0);
//    Teuchos::ArrayRCP<float> right_intensities(right_image->width()*right_image->height(),0.0);
//    for(int_t i=0;i<left_intensities_dbl.size();++i)
//      left_intensities[i] = static_cast<float>(left_intensities_dbl[i]);
//    for(int_t i=0;i<right_intensities_dbl.size();++i)
//      right_intensities[i] = static_cast<float>(right_intensities_dbl[i]);
//#else
//    Teuchos::ArrayRCP<intensity_t> left_intensities = left_image->intensities();
//    Teuchos::ArrayRCP<intensity_t> right_intensities = right_image->intensities();
//#endif
//    cv::Mat img1 = cv::Mat(left_image->width(),left_image->height(),CV_32F,left_intensities.getRawPtr());
//    cv::Mat img2 = cv::Mat(right_image->width(),right_image->height(),CV_32F,right_intensities.getRawPtr());
  if(!(left_image->has_file_name()&&right_image->has_file_name())){
    // remove the left and right image temp files:
    std::remove("left_image.tif"); // delete file
    std::remove("right_image.tif"); // delete file
  }
}




}// End DICe Namespace
