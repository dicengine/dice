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

// references for this test
// https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
// https://scicomp.stackexchange.com/questions/6878/fitting-one-set-of-points-to-another-by-a-rigid-motion
// https://theailearner.com/tag/cv2-getrotationmatrix2d/

#include <DICe.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#ifdef HAVE_OPENCV_CONTRIB
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

#ifdef HAVE_OPENCV_CONTRIB

  // Read in the template image
  cv::Mat temp = cv::imread("./images/GT4-0000_0.tif", cv::ImreadModes::IMREAD_GRAYSCALE);

  // Read in the target image
  cv::Mat targ = cv::imread("./images/GT4-0000_0_70.tif", cv::ImreadModes::IMREAD_GRAYSCALE);

  // Do a SIFT feature match between the two
  cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
  std::vector<cv::KeyPoint> keypointsL, keypointsR;
  cv::Mat descriptors1, descriptors2;
  detector->detectAndCompute(temp, cv::noArray(), keypointsL, descriptors1 );
  detector->detectAndCompute(targ, cv::noArray(), keypointsR, descriptors2 );
  //-- Step 2: Matching descriptor vectors with a FLANN based matcher
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector< std::vector<cv::DMatch> > knn_matches;
  //std::vector<cv::DMatch> good_matches;
  matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.3f;

  std::vector<float> left_pts_x; // all in float due to LAPACK routines needing floats
  std::vector<float> left_pts_y;
  std::vector<float> right_pts_x;
  std::vector<float> right_pts_y;
  left_pts_x.reserve(knn_matches.size());
  left_pts_y.reserve(knn_matches.size());
  right_pts_x.reserve(knn_matches.size());
  right_pts_y.reserve(knn_matches.size());
  float cx = 0.0, cpx = 0.0;
  float cy = 0.0, cpy = 0.0;

  // compute the centroids of the point clouds and normalize the feature points
  for(size_t i = 0; i < knn_matches.size(); i++) {
    if(knn_matches[i].size()<2)continue;
    cv::DMatch first = knn_matches[i][0];
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
      //good_matches.push_back(first);
      left_pts_x.push_back(keypointsL[first.queryIdx].pt.x);
      cx += keypointsL[first.queryIdx].pt.x;
      left_pts_y.push_back(keypointsL[first.queryIdx].pt.y);
      cy += keypointsL[first.queryIdx].pt.y;
      right_pts_x.push_back(keypointsR[first.trainIdx].pt.x);
      cpx += keypointsR[first.trainIdx].pt.x;
      right_pts_y.push_back(keypointsR[first.trainIdx].pt.y);
      cpy += keypointsR[first.trainIdx].pt.y;
    }
  }

  //-- Draw matches
//  cv::Mat img_matches;
//  cv::drawMatches( temp, keypointsL, targ, keypointsR, good_matches, img_matches, cv::Scalar::all(-1),
//               cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//  cv::imwrite("image_matches.png",img_matches);

  const int num_pts = left_pts_x.size();
  TEUCHOS_TEST_FOR_EXCEPTION(num_pts<3,std::runtime_error,"");
  cx/=(float)num_pts;
  cy/=(float)num_pts;
  cpx/=(float)num_pts;
  cpy/=(float)num_pts;
  //std::cout << "cx " << cx << " cy " << cy << " cpx " << cpx << " cpy " << cpy << std::endl;

  // normalize the points
  for(int_t i=0;i<num_pts;++i){
    left_pts_x[i] -= cx;
    left_pts_y[i] -= cy;
    right_pts_x[i] -= cpx;
    right_pts_y[i] -= cpy;
  }

  // assemble the coefficients of X*W*Y' where the weighting matrix W is simply eye(n)
  const int_t D = 2; // two-dimensional
  Teuchos::SerialDenseMatrix<int_t,float> A(D,D,true);
  Teuchos::SerialDenseMatrix<int_t,float> S(D,D,true);
  Teuchos::SerialDenseMatrix<int_t,float> U(D,D,true);
  Teuchos::SerialDenseMatrix<int_t,float> VT(D,D,true);
  for(int_t i=0;i<num_pts;++i){
    A(0,0) += left_pts_x[i]*right_pts_x[i];
    A(1,0) += left_pts_y[i]*right_pts_x[i];
    A(0,1) += left_pts_x[i]*right_pts_y[i];
    A(1,1) += left_pts_y[i]*right_pts_y[i];
  }

  // compute the SVD of the matrix
  Teuchos::LAPACK<int,float> lapack;
  const char JOBU = 'A';
  const char JOBV = 'A';
  const int_t LWORK = 10*D;
  std::vector<float> WORK(LWORK);
  std::vector<float> RWORK(LWORK);
  int_t INFO = 0;
  lapack.GESVD(JOBU,JOBV,D,D,A.values(),D,S.values(),U.values(),D,VT.values(),D,&WORK[0],LWORK,&RWORK[0],&INFO);

  // compute the R = V*U', assuming no reflections (so capital sigma is eye(d)
  const float R00 = VT(0,0)*U(0,0) + VT(1,0)*U(0,1);
  const float R01 = VT(0,0)*U(1,0) + VT(1,0)*U(1,1);
  //const float R10 = VT(0,1)*U(0,0) + VT(1,1)*U(0,1);
  //const float R11 = VT(0,1)*U(1,0) + VT(1,1)*U(1,1);
  //std::cout << " R " << R00 << " " << R01 << " " << R10 << " " << R11 << std::endl;
  // TODO add test that checks the values of R to make sure it's orthogonal

  // NOTE the rotation from Sorkine above is about the origin, in the opencv rotate image routines, the rotation center is specified
  float theta = std::acos(R00) * 180.0 / 3.141592654;
  if(R01 < 0.0) theta = 360.0 - theta;
  //std::cout << "THETA " << theta << std::endl;
  // ALSO NOTE the displacements from Sorkine are from the centroid of the points after they have been rotated about the origin,
  // if you rotate the points around the centroid, the displacements are just the difference of the centroids
  // get the center coordinates of the image to create the 2D rotation matrix
  //const float u = cpx - (R00*cx + R01*cy); // Sorkine displacements (with the points rotated about the origin)
  //const float v = cpy - (R10*cx + R11*cy);
  const float u = cpx - cx;
  const float v = cpy - cy;
  //std::cout << " u " << u << " v " << v << std::endl;

  const float theta_gold = 290.0;
  const float u_gold = 273.408;
  const float v_gold = 174.849;

  const float tol = 0.5;
  if(std::abs(theta - theta_gold) > tol){
    *outStream << "Error, theta is not correct " << theta << " should be " << theta_gold << std::endl;
    errorFlag++;
  }
  if(std::abs(u - u_gold) > tol){
    *outStream << "Error, u is not correct " << u << " should be " << u_gold << std::endl;
    errorFlag++;
  }
  if(std::abs(v - v_gold) > tol){
    *outStream << "Error, v is not correct " << v << " should be " << v_gold << std::endl;
    errorFlag++;
  }

#endif

  // compare with the ground truth using opencv to transform the image using the solution above
//  cv::Point2f center(cx,cy);
//  cv::Mat trans_matrix = cv::getRotationMatrix2D(center, theta, 1.0); // Rotation matrix is 2x3 see https://theailearner.com/tag/cv2-getrotationmatrix2d/ for details
//  // add u and v to the rotation matrix
//  trans_matrix.at<double>(0,2) += u;
//  trans_matrix.at<double>(1,2) += v;
//  cv::Mat trans_image;
//  cv::warpAffine(temp,trans_image,trans_matrix,temp.size());
//  cv::imwrite("trans_im.png",trans_image);

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

