/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warranty, support or any guarantee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008

   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

   OPENCV WEBSITES:
     Homepage:      http://opencv.org
     Online docs:   http://docs.opencv.org
     Q&A forum:     http://answers.opencv.org
     Issue tracker: http://code.opencv.org
     GitHub:        https://github.com/opencv/opencv/
 ************************************************** */

#include <DICe_StereoCalib.h>
#include <DICe_Shape.h>
#include <DICe_PointCloud.h>

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
        if(mode==1||2){
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
          Mat im_with_keypoints;
          if(mode==1){
            drawKeypoints( not_src, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          }else{
            drawKeypoints( bi_src, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          }
          imwrite(corner_filename.str().c_str(), im_with_keypoints);
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
          Mat im_with_dots;
          drawKeypoints( bi_src, dots, im_with_dots, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          // Show blobs
          std::stringstream dots_filename;
          dots_filename << rawname << "_dots.png";
          imwrite(dots_filename.str().c_str(), im_with_dots);

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
