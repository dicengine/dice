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

/*! \file  DICe_OpenCVServer.cpp
    \brief Utility server to perform OpenCV operations on an image, used by the GUI
*/

#include <DICe.h>
//#include <DICe_Shape.h>
//#include <DICe_PointCloud.h>
//#include <DICe_Triangulation.h>
#include <DICe_StereoCalib.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
//#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_ParameterList.hpp>
//#include <Teuchos_LAPACK.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <cassert>

using namespace cv;
using namespace DICe;

int main(int argc, char *argv[]) {

  /// usage ./DICe_OpenCVServer <image1> <image2> ... <Filter:filter1> <args> <Filter:filter2> <args>

  DICe::initialize(argc, argv);

  //Teuchos::oblackholestream bhs; // outputs nothing
  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);
  std::string delimiter = " ,\r";

  DEBUG_MSG("User specified " << argc << " arguments");
  //for(int_t i=0;i<argc;++i){
  //  DEBUG_MSG(argv[i]);
  //}

  int error_code = 0;

  // read the list of images:
  // all the arguments that have a file extension and are not a number or filter are images
  // the ordering of images is in input/output pairs ...
  std::vector<std::string> input_images;
  std::vector<std::string> output_images;
  int_t end_images = 0;
  bool i_o_flag = true;
  for(int_t i=1;i<argc;++i){
    std::string arg = argv[i];
    if(arg.find('.')!=std::string::npos&&!std::isdigit(arg[0])&&arg.find(':')==std::string::npos){
      if(i_o_flag){
        input_images.push_back(arg);
        DEBUG_MSG("Adding input image: " << arg);
      }else{
        if(arg.find(".png")==std::string::npos){
          std::cout << "error, invalid output image format, (only .png is allowed)" << std::endl;
          return -1;
        }
        output_images.push_back(arg);
        DEBUG_MSG("Adding output image: " << arg);
      }
      i_o_flag = !i_o_flag;
    }else{
      end_images = i;
      break;
    }
  }
  // make sure that end_images is odd at this point since the i/o image names have to come in pairs
  if(input_images.size()==0||input_images.size()!=output_images.size()||!end_images%2){
    std::cout << "error, invalid input/output images, num input " << input_images.size() << " num output " << output_images.size() << " " << end_images%2 << std::endl;
    return -1;
  }

  std::vector<std::string> filters;
  std::vector<std::vector <scalar_t> > filter_params;
  // read the list of filters:
  // (for now only one filter allowed: binary thresholding)
  for(int_t i=end_images;i<argc;++i){
    std::string arg = argv[i];
    if(arg.find("Filter")!=std::string::npos){
      // read all the parameters for this filter
      filter_params.push_back(std::vector<scalar_t>());
      filters.push_back(arg);
      for(int_t ii=i+1;ii<argc;++ii){
        std::string argi = argv[ii];
        if(argi.find("Filter")!=std::string::npos){
          break;
        }else{
          filter_params[filter_params.size()-1].push_back(std::strtod(argv[ii],NULL));
          i++;
        }
      }
    }
  }

  // filter parameters:
  int filterMode = -1;
  int invertedMode = -1;
  //int antiInvertedMode = -1;
  double blockSize = -1.0;
  double binaryConstant = -1.0;
  int board_width = 0;
  int board_height = 0;
  float pattern_spacing = 0.0;
  SimpleBlobDetector::Params params;

  // check if this is a calibration preview and also the filters applied:
  bool is_cal = false;
  bool has_binary = false;
  bool has_blob = false;
  bool preview_thresh = false;
  bool has_adaptive = false;

  // set up the filters
  DEBUG_MSG("number of filters: " << filters.size());
  if(filters.size()!=filter_params.size()){
    std::cout << "error, filters vec and filter params vec are not the same size" << std::endl;
    return -1;
  }
  for(size_t i=0;i<filters.size();++i){
    DEBUG_MSG("filter: " << filters[i]);
    for(size_t j=0;j<filter_params[i].size();++j)
      DEBUG_MSG("    parameter: " << filter_params[i][j]);
    if(filters[i]=="Filter:CalPreview"){
      DEBUG_MSG("CalPreview filter is active");
      is_cal = true;
    }
    if(filters[i]=="Filter:PreviewThreshold"){
      DEBUG_MSG("Preview images will use thresholded image for background");
      preview_thresh = true;
    }
    if(filters[i]=="Filter:AdaptiveThreshold"){
      DEBUG_MSG("Preview images will use adaptive thresholding");
      has_adaptive = true;
    }
    if(filters[i]=="Filter:BinaryThreshold"){
      has_binary = true;
      if(filter_params[i].size()!=4){
        std::cout << "error, the number of parameters for a binary filter should be 4, not " << filter_params[i].size() << std::endl;
        return -1;
      }
      filterMode = CV_ADAPTIVE_THRESH_GAUSSIAN_C;
      if(filter_params[i][0]==1.0) filterMode = CV_ADAPTIVE_THRESH_MEAN_C;
      invertedMode = CV_THRESH_BINARY;
      //antiInvertedMode = CV_THRESH_BINARY_INV;
      if(filter_params[i][3]==1.0){
        invertedMode = CV_THRESH_BINARY_INV;
        //antiInvertedMode = CV_THRESH_BINARY;
      }
      blockSize = filter_params[i][1];
      binaryConstant = filter_params[i][2];
    } // end binary filter
    if(filters[i]=="Filter:BoardSize"){
      if(filter_params[i].size()!=2){
        std::cout << "error, the number of parameters for a board size should be 2, not " << filter_params[i].size() << std::endl;
        return -1;
      }
      board_width = filter_params[i][0];
      board_height = filter_params[i][1];
      std::cout << "board size " << board_width << " x " << board_height << std::endl;
    }
    if(filters[i]=="Filter:PatternSpacing"){
      if(filter_params[i].size()!=1){
        std::cout << "error, the number of parameters for pattern spacing should be 1, not " << filter_params[i].size() << std::endl;
        return -1;
      }
      pattern_spacing = filter_params[i][0];
      std::cout << "pattern spacing " << pattern_spacing << std::endl;
    }
    if(filters[i]=="Filter:Blob"){
      has_blob = true;
      if(filter_params[i].size()!=12){
        std::cout << "error, the number of parameters for a blob filter should be 12, not " << filter_params[i].size() << std::endl;
        return -1;
      }
      // find the dots with holes as blobs
      // area
      if(filter_params[i][0]!=0.0){
        params.filterByArea = true;
        params.minArea = filter_params[i][1];
        params.maxArea = filter_params[i][2];
      }
      // circularity
      if(filter_params[i][3]!=0.0){
        params.filterByCircularity = true;
        params.minCircularity = filter_params[i][4];
        params.maxCircularity = filter_params[i][5];
      }
      // eccentricity
      if(filter_params[i][6]!=0.0){
        params.filterByInertia = true;
        params.minInertiaRatio = filter_params[i][7];
        params.maxInertiaRatio = filter_params[i][8];
      }
      // convexity
      if(filter_params[i][9]!=0.0){
        params.filterByConvexity = true;
        params.minConvexity = filter_params[i][10];
        params.maxConvexity = filter_params[i][11];
      }
    } // end blob filter
  } // end loop over filters

  Teuchos::RCP<Teuchos::ParameterList> preview_params = rcp(new Teuchos::ParameterList());
  preview_params->set("preview_thresh",preview_thresh);
  preview_params->set("has_adaptive",has_adaptive);
  preview_params->set("filterMode",filterMode);
  preview_params->set("invertedMode",invertedMode);
  //preview_params->set("antiInvertedMode",antiInvertedMode);
  preview_params->set("blockSize",blockSize);
  preview_params->set("binaryConstant",binaryConstant);
  preview_params->set("pattern_spacing",pattern_spacing);

  for(size_t image_it=0;image_it<input_images.size();++image_it){
    DEBUG_MSG("processing image " << input_images[image_it]);

    if(is_cal){
      if(!has_blob || !has_binary){
        std::cout << "error, cal preview requires blob and binary filters to be active" << std::endl;
        return -1;
      }
      std::vector<Point2f> image_points;
      std::vector<Point3f> object_points;
      // find the cal dot points and check show a preview of their locations
      const int pre_code = pre_process_cal_image(input_images[image_it],output_images[image_it],preview_params,
        image_points,object_points);
      DEBUG_MSG("pre_process_cal_image return value: " << pre_code);
      if(pre_code==-1)
        return -1;
      else if(pre_code==-4) // image load failure
        return 9;
      else if(pre_code!=0){
        if(image_it==0)
          error_code = 2; // left alone
        else if(image_it==1&&error_code==0)
          error_code = 3; // right alone
        else if(image_it==1&&error_code!=0)
          error_code = 4; // left and right failed
        else if(image_it==2&&error_code==0)
          error_code = 5; // middle alone
        else if(image_it==2&&error_code==2)
          error_code = 6; // middle and left
        else if(image_it==2&&error_code==3)
          error_code = 7; // middle and right
        else if(image_it==2&&error_code==4)
          error_code = 8; // all
        continue;
      } // pre_code != 0
    } // end is cal
    else{
      // load the image as an openCV mat
      Mat img = imread(input_images[image_it], IMREAD_GRAYSCALE);
//      Mat binary_img(img.size(),CV_8UC3);
//      //Mat contour_img(img.size(), CV_8UC3,Scalar::all(255));
//      Mat out_img(img.size(), CV_8UC3);
//      Mat bi_copy_img(img.size(), CV_8UC3);
//      cvtColor(img, out_img, CV_GRAY2RGB);

      //Mat out_img = imread(input_images[image_it], IMREAD_GRAYSCALE);
      if(img.empty()){
        std::cout << "error, the image is empty" << std::endl;
        return -1;
      }
      if(!img.data){
        std::cout << "error, the image failed to load" << std::endl;
        return -1;
      }
      for(size_t filter_it=0;filter_it<filters.size();++filter_it){
        DEBUG_MSG("processing filter " << filter_it);

        if(filters[filter_it]=="Filter:BinaryThreshold"){
          DEBUG_MSG("applying a binary threshold");
          adaptiveThreshold(img,img,255,filterMode,invertedMode,blockSize,binaryConstant);
          DEBUG_MSG("binary threshold successful");
        } // end binary filter

        if(filters[filter_it]=="Filter:Blob"){
          DEBUG_MSG("applying a blob detector");
          cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
          std::vector<KeyPoint> keypoints;
          detector->detect( img, keypoints );
          drawKeypoints( img, keypoints, img, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          DEBUG_MSG("blob detector successful");
        } // end blob filter

      } // end filters
      DEBUG_MSG("writing output image");
      imwrite(output_images[image_it], img);
    }

//    // pre-processing for CalPreview
//    if(is_cal){
//      if(!has_blob || !has_binary){
//        std::cout << "error, cal preview requires blob and binary filters to be active" << std::endl;
//        return -1;
//      }
//      // blur the image to remove noise
//      GaussianBlur(img, binary_img, Size(9, 9), 2, 2 );
//
//      // binary threshold to create black and white image
//      if(has_adaptive)
//        adaptiveThreshold(binary_img,binary_img,255,filterMode,invertedMode,blockSize,binaryConstant);
//      else
//        threshold(binary_img, binary_img, binaryConstant, 255, CV_THRESH_BINARY);
//      //threshold(binary_img, binary_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
//
//      // output the preview images using the thresholded image as the background
//      if(preview_thresh){
//        cvtColor(binary_img, out_img, CV_GRAY2RGB);
//      }
//
//      //medianBlur ( median_img, median_img, 7 );
//
//      // find the contours in the image with parents and children to locate the doughnut looking dots
//      std::vector<std::vector<Point> > contours;
//      std::vector<std::vector<Point> > trimmed_contours;
//      std::vector<Vec4i> hierarchy;
//      findContours(binary_img.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );
//      // make sure the vectors are the same size:
//      if(hierarchy.size()!=contours.size()){
//        std::cout << "error, hierarchy and contours vectors are not the same size" << std::endl;
//        return -1;
//      }
//
//      //std::vector<RotatedRect> minEllipse( contours.size() );
//
//      // count up the number of potential candidates for marker blobs
//      size_t min_marker_blob_size = 75;
//      std::vector<KeyPoint> potential_markers;
//      // global centroid of the potential markers
//      //float centroid_x = (float)binary_img.cols/2;//0.0;
//      //float centroid_y = (float)binary_img.rows/2;//0.0;
//      for(size_t idx=0; idx<hierarchy.size(); ++idx){
//          if(hierarchy[idx][3]!=-1&&hierarchy[idx][2]!=-1&&contours[idx].size()>min_marker_blob_size){
//            // compute the centroid of this marker
//            float cx = 0.0;
//            float cy = 0.0;
//            for(size_t i=0;i<contours[idx].size();++i){
//              cx += contours[idx][i].x;
//              cy += contours[idx][i].y;
//            }
//            float denom = (float)contours[idx].size();
//            cx /= denom;
//            cy /= denom;
//            potential_markers.push_back(KeyPoint(cx,cy,contours[idx].size()));
//            trimmed_contours.push_back(contours[idx]);
//            //centroid_x += cx;
//            //centroid_y += cy;
//          }
//      }
//      if(potential_markers.size() < 3){
//        // output the image with markers located
//        for(size_t i=0;i<potential_markers.size();++i)
//          circle(out_img,potential_markers[i].pt,20,Scalar(255,255,100),-1);
//        imwrite(output_images[image_it], out_img);
//        // error codes: 2 left only 3 right only 4 left and right 5 middle only 6 middle and left 7 middle and right 8 all three
//        if(image_it==0)
//          error_code = 2; // left alone
//        else if(image_it==1&&error_code==0)
//          error_code = 3; // right alone
//        else if(image_it==1&&error_code!=0)
//          error_code = 4; // left and right failed
//        else if(image_it==2&&error_code==0)
//          error_code = 5; // middle alone
//        else if(image_it==2&&error_code==2)
//          error_code = 6; // middle and left
//        else if(image_it==2&&error_code==3)
//          error_code = 7; // middle and right
//        else if(image_it==2&&error_code==4)
//          error_code = 8; // all
//        continue;
//      }
//      if(potential_markers.size()==0){
//        std::cout << "error, no marker dots could be located" << std::endl;
//        return -1;
//      }
//      if(potential_markers.size()!=trimmed_contours.size()){
//        std::cout << "error, potential markers and trimmed contours should be the same size" << std::endl;
//        return -1;
//      }
//      //centroid_x /= (float)potential_markers.size();
//      //centroid_y /= (float)potential_markers.size();
//
//      // final set of marker dots
//      std::vector<KeyPoint> marker_dots(3);
//      for(size_t i=0;i<3;++i)
//        marker_dots[i] = potential_markers[i];
//
//      // downselect to the three dots closest to the centroid if there are more than three
//      if(potential_markers.size()>3){
//
//        std::vector<std::pair<int,float> > id_quant;
//
//        // create a copy of the binary image with the contours turned white:
//        bi_copy_img = binary_img.clone();
//        drawContours(bi_copy_img, trimmed_contours, -1, Scalar::all(255), CV_FILLED, 8);
//        // invert the image
//        bitwise_not(bi_copy_img, bi_copy_img);
//
//        for(size_t idx=0;idx<trimmed_contours.size();++idx){
//          // determine the bounding box for each contour:
//          Rect boundingBox = boundingRect(trimmed_contours[idx]);
//          // find the center of the box
//          int center_x = (boundingBox.br().x + boundingBox.tl().x)/2;
//          int center_y = (boundingBox.br().y + boundingBox.tl().y)/2;
//          int x_span = (boundingBox.br().x - center_x)*1.5; // expand 1.5 times the original width and height
//          int y_span = (boundingBox.br().y - center_y)*1.5;
//
//          //rectangle( bi_copy_img, boundingBox.tl(), boundingBox.br(), Scalar(0,255,0), 2, 8, 0 );
//          // iterate over the box and count up the intensity values from the copied binary image
//          float total_intensity = 0.0;
//          for(int j=center_y-y_span;j<center_y+y_span;++j){
//            if(j<0||j>bi_copy_img.rows)continue;
//            for(int i=center_x-x_span;i<center_x+x_span;++i){
//              if(i<0||i>bi_copy_img.cols)continue;
//              total_intensity += (float)bi_copy_img.at<uchar>(j,i);
//            }
//          }
//          //std::cout << " QUANTITY FOR CONTOUR: " << total_intensity << std::endl;
//          id_quant.push_back(std::pair<int,float>(idx,total_intensity));
//        }
//        // order the vector by smallest quantity
//        std::sort(id_quant.begin(), id_quant.end(), pair_compare);
//        // take the top three as the markers:
//        for(size_t i=0;i<3;++i){
//          marker_dots[i] = potential_markers[id_quant[i].first];
//        }
//      }
////      for(size_t i=0;i<3;++i){
////        circle(out_img,marker_dots[i].pt,20,Scalar(255,255,100),-1);
////      }
//
//
//      // compute the median dot size:
//      std::vector<std::pair<int,float> > id_size;
//      for(size_t idx=0; idx<hierarchy.size(); ++idx){
//          if(hierarchy[idx][3]!=-1&&hierarchy[idx][2]==-1&&contours[idx].size()>min_marker_blob_size){
//            id_size.push_back(std::pair<int,float>(idx,contours[idx].size()));
//          }
//      }
//      int median_id = id_size.size()/2;
//      std::sort(id_size.begin(), id_size.end(), pair_compare);
//      const float median_size = id_size[median_id].second;
//
//      // now that the corner marker dots are found, look for the rest of the dots:
//      // create a copy of the binary image with the contours turned white:
//      bi_copy_img = binary_img.clone();
//      std::vector<std::vector<Point> > trimmed_dot_contours;
//      std::vector<float> trimmed_dot_cx;
//      std::vector<float> trimmed_dot_cy;
//
//      for(size_t idx=0; idx<hierarchy.size(); ++idx){
//          if(hierarchy[idx][3]!=-1&&hierarchy[idx][2]==-1&&contours[idx].size()>0.8*median_size&&contours[idx].size()<1.2*median_size){
//            trimmed_dot_contours.push_back(contours[idx]);
//            //std::cout << " found contour size " << contours[idx].size() << std::endl;
//            float cx = 0.0;
//            float cy = 0.0;
//            for(size_t i=0;i<contours[idx].size();++i){
//              cx += contours[idx][i].x;
//              cy += contours[idx][i].y;
//            }
//            float denom = (float)contours[idx].size();
//            cx /= denom;
//            cy /= denom;
//            trimmed_dot_cx.push_back(cx);
//            trimmed_dot_cy.push_back(cy);
//          }
//      }
//      //drawContours(out_img, trimmed_dot_contours, -1, Scalar(255,255,100), CV_FILLED, 8);
//      if(trimmed_dot_cy.size()!=trimmed_dot_cx.size()||trimmed_dot_cy.size()!=trimmed_dot_contours.size()){
//        std::cout << "error, centroid vectors are the wrong size" << std::endl;
//        return -1;
//      }
//
//      // iterate the marker dots and gather the closest two non-colinear points
//      Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
//      point_cloud->pts.resize(trimmed_dot_contours.size());
//      for(size_t i=0;i<trimmed_dot_contours.size();++i){
//        point_cloud->pts[i].x = trimmed_dot_cx[i];
//        point_cloud->pts[i].y = trimmed_dot_cy[i];
//      }
//
//      DEBUG_MSG("building the kd-tree");
//      Teuchos::RCP<kd_tree_2d_t> kd_tree =
//          Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
//      kd_tree->buildIndex();
//      DEBUG_MSG("kd-tree completed");
//      scalar_t query_pt[2];
//      int_t num_neigh = 4;
//      std::vector<size_t> ret_index(num_neigh);
//      std::vector<scalar_t> out_dist_sqr(num_neigh);
//
//      std::vector<std::vector<Point> > marker_line_dots(3);
//      //const float dist_tol = 200.0;
//      for(size_t i=0;i<marker_dots.size();++i){
//        query_pt[0] = marker_dots[i].pt.x;
//        query_pt[1] = marker_dots[i].pt.y;
//        // find the closest dots to this point
//        kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
//        // let the first closest dot define one of the cardinal directions for this marker dot
//        marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[0]],trimmed_dot_cy[ret_index[0]]));
//        // get another dot that is not colinear to define the second cardinal direction for this marker
//        const float dist_1 = std::abs(dist_from_line(trimmed_dot_cx[ret_index[1]],trimmed_dot_cy[ret_index[1]],marker_dots[i].pt.x,marker_dots[i].pt.y,trimmed_dot_cx[ret_index[0]],trimmed_dot_cy[ret_index[0]]));
//        const float dist_2 = std::abs(dist_from_line(trimmed_dot_cx[ret_index[2]],trimmed_dot_cy[ret_index[2]],marker_dots[i].pt.x,marker_dots[i].pt.y,trimmed_dot_cx[ret_index[0]],trimmed_dot_cy[ret_index[0]]));
//        const float dist_3 = std::abs(dist_from_line(trimmed_dot_cx[ret_index[3]],trimmed_dot_cy[ret_index[3]],marker_dots[i].pt.x,marker_dots[i].pt.y,trimmed_dot_cx[ret_index[0]],trimmed_dot_cy[ret_index[0]]));
//        if(dist_1>=dist_2&&dist_1>=dist_3)
//          marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[1]],trimmed_dot_cy[ret_index[1]]));
//        if(dist_2>=dist_1&&dist_2>=dist_3)
//          marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[2]],trimmed_dot_cy[ret_index[2]]));
//        else
//          marker_line_dots[i].push_back(Point(trimmed_dot_cx[ret_index[3]],trimmed_dot_cy[ret_index[3]]));
//        if(marker_line_dots[i].size()!=2){
//          std::cout << "error, marker line dots vector is the wrong size" << std::endl;
//          return -1;
//        }
//        //circle(out_img,marker_line_dots[i][0],20,Scalar(255,0,0),-1);
//        //circle(out_img,marker_line_dots[i][1],20,Scalar(255,0,0),-1);
//        //drawContours(out_img, marker_line_dots[i], -1, Scalar(255,0,0), CV_FILLED, 8);
//      } // end marker dot loop
//
//      // at this point, it's not known which marker dot is the origin. The marker line dots define two dots for each marker dot, one dot in each cardinal direction.
//      // if a line is extended from each marker dot to the cardinal direction dots, the origin will be the marker that lies closes to the four lines that extend from
//      // the other marker dots
//      float min_dist = 0.0;
//      int origin_id = 0;
//      for(size_t i=0;i<marker_dots.size();++i){
//        float total_dist = 0.0;
//        for(size_t j=0;j<marker_dots.size();++j){
//          if(i==j)continue;
//          float dist_1 = std::abs(dist_from_line(marker_dots[i].pt.x,marker_dots[i].pt.y,marker_dots[j].pt.x,marker_dots[j].pt.y,marker_line_dots[j][0].x,marker_line_dots[j][0].y));
//          float dist_2 = std::abs(dist_from_line(marker_dots[i].pt.x,marker_dots[i].pt.y,marker_dots[j].pt.x,marker_dots[j].pt.y,marker_line_dots[j][1].x,marker_line_dots[j][1].y));
//          total_dist += dist_1 + dist_2;
//        }
//        if(total_dist < min_dist || i==0){
//          min_dist = total_dist;
//          origin_id = i;
//        }
//      }
//      // draw the origin marker dot
//      circle(out_img,marker_dots[origin_id].pt,20,Scalar(0,255,255),-1);
//      // compute the distance between the origin and the other marker dots and compare with the board size to get x and y axes
//      float max_dist = 0.0;
//      int xaxis_id = 0;
//      int yaxis_id = 0;
//      for(size_t i=0;i<marker_dots.size();++i){
//        if((int)i==origin_id) continue;
//        float dist = (marker_dots[origin_id].pt.x - marker_dots[i].pt.x)*(marker_dots[origin_id].pt.x - marker_dots[i].pt.x) +
//            (marker_dots[origin_id].pt.y - marker_dots[i].pt.y)*(marker_dots[origin_id].pt.y - marker_dots[i].pt.y);
//        if(dist>max_dist){
//          max_dist = dist;
//          xaxis_id = i;
//        }else{
//          yaxis_id = i;
//        }
//      }
//      // note: shortest distance is always y
////      if(board_width < board_height){
////        int temp_id = xaxis_id;
////        xaxis_id = yaxis_id;
////        yaxis_id = temp_id;
////      }
//      circle(out_img,marker_dots[xaxis_id].pt,20,Scalar(255,255,100),-1);
//      circle(out_img,marker_dots[yaxis_id].pt,20,Scalar(0,0,255),-1);
//
//      // find the opposite corner dot...
//      // it will be the closest combined distance from x/y axis that is not the origin dot
//      float opp_dist = 1.0E10;
//      int opp_id = 0;
//      for(size_t i=0;i<trimmed_dot_contours.size();++i){
//        float dist_11 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y,marker_line_dots[xaxis_id][0].x,marker_line_dots[xaxis_id][0].y));
//        float dist_12 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y,marker_line_dots[xaxis_id][1].x,marker_line_dots[xaxis_id][1].y));
//        // take the smaller of the two
//        float dist_1 = dist_11 < dist_12 ? dist_11 : dist_12;
//        float dist_21 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_line_dots[yaxis_id][0].x,marker_line_dots[yaxis_id][0].y));
//        float dist_22 = std::abs(dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_line_dots[yaxis_id][1].x,marker_line_dots[yaxis_id][1].y));
//        // take the smaller of the two
//        float dist_2 = dist_21 < dist_22 ? dist_21 : dist_22;
//        float total_dist = dist_1 + dist_2;
//        //std::stringstream dot_text;
//        //dot_text << total_dist;
//        //putText(out_img, dot_text.str(), Point2f(trimmed_dot_cx[i],trimmed_dot_cy[i]) + Point2f(20,20),
//        //  FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255,0,0), 1.5, CV_AA);
//        if(total_dist <= opp_dist){
//          opp_dist = total_dist;
//          opp_id = i;
//        }
//      }
//      // TODO test for tol from line...
//      circle(out_img,Point(trimmed_dot_cx[opp_id],trimmed_dot_cy[opp_id]),20,Scalar(100,255,100),-1);
//
//      // determine the pattern size
//      float axis_tol = 10.0;
//      int pattern_width = 0;
//      int pattern_height = 0;
//      const float dist_xaxis_dot = dist_from_line(marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y,marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y);
//      const float dist_yaxis_dot = dist_from_line(marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y,marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y);
//      for(size_t i=0;i<trimmed_dot_contours.size();++i){
//        float dist_x = dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y,marker_dots[xaxis_id].pt.x,marker_dots[xaxis_id].pt.y);
//        float dist_y = dist_from_line(trimmed_dot_cx[i],trimmed_dot_cy[i],marker_dots[yaxis_id].pt.x,marker_dots[yaxis_id].pt.y,marker_dots[origin_id].pt.x,marker_dots[origin_id].pt.y);
//        if(std::abs(dist_x) < axis_tol && dist_y > 0.0 && dist_y < dist_xaxis_dot){
//          circle(out_img,Point(trimmed_dot_cx[i],trimmed_dot_cy[i]),20,Scalar(0,255,100),-1);
//          pattern_width++;
//        }
//        else if(std::abs(dist_y) < axis_tol && dist_x > 0.0 && dist_x < dist_yaxis_dot){
//          circle(out_img,Point(trimmed_dot_cx[i],trimmed_dot_cy[i]),20,Scalar(255,255,255),-1);
//          pattern_height++;
//        }
//      }
//      if(pattern_width<=1||pattern_height<=1){
//        std::cout << "error, could not determine the pattern size" << std::endl;
//        return -1;
//      }
//      // add the marker dots
//      pattern_width += 2;
//      pattern_height += 2;
//      std::cout << "pattern size: " << pattern_width << " x " << pattern_height << std::endl;
//      const float rough_grid_spacing = dist_xaxis_dot / (pattern_width-1);
//
//
//      // now determine the 6 parameter image warp using the four corners
//      std::vector<scalar_t> proj_xl(4,0.0);
//      std::vector<scalar_t> proj_yl(4,0.0);
//      std::vector<scalar_t> proj_xr(4,0.0);
//      std::vector<scalar_t> proj_yr(4,0.0);
//      // board space coords
//      proj_xl[0] = 0.0;               proj_yl[0] = 0.0;
//      proj_xl[1] = pattern_width-1.0; proj_yl[1] = 0.0;
//      proj_xl[2] = pattern_width-1.0; proj_yl[2] = pattern_height-1.0;
//      proj_xl[3] = 0.0;               proj_yl[3] = pattern_height-1.0;
//      // image coords
//      proj_xr[0] = marker_dots[origin_id].pt.x; proj_yr[0] = marker_dots[origin_id].pt.y;
//      proj_xr[1] = marker_dots[xaxis_id].pt.x;  proj_yr[1] = marker_dots[xaxis_id].pt.y;
//      proj_xr[2] = trimmed_dot_cx[opp_id]; proj_yr[2] = trimmed_dot_cy[opp_id];
//      proj_xr[3] = marker_dots[yaxis_id].pt.x;  proj_yr[3] = marker_dots[yaxis_id].pt.y;
//
//      Teuchos::SerialDenseMatrix<int_t,double> proj_matrix = compute_affine_matrix(proj_xl,proj_yl,proj_xr,proj_yr);
//      std::cout << "PROJECTION MATRIX " << std::endl;
//      std::cout << proj_matrix << std::endl;
//
//      // convert all the points in the grid to the image and draw a dot in the image:
//      std::vector<size_t> closest_neigh_id(1);
//      std::vector<scalar_t> neigh_dist_2(1);
//
//      for(int j=-3;j<pattern_height+3;++j){
//        for(int i=-3;i<pattern_width+3;++i){
//          float grid_x = i;
//          float grid_y = j;
//          float image_x = 0.0;
//          float image_y = 0.0;
//          project_grid_to_image(proj_matrix,grid_x,grid_y,image_x,image_y);
//          if(image_x<=5||image_y<=5||image_x>=img.cols-5||image_y>=img.rows-5) continue;
//          // TODO add the axis and origin points
//          if(i==0&&j==0) continue;
//          if(i==pattern_width-1&&j==0) continue;
//          if(i==0&&j==pattern_height-1) continue;
//
//          // find the closest contour near this grid point and
//          query_pt[0] = image_x;
//          query_pt[1] = image_y;
//          // find the closest dots to this point
//          kd_tree->knnSearch(&query_pt[0], 1, &closest_neigh_id[0], &neigh_dist_2[0]);
//          int neigh_id = closest_neigh_id[0];
//          if(neigh_dist_2[0]<=(0.5*rough_grid_spacing)*(0.5*rough_grid_spacing)){
//            circle(out_img,Point(trimmed_dot_cx[neigh_id],trimmed_dot_cy[neigh_id]),10,Scalar(255,0,255),-1);
//          }
////          std::stringstream dot_text;
////          dot_text << i <<"," <<j;
////          putText(out_img, dot_text.str(), Point2f(image_x,image_y) + Point2f(20,20),
////            FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255,0,0), 1.5, CV_AA);
////          circle(out_img,Point(image_x,image_y),10,Scalar(0,255,255),-1);
//        }
//      }
//
////      if((int)cal_dots.size() != board_height*board_width - 3){
////        imwrite(output_images[image_it], out_img);
////        std::cout << "error, invalid number of cal dots found" << std::endl;
////        if(image_it==0)
////          error_code = 9; // left alone
////        else if(image_it==1&&error_code==0)
////          error_code = 10; // right alone
////        else if(image_it==1&&error_code!=0)
////          error_code = 11; // left and right failed
////        else if(image_it==2&&error_code==0)
////          error_code = 12; // middle alone
////        else if(image_it==2&&error_code==9)
////          error_code = 13; // middle and left
////        else if(image_it==2&&error_code==10)
////          error_code = 14; // middle and right
////        else if(image_it==2&&error_code==11)
////          error_code = 15; // all
////        continue;
////      }
//
//    } // end is_cal
  } // end images

  DICe::finalize();

  return error_code;
}

