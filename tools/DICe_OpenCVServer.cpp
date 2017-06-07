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

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

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

  // read the list of images:
  // all the arguments that start with a dot are images
  std::vector<std::string> images;
  int_t end_images = 0;
  for(int_t i=1;i<argc;++i){
    std::string arg = argv[i];
    if(arg[0]=='.'){
      images.push_back(arg);
      DEBUG_MSG("Adding image: " << arg);
    }else{
      end_images = i;
      break;
    }
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

  DEBUG_MSG("number of filters: " << filters.size());
  if(filters.size()!=filter_params.size()){
    std::cout << "filters vec and filter params vec are not the same size" << std::endl;
    return -1;
  }
  for(size_t i=0;i<filters.size();++i){
    DEBUG_MSG("filter: " << filters[i]);
    for(size_t j=0;j<filter_params[i].size();++j)
      DEBUG_MSG("    parameter: " << filter_params[i][j]);
  }

  for(size_t image_it=0;image_it<images.size();++image_it){
    DEBUG_MSG("processing image " << images[image_it]);

    // load the image as an openCV mat
    Mat img = imread(images[image_it], IMREAD_GRAYSCALE);
    if(img.empty()){
      std::cout << "the image is empty" << std::endl;
      return -1;
    }
    if(!img.data){
      std::cout << "the image failed to load" << std::endl;
      return -1;
    }
    for(size_t filter_it=0;filter_it<filters.size();++filter_it){
      DEBUG_MSG("processing filter " << filter_it);

      if(filters[filter_it]=="Filter:BinaryThreshold"){
        // set the parameters
        if(filter_params[filter_it].size()!=4){
          std::cout << "the number of parameters for a binary filter should be 4, not " << filter_params[filter_it].size() << std::endl;
          return -1;
        }
        int filterMode = CV_ADAPTIVE_THRESH_GAUSSIAN_C;
        if(filter_params[filter_it][0]==1.0) filterMode = CV_ADAPTIVE_THRESH_MEAN_C;
        int invertedMode = CV_THRESH_BINARY;
        if(filter_params[filter_it][3]==1.0) invertedMode = CV_THRESH_BINARY_INV;
        double blockSize = filter_params[filter_it][1];
        double binaryConstant = filter_params[filter_it][2];
        DEBUG_MSG("applying a binary threshold");
        adaptiveThreshold(img,img,255,filterMode,invertedMode,blockSize,binaryConstant);
        DEBUG_MSG("binary threshold successful");
      } // end binary filter

      if(filters[filter_it]=="Filter:Blob"){
        if(filter_params[filter_it].size()!=12){
          std::cout << "the number of parameters for a blob filter should be 12, not " << filter_params[filter_it].size() << std::endl;
          return -1;
        }
        // find the dots with holes as blobs
        SimpleBlobDetector::Params params;
        // area
        if(filter_params[filter_it][0]!=0.0){
          params.filterByArea = true;
          params.minArea = filter_params[filter_it][1];
          params.maxArea = filter_params[filter_it][2];
        }
        // circularity
        if(filter_params[filter_it][3]!=0.0){
          params.filterByCircularity = true;
          params.minCircularity = filter_params[filter_it][4];
          params.maxCircularity = filter_params[filter_it][5];
        }
        // eccentricity
        if(filter_params[filter_it][6]!=0.0){
          params.filterByInertia = true;
          params.minInertiaRatio = filter_params[filter_it][7];
          params.maxInertiaRatio = filter_params[filter_it][8];
        }
        // convexity
        if(filter_params[filter_it][9]!=0.0){
          params.filterByConvexity = true;
          params.minConvexity = filter_params[filter_it][10];
          params.maxConvexity = filter_params[filter_it][11];
        }
        cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
        std::vector<KeyPoint> keypoints;
        detector->detect( img, keypoints );
        drawKeypoints( img, keypoints, img, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
      } // end blob filter

    } // end filters
    size_t lastindex = images[image_it].find_last_of(".");
    std::string rawname = images[image_it].substr(0, lastindex);
    rawname += "_filter.png";
    imwrite(rawname, img);
  } // end images




  DICe::finalize();

  return 0;
}

