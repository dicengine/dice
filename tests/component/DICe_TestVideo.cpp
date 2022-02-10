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
#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_ImageIO.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include "opencv2/opencv.hpp"

#include <iostream>


using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // TODO add .avi example (uncomment the code below) when we sort out how to read avi on mac

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

  // create a basline image to use as a

  const int_t w = 100;
  const int_t h = 100;
  const int_t num_frames = 5;
  int_t fps = 1;

  // Code used to create the videos:

//  int_t codec_mp4 = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
////  int_t codec_avi = cv::VideoWriter::fourcc('m', 'j', 'p', 'g');
//  int_t codec_avi = cv::VideoWriter::fourcc('I', '4', '2', '0');
////  int_t codec_avi = cv::VideoWriter::fourcc('P', 'I', 'M', '1');
//  cv::VideoWriter video_mp4("video.mp4",codec_mp4,fps, cv::Size(w,h),false);
//  cv::VideoWriter video_avi("video.avi",codec_avi,fps, cv::Size(w,h),false);
//  for(int_t frame=0;frame<num_frames;++frame){
//    cv::Mat img = cv::Mat::zeros(cv::Size(w,h),CV_8UC1);
//    for(size_t j=0;j<h;++j){
//      for(size_t i=0;i<w;++i){
//        size_t value = i+j+frame;
////        //Note: in the images that are read back in, the failed pixels should be filtered out
////        if(i==j&&i%10==0&&i!=0){ // place failed pixels along the diagonal
////          value = 255;
////        }
//        img.at<uchar>(j,i) = value;
//      }
//    }
//    video_mp4.write(img);
//    video_avi.write(img);
//  }
//  video_mp4.release();
//  video_avi.release();

  // test frame width/height
  int_t w_mp4 = 0, h_mp4 = 0;
//  int_t w_avi = 0, h_avi = 0;
  DICe::utils::read_image_dimensions("./images/video_0.mp4",w_mp4,h_mp4);
//  DICe::utils::read_image_dimensions("video_0.avi",w_avi,h_avi);
  if(w_mp4!=w){
    errorFlag++;
    *outStream << "Error, mp4 width " << w_mp4 << " is not correct " << w << std::endl;
  }
  if(h_mp4!=h){
    errorFlag++;
    *outStream << "Error, mp4 height " << h_mp4 << " is not correct " << h << std::endl;
  }
//  if(w_avi!=w){
//    errorFlag++;
//    *outStream << "Error, avi width " << w_avi << " is not correct " << w << std::endl;
//  }
//  if(h_avi!=h){
//    errorFlag++;
//    *outStream << "Error, avi height " << h_mp4 << " is not correct " << h << std::endl;
//  }
  Teuchos::RCP<cv::VideoCapture> vc_mp4 = DICe::utils::Video_Singleton::instance().video_capture("./images/video.mp4");
//  Teuchos::RCP<cv::VideoCapture> vc_avi = DICe::utils::Video_Singleton::instance().video_capture("video.avi");
  if(vc_mp4->get(cv::CAP_PROP_FRAME_WIDTH)!=w){
    errorFlag++;
    *outStream << "Error, vc mp4 width " << vc_mp4->get(cv::CAP_PROP_FRAME_WIDTH) << " is not correct " << w << std::endl;
  }
  if(vc_mp4->get(cv::CAP_PROP_FRAME_HEIGHT)!=h){
    errorFlag++;
    *outStream << "Error, vc mp4 height " << vc_mp4->get(cv::CAP_PROP_FRAME_HEIGHT) << " is not correct " << h << std::endl;
  }
//  if(vc_avi->get(cv::CAP_PROP_FRAME_WIDTH)!=w){
//    errorFlag++;
//    *outStream << "Error, vc avi width " << vc_avi->get(cv::CAP_PROP_FRAME_WIDTH) << " is not correct " << w << std::endl;
//  }
//  if(vc_avi->get(cv::CAP_PROP_FRAME_HEIGHT)!=h){
//    errorFlag++;
//    *outStream << "Error, vc avi height " << vc_avi->get(cv::CAP_PROP_FRAME_HEIGHT) << " is not correct " << h << std::endl;
//  }
  if(vc_mp4->get(cv::CAP_PROP_FPS)!=fps){
    errorFlag++;
    *outStream << "Error fps is not correct for mp4" << std::endl;
  }
//  if(vc_avi->get(cv::CAP_PROP_FPS)!=fps){
//    errorFlag++;
//    *outStream << "Error fps is not correct for avi" << std::endl;
//  }
  if(vc_mp4->get(cv::CAP_PROP_FRAME_COUNT)!=num_frames){
    errorFlag++;
    *outStream << "Error fps is not correct for mp4" << std::endl;
  }
//  if(vc_avi->get(cv::CAP_PROP_FRAME_COUNT)!=num_frames){
//    errorFlag++;
//    *outStream << "Error fps is not correct for avi" << std::endl;
//  }

  const int_t tol = 20; // counts (loose due to compression)
  bool count_fail_mp4 = false;
//  bool count_fail_avi = false;

  for(int_t frame=0;frame<num_frames;++frame){
    std::stringstream img_name_mp4, img_name_avi;
    img_name_mp4 << "./images/video_" << frame << ".mp4";
//    img_name_avi << "video_" << frame << ".avi";
    cv::Mat img_mp4 = DICe::utils::read_image(img_name_mp4.str().c_str());
//    cv::Mat img_avi = DICe::utils::read_image(img_name_avi.str().c_str());
    for(size_t j=0;j<h;++j){
      for(size_t i=0;i<w;++i){
        size_t value = i+j+frame;
        if(std::abs((scalar_t)img_mp4.at<uchar>(j,i)-value)>tol){
          *outStream << "mp4 pixel value fail. is " << (int_t)img_mp4.at<uchar>(j,i) << " should be " << value << std::endl;
          count_fail_mp4 = true;
        }
//        if(std::abs((scalar_t)img_avi.at<uchar>(j,i)-value)>tol){
//          *outStream << "avi pixel value fail. is " << (int_t)img_avi.at<uchar>(j,i) << " should be " << value << std::endl;
//          count_fail_avi = true;
//        }
      }
    }
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

