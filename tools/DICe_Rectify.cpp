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
#include <DICe_Triangulation.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <cassert>

using namespace std;
using namespace DICe;

/// Rectifies stereo images

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // read the parameters from the input file:
  if(argc!=4){
    std::cout << "Usage: DICe_Rectify <left_img> <right_img> <cal_file>" << std::endl;
    exit(1);
  }

  const std::string left_img_name = argv[1];
  std::string left_img_out = left_img_name;
  left_img_out.insert(left_img_name.find('.'),"_rect");
  const std::string right_img_name = argv[2];
  std::string right_img_out = right_img_name;
  right_img_out.insert(right_img_name.find('.'),"_rect");
  const std::string cal_file = argv[3];

  std::cout << "DICe_Rectify: left image " << left_img_name << " right image " << right_img_name << " calibration file " << cal_file << std::endl;

  cv::Mat left_img = cv::imread(left_img_name, cv::ImreadModes::IMREAD_GRAYSCALE);
  cv::Mat right_img = cv::imread(right_img_name, cv::ImreadModes::IMREAD_GRAYSCALE);
  const int img_w = left_img.cols;
  const int img_h = left_img.rows;

  std::cout << "image dimensions w " << img_w <<  " h " << img_h << std::endl;

  Teuchos::RCP<Teuchos::ParameterList> input_params = Teuchos::rcp( new Teuchos::ParameterList());
  input_params->set("camera_system_file",cal_file);
  DICe::update_vic3d_cal_input(input_params,img_w,img_h); // some vivc3d formats don't have the image width and height as parameters, if that's the case, add them manually

  Teuchos::RCP<DICe::Triangulation> tri = Teuchos::rcp(new DICe::Triangulation(input_params->get<std::string>("camera_system_file")));
  cv::Mat M1 = tri->camera_matrix(0);
  cv::Mat M2 = tri->camera_matrix(1);
  cv::Mat D1 = tri->distortion_matrix(0);
  cv::Mat D2 = tri->distortion_matrix(1);
  cv::Mat R  = tri->rotation_matrix();
  cv::Mat T  = tri->translation_matrix();
//  std::cout << " M1 " << M1 << std::endl;
//  std::cout << " D1 " << D1 << std::endl;
//  std::cout << " M2 " << M2 << std::endl;
//  std::cout << " D2 " << D2 << std::endl;
//  std::cout << " R " << R << std::endl;
//  std::cout << " T " << T << std::endl;


  cv::Mat R1, R2, P1, P2, map11, map12, map21, map22;

  // TODO possibly store the disparity to depth matrix

  cv::stereoRectify(M1, D1, M2, D2, left_img.size(), R, T, R1, R2, P1, P2, cv::noArray(), 0);

//  std::cout << " R1 " << R1 << std::endl;
//  std::cout << " R2 " << R2 << std::endl;
//  std::cout << " P1 " << P1 << std::endl;
//  std::cout << " P2 " << P2 << std::endl;


  //  cv::isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
  // Precompute maps for cvRemap()
  cv::initUndistortRectifyMap(M1, D1, R1, P1, left_img.size(),  CV_16SC2, map11, map12);
  cv::initUndistortRectifyMap(M2, D2, R2, P2, right_img.size(), CV_16SC2, map21, map22);

  cv::Mat left_img_rmp, right_img_rmp, disp, vdisp;
  cv::remap(left_img, left_img_rmp, map11, map12, cv::INTER_LINEAR);
  cv::remap(right_img, right_img_rmp, map21, map22, cv::INTER_LINEAR);


  // https://github.com/oreillymedia/Learning-OpenCV-3_examples/blob/master/example_19-03.cpp
  //  cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, 512,11,72,288);
  //  stereo->compute(left_img_rmp, right_img_rmp, disp);
  //  cv::normalize(disp, vdisp, 0, 256, cv::NORM_MINMAX, CV_8U);
  //  //cv::imshow("disparity", vdisp);
  //  left_img_rmp.convertTo(left_img_rmp, CV_8U);
  cv::imwrite(left_img_out,left_img_rmp);
  cv::imwrite(right_img_out,right_img_rmp);
  //  cv::imwrite("disp.png",vdisp);


  DICe::finalize();
  return 0;
}

