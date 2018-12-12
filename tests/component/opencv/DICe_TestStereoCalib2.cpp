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

#include <DICe.h>

#include <DICe_StereoCalib2.h>
#include <DICe_CamSystem.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include <iostream>


//using code from TestStereoCalib as a basis to start

using namespace DICe;



int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint = argc - 1;
  int_t error_flag = 0;

  StereoCalib2 calibration;
  CamSystem camera_system;
  std::vector<std::vector<std::string>> image_list;
  image_list.resize(2);
  std::string zero_pad;
  std::string arg = "";
  std::string calfile = "";
  float rms;
  bool test_passed = true;
  if (argc == 2) arg = argv[1];

  calibration.set_Verbose(true);

  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0) {
    outStream = Teuchos::rcp(&std::cout, false);
    calibration.set_Verbose(true);
  }
  else {
    outStream = Teuchos::rcp(&bhs, false);
    calibration.set_Verbose(false);
  }

  *outStream << "--- Begin test ---" << std::endl;


  std::string image_dir = "../images/";
  std::string output_dir = "../cal/";


  if (argc == 1 || arg == "ECh") { //extract and save intersection points from the checkerboard pattern
    *outStream << "extracting intersections from a checkerboard pattern" << std::endl;

    //note that images 0 and 9 are not in the directory and will not run
    //setup the image list
    image_list[0].clear();
    image_list[1].clear();
    for (int_t i_image = 0; i_image < 14; ++i_image) {
      if (i_image < 10) zero_pad = "0";
      else zero_pad = "";
      std::stringstream left_name;
      std::stringstream right_name;
      left_name << "left" << zero_pad << i_image << ".jpg";
      right_name << "right" << zero_pad << i_image << ".jpg";
      image_list[0].push_back(left_name.str());
      image_list[1].push_back(right_name.str());
    }
    //set the image list
    calibration.set_Image_List(image_list, image_dir);

    //set the target information
    int total_width = 9;
    int total_height = 6;
    double spacing = 10.0;
    int marker_space_x = 0;
    int marker_space_y = 0;
    int orig_loc_x = 0;
    int orig_loc_y = 0;
    DICe_StereoCalib2::Target_Type target = DICe_StereoCalib2::CHECKER_BOARD;
    //set the target information
    calibration.set_Target_Info(target, total_width, total_height, spacing, orig_loc_x, orig_loc_y, marker_space_x, marker_space_y);
    //draw the intersection images
    calibration.Draw_Intersection_Image(true, output_dir);
    //extract the intersections and save
    calibration.Extract_Target_Points();
    //write the intersection file
    calibration.write_intersection_file("checkerboard_intersections.xml", output_dir);
  }

  if (argc == 1 || arg == "CCh") {
    camera_system.clear_system();
    //Run the calibration
    *outStream << "reading checkerboard intersections and running calibration" << std::endl;
    //read intersection file
    calibration.read_intersection_file("checkerboard_intersections.xml", output_dir);
    //do the calibration
    rms = calibration.do_openCV_calibration(camera_system);
    //write the calibration file
    calfile = output_dir + "checkerboard_calibration.xml";
    calibration.write_calibration_file(calfile, true, camera_system);
    if (rms > 0.75) error_flag = 1;
  }


  if (argc == 1 || arg == "ED0") { //extract and save intersection points from the original dot patterns

    *outStream << "extracting intersections from a dot pattern" << std::endl;

    //setup the image list
    image_list[0].clear();
    image_list[1].clear();
    for (int_t i_image = 0; i_image < 14; ++i_image) {
      if (i_image < 10) zero_pad = "0";
      else zero_pad = "";
      std::stringstream left_name;
      std::stringstream right_name;
      left_name << "CalB-sys2-00" << zero_pad << i_image << "_0.jpeg";
      right_name << "CalB-sys2-00" << zero_pad << i_image << "_1.jpeg";
      image_list[0].push_back(left_name.str());
      image_list[1].push_back(right_name.str());
    }
    //set the image list
    calibration.set_Image_List(image_list, image_dir);

    //set the target information
    int total_width = 10;
    int total_height = 8;
    double spacing = 3.5;
    int marker_space_x = 6;
    int marker_space_y = 4;
    int orig_loc_x = 2;
    int orig_loc_y = 2;
    DICe_StereoCalib2::Target_Type target = DICe_StereoCalib2::BLACK_ON_WHITE_W_DONUT_DOTS;
    calibration.set_Target_Info(target, total_width, total_height, spacing, orig_loc_x, orig_loc_y, marker_space_x, marker_space_y);
    //draw the intersection images
    calibration.Draw_Intersection_Image(true, output_dir);
    //extract the intersections and save
    calibration.Extract_Target_Points(30, 50);
    //write the intersection file
    calibration.write_intersection_file("dot_intersections.xml", output_dir);
  }

  if (argc == 1 || arg == "CD0") {
    camera_system.clear_system();
    //Run the calibration
    calibration.read_intersection_file("dot_intersections.xml", output_dir);
    calibration.do_openCV_calibration(camera_system);
    //write the calibration file
    calfile = output_dir + "dot_calibration.xml";
    calibration.write_calibration_file(calfile, true, camera_system);
    if (rms > 0.75) error_flag = error_flag + 2;
  }


  if (arg == "ED1") { //extract and save intersection points from the DICe challenge dot images

    *outStream << "extracting intersections from DIC challenge dot pattern" << std::endl;

    //setup the image list
    image_list[0].clear();
    image_list[1].clear();
    for (int_t i_image = 0; i_image < 84; ++i_image) {
      if (i_image < 10) zero_pad = "0";
      else zero_pad = "";
      std::stringstream left_name;
      std::stringstream right_name;
      left_name << "Cal-sys1-00" << zero_pad << i_image << "_0.tif";
      right_name << "Cal-sys1-00" << zero_pad << i_image << "_1.tif";
      image_list[0].push_back(left_name.str());
      image_list[1].push_back(right_name.str());
    }
    //set the image list
    calibration.set_Image_List(image_list, image_dir);

    //set the target information
    int total_width = 14;
    int total_height = 10;
    double spacing = 10.0;
    int marker_space_x = 10;
    int marker_space_y = 6;
    int orig_loc_x = 2;
    int orig_loc_y = 2;
    DICe_StereoCalib2::Target_Type target = DICe_StereoCalib2::BLACK_ON_WHITE_W_DONUT_DOTS;
    //set the target information
    calibration.set_Target_Info(target, total_width, total_height, spacing, orig_loc_x, orig_loc_y, marker_space_x, marker_space_y);
    //draw the intersection images
    calibration.Draw_Intersection_Image(true, output_dir);
    //extract the intersections and save
    calibration.Extract_Target_Points();
    //write the intersection file
    calibration.write_intersection_file("DICe_chal_intersections.xml", output_dir);
  }

  if (arg == "CD1") {
    //Run the calibration
    calibration.read_intersection_file("DICe_chal_intersections.xml", output_dir);
    calibration.do_openCV_calibration(camera_system);
    //write the calibration file
    calfile = output_dir + "DICe_chal_calibration.xml";
    calibration.write_calibration_file(calfile, true, camera_system);
    if (rms > 0.75) test_passed = false;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (error_flag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return error_flag;

}


