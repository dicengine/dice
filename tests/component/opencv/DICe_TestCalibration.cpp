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

#include <DICe_Calibration.h>
#include <DICe_CameraSystem.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint = argc - 1;
  int_t error_flag = 0;
  const scalar_t error_tol = 1.0E-5;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // SINGLE CAM CHECKERBOARD
  try{
    DICe::Calibration cal_checkerboard("../cal/checkerboard_input.xml");
    // create the intersection points and generate a calibrated camera system
    scalar_t checkerboard_rms = 0.0;
    Teuchos::RCP<Camera_System> checkerboard_cam_sys = cal_checkerboard.calibrate(checkerboard_rms);
    // write the calibration parameters to file which includes the interesection points
    cal_checkerboard.write_calibration_file("checkerboard_calibration.xml");
    // create a new calibration object that reads in the parameter file above with the intersection points
    DICe::Calibration cal_checkerboard_with_intersections("checkerboard_calibration.xml");
    // calibrate again using the intersection points read from file
    scalar_t test_checkerboard_rms = 0.0;
    Teuchos::RCP<Camera_System> test_checkerboard_cam_sys = cal_checkerboard_with_intersections.calibrate(test_checkerboard_rms);
    // now compare the two camera systems that were generated to make sure they are identical:
    if(*checkerboard_cam_sys.get()!=*test_checkerboard_cam_sys.get()){
      *outStream << "error, checkerboard camera systems do not match" << std::endl;
      error_flag++;
    }
    if(std::abs(checkerboard_rms-test_checkerboard_rms)>error_tol){
      *outStream << "error, rms from checkerboard calibrations do not match" << std::endl;
      error_flag++;
    }
  }catch(std::exception & e){
    *outStream << e.what() << std::endl;
    *outStream << "error, single camera checkerboard case failed" << std::endl;
    error_flag++;
  }

  // STEREO CHECKERBOARD
  try{
    DICe::Calibration cal_stereo_checkerboard("../cal/stereo_checkerboard_input.xml");
    // create the intersection points and generate a calibrated camera system
    scalar_t stereo_checkerboard_rms = 0.0;
    Teuchos::RCP<Camera_System> stereo_checkerboard_cam_sys = cal_stereo_checkerboard.calibrate(stereo_checkerboard_rms);
    // write the calibration parameters to file which includes the interesection points
    cal_stereo_checkerboard.write_calibration_file("stereo_checkerboard_calibration.xml");
    // create a new calibration object that reads in the parameter file above with the intersection points
    DICe::Calibration cal_stereo_checkerboard_with_intersections("stereo_checkerboard_calibration.xml");
    // calibrate again using the intersection points read from file
    scalar_t test_stereo_checkerboard_rms = 0.0;
    Teuchos::RCP<Camera_System> test_stereo_checkerboard_cam_sys = cal_stereo_checkerboard_with_intersections.calibrate(test_stereo_checkerboard_rms);
    // now compare the two camera systems that were generated to make sure they are identical:
    if(*stereo_checkerboard_cam_sys.get()!=*test_stereo_checkerboard_cam_sys.get()){
      *outStream << "error, stereo_checkerboard camera systems do not match" << std::endl;
      error_flag++;
    }
    if(std::abs(stereo_checkerboard_rms-test_stereo_checkerboard_rms)>error_tol){
      *outStream << "error, rms from stereo_checkerboard calibrations do not match" << std::endl;
      error_flag++;
    }
  }catch(std::exception & e){
    *outStream << e.what() << std::endl;
    *outStream << "error, stereo checkerboard case failed" << std::endl;
    error_flag++;
  }

  // STEREO DONUT DOTS
  try{
    DICe::Calibration cal_stereo_marker_dots("../cal/stereo_marker_dots_input.xml");
    // create the intersection points and generate a calibrated camera system
    scalar_t stereo_marker_dots_rms = 0.0;
    Teuchos::RCP<Camera_System> stereo_marker_dots_cam_sys = cal_stereo_marker_dots.calibrate(stereo_marker_dots_rms);
    // write the calibration parameters to file which includes the interesection points
    cal_stereo_marker_dots.write_calibration_file("stereo_marker_dots_calibration.xml");
    // create a new calibration object that reads in the parameter file above with the intersection points
    DICe::Calibration cal_stereo_marker_dots_with_intersections("stereo_marker_dots_calibration.xml");
    // calibrate again using the intersection points read from file
    scalar_t test_stereo_marker_dots_rms = 0.0;
    Teuchos::RCP<Camera_System> test_stereo_marker_dots_cam_sys = cal_stereo_marker_dots_with_intersections.calibrate(test_stereo_marker_dots_rms);
    // now compare the two camera systems that were generated to make sure they are identical:
    if(*stereo_marker_dots_cam_sys.get()!=*test_stereo_marker_dots_cam_sys.get()){
      *outStream << "error, stereo_marker_dots camera systems do not match" << std::endl;
      error_flag++;
    }
    if(std::abs(stereo_marker_dots_rms-test_stereo_marker_dots_rms)>error_tol){
      *outStream << "error, rms from stereo_marker_dots calibrations do not match" << std::endl;
      error_flag++;
    }
  }catch(std::exception & e){
    *outStream << e.what() << std::endl;
    *outStream << "error, stereo marker dots case failed" << std::endl;
    error_flag++;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (error_flag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return error_flag;

}


