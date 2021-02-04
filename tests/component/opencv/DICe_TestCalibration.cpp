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
  const work_t error_tol = 1.0E-5;
  const work_t max_rms = 0.75;
  const work_t max_cal_diff = 5.0; // loose tol due to only using a portion of the total cal image collection to save space
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  *outStream << "\n--- single camera checkerboard ---\n" << std::endl;

  // SINGLE CAM CHECKERBOARD
  try{
    DICe::Calibration cal_checkerboard("../cal/checkerboard_input.xml");
    // create the intersection points and generate a calibrated camera system
    work_t checkerboard_rms = 0.0;
    Teuchos::RCP<Camera_System> checkerboard_cam_sys = cal_checkerboard.calibrate(checkerboard_rms);
    // write the calibration parameters to file which includes the interesection points
    cal_checkerboard.write_calibration_file("checkerboard_calibration.xml");
    // create a new calibration object that reads in the parameter file above with the intersection points
    DICe::Calibration cal_checkerboard_with_intersections("checkerboard_calibration.xml");

    *outStream << "\n--- single camera checkerboard using pre-defined intersection points ---\n" << std::endl;

    // calibrate again using the intersection points read from file
    work_t test_checkerboard_rms = 0.0;
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
    if(checkerboard_rms<0||checkerboard_rms>max_rms){
      *outStream << "error, rms from checkerboard calibration does not meet tolerance" << std::endl;
      error_flag++;
    }
  }catch(std::exception & e){
    *outStream << e.what() << std::endl;
    *outStream << "error, single camera checkerboard case failed" << std::endl;
    error_flag++;
  }

  *outStream << "\n--- stereo checkerboard ---\n" << std::endl;

  // STEREO CHECKERBOARD
  try{
    DICe::Calibration cal_stereo_checkerboard("../cal/stereo_checkerboard_input.xml");
    // create the intersection points and generate a calibrated camera system
    work_t stereo_checkerboard_rms = 0.0;
    Teuchos::RCP<Camera_System> stereo_checkerboard_cam_sys = cal_stereo_checkerboard.calibrate(stereo_checkerboard_rms);
    // write the calibration parameters to file which includes the interesection points
    cal_stereo_checkerboard.write_calibration_file("stereo_checkerboard_calibration.xml");
    // create a new calibration object that reads in the parameter file above with the intersection points
    DICe::Calibration cal_stereo_checkerboard_with_intersections("stereo_checkerboard_calibration.xml");

    *outStream << "\n--- stereo checkerboard using pre-defined intersection points ---\n" << std::endl;

    // calibrate again using the intersection points read from file
    work_t test_stereo_checkerboard_rms = 0.0;
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
    if(stereo_checkerboard_rms<0||stereo_checkerboard_rms>max_rms){
      *outStream << "error, rms from stereo checkerboard calibration does not meet tolerance" << std::endl;
      error_flag++;
    }
  }catch(std::exception & e){
    *outStream << e.what() << std::endl;
    *outStream << "error, stereo checkerboard case failed" << std::endl;
    error_flag++;
  }

  *outStream << "\n--- simulated checkerboard with validation ---\n" << std::endl;

  // SIMULATED CHECKERBOARD WITH COMPARISON TO OUTPUT FROM ANOTHER CODE'S CALIBRATION
  try{
    DICe::Calibration sim_checkerboard_cal("../cal/sim_checkerboard_input.xml");
    // create the intersection points and generate a calibrated camera system
    work_t sim_checkerboard_rms = 0.0;
    Teuchos::RCP<Camera_System> sim_checkerboard_cam_sys = sim_checkerboard_cal.calibrate(sim_checkerboard_rms);
    Camera_System sim_checkerboard_gold_cam_sys("../cal/sim_checkerboard_cal_gold.xml");
    const work_t sim_diff = sim_checkerboard_cam_sys->diff(sim_checkerboard_gold_cam_sys);
    *outStream << "sim checkerboard error norm: " << sim_diff << std::endl;
    if(sim_diff>max_cal_diff){
      *outStream << "error, simulated checkerboard calibration not correct" << std::endl;
      error_flag++;
    }
  }catch(std::exception & e){
    *outStream << e.what() << std::endl;
    *outStream << "error, simulated checkerboard case failed" << std::endl;
    error_flag++;
  }

  *outStream << "\n--- stereo donut dots ---\n" << std::endl;

  // STEREO DONUT DOTS
  try{
    DICe::Calibration cal_stereo_marker_dots("../cal/stereo_marker_dots_input.xml");
    // create the intersection points and generate a calibrated camera system
    work_t stereo_marker_dots_rms = 0.0;
    Teuchos::RCP<Camera_System> stereo_marker_dots_cam_sys = cal_stereo_marker_dots.calibrate(stereo_marker_dots_rms);
    // write the calibration parameters to file which includes the interesection points
    cal_stereo_marker_dots.write_calibration_file("stereo_marker_dots_calibration.xml");
    // create a new calibration object that reads in the parameter file above with the intersection points
    DICe::Calibration cal_stereo_marker_dots_with_intersections("stereo_marker_dots_calibration.xml");
    // calibrate again using the intersection points read from file
    work_t test_stereo_marker_dots_rms = 0.0;
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
    if(stereo_marker_dots_rms<0||stereo_marker_dots_rms>0.75){
      *outStream << "error, rms from stereo marker dot calibration does not meet tolerance" << std::endl;
      error_flag++;
    }
    test_stereo_marker_dots_cam_sys->write_camera_system_file("stereo_marker_dots_camera_system.xml");
    // test creating a camera system with the file that was just output
    Teuchos::RCP<Camera_System> trial_cam_sys = Teuchos::rcp(new Camera_System("stereo_marker_dots_camera_system.xml"));
  }catch(std::exception & e){
    *outStream << e.what() << std::endl;
    *outStream << "error, stereo marker dots case failed" << std::endl;
    error_flag++;
  }

  *outStream << "\n--- simulated donut dots with validation ---\n" << std::endl;

  // SIMULATED DOT WITH COMPARISON TO OUTPUT FROM ANOTHER CODE'S CALIBRATION
  try{
    DICe::Calibration sim_dot_cal("../cal/sim_dot_input.xml");
    // create the intersection points and generate a calibrated camera system
    work_t sim_dot_rms = 0.0;
    Teuchos::RCP<Camera_System> sim_dot_cam_sys = sim_dot_cal.calibrate(sim_dot_rms);
    //std::cout << *sim_dot_cam_sys.get() << std::endl;
    Camera_System sim_dot_gold_cam_sys("../cal/sim_dot_cal_gold.xml");
    const work_t sim_dot_diff = sim_dot_cam_sys->diff(sim_dot_gold_cam_sys);
    *outStream << "sim dot error norm: " << sim_dot_diff << std::endl;
    if(sim_dot_diff>max_cal_diff){
      *outStream << "error, simulated dot calibration not correct" << std::endl;
      error_flag++;
    }
  }catch(std::exception & e){
    *outStream << e.what() << std::endl;
    *outStream << "error, simulated dot case failed" << std::endl;
    error_flag++;
  }

  *outStream << "\n--- simulated donut dots white on black background ---\n" << std::endl;

  // SIMULATED SINGLE CAMERA WHITE DOT ON BLACK BOARD WITH COMPARISON TO OUTPUT FROM ANOTHER CODE'S CALIBRATION
  try{
    DICe::Calibration sim_white_dot_cal("../cal/sim_white_dot_input.xml");
    // create the intersection points and generate a calibrated camera system
    work_t sim_white_dot_rms = 0.0;
    Teuchos::RCP<Camera_System> sim_white_dot_cam_sys = sim_white_dot_cal.calibrate(sim_white_dot_rms);
    sim_white_dot_cam_sys->write_camera_system_file("sim_white_dot_cal.xml");
    std::cout << *sim_white_dot_cam_sys.get() << std::endl;
    Camera_System sim_white_dot_gold_cam_sys("../cal/sim_white_dot_cal_gold.xml"); // same gold file as black on white example
    const work_t sim_white_dot_diff = sim_white_dot_cam_sys->diff(sim_white_dot_gold_cam_sys);
    *outStream << "sim white dot error norm: " << sim_white_dot_diff << std::endl;
    if(sim_white_dot_diff>max_cal_diff){
      *outStream << "error, simulated white dot calibration not correct" << std::endl;
      error_flag++;
    }
  }catch(std::exception & e){
    *outStream << e.what() << std::endl;
    *outStream << "error, simulated white dot case failed" << std::endl;
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


