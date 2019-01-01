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

#include <DICe_CameraSystem.h>
#include <DICe_Parser.h>
#include <DICe_XMLUtils.h>

#include <fstream>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

namespace DICe{

void
Camera_System::clear_system() {
  //clear the cameras
  for (int_t i = 0; i < MAX_NUM_CAMERAS_PER_SYSTEM; i++)
    cameras_[i].clear_camera();

  //clear the 4x4 parameter matrix
  user_trans_4x4_params_.clear();
  for (int_t i = 0; i < 4; ++i) {
    user_trans_4x4_params_.push_back(std::vector<scalar_t>(4, 0.0));
    user_trans_4x4_params_[i][i] = 1.0;
  }
  has_4x4_transform_ = false;

  rotation_3x3_params_.clear();
  for (int_t i = 0; i < 3; ++i) {
    rotation_3x3_params_.push_back(std::vector<scalar_t>(3, 0.0));
  }
  has_opencv_rot_trans_ = false;

  //clear the 6 parameter transform
  user_trans_6_params_.assign(6, 0.0);
  has_6_transform_ = false;

  //reset the system type and number of cameras
  sys_type_ = UNKNOWN_SYSTEM;
  num_cams_ = 0;
  valid_cal_file_ = false;
  cal_file_error_ = std::stringstream();
}

//*********************************load_calibration_parameters*********************************************
void
Camera_System::read_calibration_parameters(const std::string & cal_file) {
  DEBUG_MSG(" ");
  DEBUG_MSG("*****************************load_calibration_parameters**************************");
  DEBUG_MSG("Camera_System::load_calibration_parameters() was called");

  //clear the cameras and initialize required values
  clear_system();

  const std::string xml("xml");
  const std::string txt("txt");

  bool valid_xml = true;
  bool param_found = false;
  valid_cal_file_ = true;

  std::string msg = "";
  std::string param_text = "";
  std::stringstream param_title;
  std::stringstream param_val;

  DEBUG_MSG("Camera_System::load_calibration_parameters():Trying to read file with Teuchos XML parser: " << cal_file);
  Teuchos::RCP<Teuchos::ParameterList> sys_parms = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::Ptr<Teuchos::ParameterList> sys_parms_ptr(sys_parms.get());
  try {
    Teuchos::updateParametersFromXmlFile(cal_file, sys_parms_ptr);
  }
  catch (std::exception & e) {
    DEBUG_MSG("Camera_System::load_calibration_parameters():Invalid XML file: " << cal_file);
    DEBUG_MSG("Camera_System::load_calibration_parameters():" << e.what());
    valid_xml = false;
  }

  if (valid_xml) {  //file read into parameters
    DEBUG_MSG("Camera_System::load_calibration_parameters():Valid XML file for Teuchos parser");
    param_found = sys_parms->isParameter(cal_file_ID);
    if (param_found)	param_text = sys_parms->get<std::string>(cal_file_ID);
    if (param_text != DICe_XML_Calibration_File || !param_found) {
      DEBUG_MSG("Camera_System::load_calibration_parameters():XML calibration ID file not valid");
      valid_cal_file_ = false;
      cal_file_error_ << "DICe xml calibration file does not have proper ID field: cal_field_ID: DICe_XML_Calibration_File";
    }

    else {
      DEBUG_MSG("Camera_System::load_calibration_parameters(): " << cal_file_ID << " = " << param_text);
      if (sys_parms->isParameter(system_type_3D)) {
        param_text = sys_parms->get<std::string>(system_type_3D);
        DEBUG_MSG("Camera_System::load_calibration_parameters(): " << system_type_3D << " = " << param_text);
        sys_type_ = string_to_system_type_3d(param_text);

        //cycle through all the cameras to see if they are assigned
        for (int_t camera_index = 0; camera_index < 10; camera_index++) {
          param_title = std::stringstream();
          param_title << "CAMERA " << camera_index;
          //camera found
          if (sys_parms->isSublist(param_title.str())) {
            //increment the number of cameras
            num_cams_++;
            //create vectors to hold the camera intrinsic and extrinsic parameters
            std::vector<scalar_t> intrinsics(Camera::MAX_CAM_INTRINSIC_PARAM, 0);
            std::vector<scalar_t> extrinsics(Camera::MAX_CAM_EXTRINSIC_PARAM, 0);
            DEBUG_MSG("Camera_System::load_calibration_parameters(): reading " << param_title.str());
            //access the sublist of camera parameters
            Teuchos::ParameterList camParams = sys_parms->sublist(param_title.str());
            //fill the array with any intrinsic parameters
            for (int_t i = 0; i < Camera::LENS_DISTORTION_MODEL; i++) {
              if (camParams.isParameter(Camera::to_string(static_cast<Camera::Cam_Intrinsic_Param>(i)))) {
                intrinsics[i] = camParams.get<std::double_t>(Camera::to_string(static_cast<Camera::Cam_Intrinsic_Param>(i)));
                DEBUG_MSG("Camera_System::load_calibration_parameters(): reading " << Camera::to_string(static_cast<Camera::Cam_Intrinsic_Param>(i)));
              }
            }
            //the lens distorion model is handled here
            if (camParams.isParameter(Camera::to_string(Camera::LENS_DISTORTION_MODEL))) {
              param_text = camParams.get<std::string>(Camera::to_string(Camera::LENS_DISTORTION_MODEL));
              DEBUG_MSG("Camera_System::load_calibration_parameters(): reading " << Camera::to_string(Camera::LENS_DISTORTION_MODEL));
              intrinsics[Camera::LENS_DISTORTION_MODEL] = Camera::string_to_lens_distortion_model(param_text);
            }
            //fill the array with any extrinsic parameters
            for (int_t i = 0; i < Camera::MAX_CAM_EXTRINSIC_PARAM; i++) {
              if (camParams.isParameter(Camera::to_string(static_cast<Camera::Cam_Extrinsic_Param>(i)))) {
                extrinsics[i] = camParams.get<std::double_t>(Camera::to_string(static_cast<Camera::Cam_Extrinsic_Param>(i)));
                DEBUG_MSG("Camera_System::load_calibration_parameters(): reading " << Camera::to_string(static_cast<Camera::Cam_Extrinsic_Param>(i)));
              }
            }
            //Save the parameters to the cameras
            cameras_[camera_index].set_intrinsics(intrinsics);
            cameras_[camera_index].set_extrinsics(extrinsics);

            //Get the camera ID if it exists
            param_title = std::stringstream();
            param_title << "CAMERA_ID";
            param_val = std::stringstream();
            if (camParams.isParameter(param_title.str())) {
              param_val << camParams.get<std::string>(param_title.str());
              DEBUG_MSG("Camera_System::load_calibration_parameters(): reading CAMERA_ID ");
            }
            else {
              //If not use the camera and number by default
              param_val << "CAMERA " << camera_index;
              DEBUG_MSG("Camera_System::load_calibration_parameters(): CAMERA_ID not found using camera #");
            }
            cameras_[camera_index].set_identifier(param_val.str());

            //Get the image height and width if they exists
            param_title = std::stringstream();
            param_title << "IMAGE_HEIGHT_WIDTH";
            if (camParams.isParameter(param_title.str())) {
              DEBUG_MSG("Camera_System::load_calibration_parameters(): reading IMAGE_HEIGHT_WIDTH");
              param_text = camParams.get<std::string>(param_title.str());
              Teuchos::Array<int_t>  tArray;
              tArray = Teuchos::fromStringToArray<int_t>(param_text);
              cameras_[camera_index].set_image_height(tArray[0]);
              cameras_[camera_index].set_image_width(tArray[1]);
            }

            //Get the pixel depth if it exists
            param_title = std::stringstream();
            param_title << "PIXEL_DEPTH";
            if (camParams.isParameter(param_title.str())) {
              DEBUG_MSG("Camera_System::load_calibration_parameters(): reading PIXEL_DEPTH");
              int_t pixel_depth = camParams.get<int_t>(param_title.str());
              cameras_[camera_index].set_pixel_depth(pixel_depth);
            }

            //get the lens comment if it exists
            param_title = std::stringstream();
            param_title << "LENS";
            if (camParams.isParameter(param_title.str())) {
              DEBUG_MSG("Camera_System::load_calibration_parameters(): reading LENS");
              param_text = camParams.get<std::string>(param_title.str());
              cameras_[camera_index].set_camera_lens(param_text);
            }

            //get the general comments if they exists
            param_title = std::stringstream();
            param_title << "COMMENTS";
            if (camParams.isParameter(param_title.str())) {
              DEBUG_MSG("Camera_System::load_calibration_parameters(): reading COMMENTS");
              param_text = camParams.get<std::string>(param_title.str());
              cameras_[camera_index].set_camera_comments(param_text);
            }

            //does the camera have a 3x3 rotation transformation matrix?
            if (camParams.isSublist(rotation_3x3_matrix)) {
              DEBUG_MSG("Camera_System::load_calibration_parameters(): reading " << rotation_3x3_matrix);
              Teuchos::ParameterList camRot = camParams.sublist(rotation_3x3_matrix);
              Teuchos::Array<scalar_t>  tArray;
              for (int_t j = 0; j < 3; j++) {
                param_title = std::stringstream();
                param_title << "LINE " << j;
                param_text = camParams.get<std::string>(param_title.str());
                tArray = Teuchos::fromStringToArray<scalar_t>(param_text);
                for (int_t i = 0; i < 3; i++)
                  rotation_3x3_params_[j][i] = tArray[i];
              }
              cameras_[camera_index].set_3x3_rotation_matrix(rotation_3x3_params_);
            }

            if (!cameras_[camera_index].camera_valid(msg)) {
              DEBUG_MSG("Camera_System::load_calibration_parameters(): camera " << camera_index << " is invalid");
              DEBUG_MSG(msg);
              valid_cal_file_ = false;
              cal_file_error_ << "Camera_System::load_calibration_parameters(): camera " << camera_index << " is invalid" << "\n";
              cal_file_error_ << msg;
            }

          }
        }
        //does the file have a 6 parameter transform?
        if (sys_parms->isParameter(user_6_param_transform)) {
          DEBUG_MSG("Camera_System::load_calibration_parameters(): reading " << user_6_param_transform);
          has_6_transform_ = true;
          param_text = sys_parms->get<std::string>(user_6_param_transform);
          Teuchos::Array<scalar_t>  tArray;
          tArray = Teuchos::fromStringToArray<scalar_t>(param_text);
          for (int_t i = 0; i < 6; i++)
            user_trans_6_params_[i] = tArray[i];
        }
        //does the file have a 4x4 parameter transform?
        if (sys_parms->isSublist(user_4x4_param_transform)) {
          DEBUG_MSG("Camera_System::load_calibration_parameters(): reading " << user_4x4_param_transform);
          has_4x4_transform_ = true;
          Teuchos::ParameterList camParams = sys_parms->sublist(user_4x4_param_transform);
          Teuchos::Array<scalar_t>  tArray;
          for (int_t j = 0; j < 4; j++) {
            param_title = std::stringstream();
            param_title << "LINE " << j;
            param_text = camParams.get<std::string>(param_title.str());
            tArray = Teuchos::fromStringToArray<scalar_t>(param_text);
            for (int_t i = 0; i < 4; i++)
              user_trans_4x4_params_[j][i] = tArray[i];
          }
        }
      }
    }

  }
  else {

    DEBUG_MSG("Camera_System::load_calibration_parameters(): Parsing calibration parameters from non Teuchos file: " << cal_file);
    std::fstream dataFile(cal_file.c_str(), std::ios_base::in);
    if (!dataFile.good()) {
      TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(), std::runtime_error,
        "Error, the calibration file does not exist or is corrupt: " << cal_file);
      valid_cal_file_ = false;
      cal_file_error_ << "Error, the calibration file does not exist or is corrupt: " << cal_file;
    }

    //check the file extension for xml or txt
    if (cal_file.find(xml) != std::string::npos) {
      DEBUG_MSG("Camera_System::load_calibration_parameters(): assuming calibration file is vic3D xml format");
      // cal.xml file can't be ready by Teuchos parser because it has a !DOCTYPE
      // have to manually read the file here, lots of assumptions in how the file is formatted
      // camera orientation for each camera in vic3d is in terms of the world to camera
      // orientation and the order of variables is alpha beta gamma tx ty tz (the Cardan Bryant angles + translations)
      // read each line of the file
      sys_type_ = VIC3D;
      std::vector<int_t> param_order_int = { Camera::CX, Camera::CY, Camera::FX, Camera::FY, Camera::FS, Camera::K1, Camera::K2, Camera::K3 };
      std::vector<int_t> param_order_ext = { Camera::ALPHA, Camera::BETA, Camera::GAMMA, Camera::TX, Camera::TY, Camera::TZ };

      while (!dataFile.eof()) {
        std::vector<std::string> tokens = tokenize_line(dataFile, " \t<>\"");
        if (tokens.size() == 0) continue;
        if (tokens[0] != "CAMERA") continue;
        num_cams_++;
        const int_t camera_index = std::atoi(tokens[2].c_str());
        param_title.clear();
        param_title << "CAMERA " << camera_index;
        cameras_[camera_index].set_identifier(param_title.str());
        //create the intrinsic and extrinsic arrays
        std::vector<scalar_t> intrinsics(Camera::MAX_CAM_INTRINSIC_PARAM, 0);
        std::vector<scalar_t> extrinsics(Camera::MAX_CAM_EXTRINSIC_PARAM, 0);

        DEBUG_MSG("Camera_System::load_calibration_parameters(): reading camera " << camera_index << " parameters");
        assert(camera_index < 10);
        assert(tokens.size() > 18);
        //Store the intrinsic parameters
        for (int_t i = 0; i < (int_t)param_order_int.size(); ++i)
          intrinsics[param_order_int[i]] = strtod(tokens[i + 3].c_str(), NULL);
        intrinsics[Camera::LENS_DISTORTION_MODEL] = Camera::K1R1_K2R2_K3R3;

        //Store the extrinsic parameters
        assert(tokens[11] == "ORIENTATION");
        for (int_t i = 0; i < (int_t)param_order_ext.size(); ++i)
          extrinsics[param_order_ext[i]] = strtod(tokens[i + 12].c_str(), NULL);

        cameras_[camera_index].set_intrinsics(intrinsics);
        cameras_[camera_index].set_extrinsics(extrinsics);
        if (!cameras_[camera_index].camera_valid(msg)) {
          DEBUG_MSG("Camera_System::load_calibration_parameters(): camera " << camera_index << " is invalid");
          DEBUG_MSG(msg);
          valid_cal_file_ = false;
          cal_file_error_ << "Camera_System::load_calibration_parameters(): camera " << camera_index << " is invalid" << "\n";
          cal_file_error_ << msg;
        }

      } // end file read
    }

    else if (cal_file.find(txt) != std::string::npos) {
      DEBUG_MSG("Camera_System::load_calibration_parameters(): calibration file is generic txt format");
      //may want to modify this file format to allow more than 2 cameras in the future
      sys_type_ = GENERIC_SYSTEM;
      num_cams_ = 2;
      std::vector<int_t> param_order_int = { Camera::CX, Camera::CY, Camera::FX, Camera::FY, Camera::FS, Camera::K1, Camera::K2, Camera::K3 };
      std::vector<int_t> param_order_ext = { Camera::ALPHA, Camera::BETA, Camera::GAMMA, Camera::TX, Camera::TY, Camera::TZ };
      const int_t num_values_base = 22;
      const int_t num_values_expected_with_R = 28;
      bool has_extrinsic = false;
      //const int_t num_values_with_custom_transform = 28;
      int_t total_num_values = 0;
      //create the intrinsic and extrinsic arrays
      std::vector<scalar_t> intrinsics_c0(Camera::MAX_CAM_INTRINSIC_PARAM, 0);
      std::vector<scalar_t> extrinsics_c0(Camera::MAX_CAM_EXTRINSIC_PARAM, 0);
      std::vector<scalar_t> intrinsics_c1(Camera::MAX_CAM_INTRINSIC_PARAM, 0);
      std::vector<scalar_t> extrinsics_c1(Camera::MAX_CAM_EXTRINSIC_PARAM, 0);

      //Run through the file to determine the format
      while (!dataFile.eof())
      {
        std::vector<std::string>tokens = tokenize_line(dataFile, " \t<>");
        if (tokens.size() == 0) continue;
        if (tokens[0] == "#") continue;
        if (tokens[0] == "TRANSFORM") {
          has_6_transform_ = true; //file has a 6 parm user transformation
          has_extrinsic = true; //file has extrinsic values
          continue;
        }
        total_num_values++;
      }

      if (total_num_values == num_values_base) {
        has_extrinsic = true; //file has extrinsic values
      }
      has_4x4_transform_ = !has_extrinsic;

      TEUCHOS_TEST_FOR_EXCEPTION(total_num_values != num_values_base && total_num_values != num_values_expected_with_R, std::runtime_error,
        "Error, wrong number of parameters in calibration file: " << cal_file);

      // return to start of file:
      dataFile.clear();
      dataFile.seekg(0, std::ios::beg);

      //text file has cam0 aligned with the world coordinates so all extrinsics are 0 for cam 0
      for (int_t i = 0; i < Camera::MAX_CAM_EXTRINSIC_PARAM; ++i)
        extrinsics_c0[i] = 0.0;
      intrinsics_c0[Camera::LENS_DISTORTION_MODEL] = Camera::K1R1_K2R2_K3R3;
      intrinsics_c1[Camera::LENS_DISTORTION_MODEL] = Camera::K1R1_K2R2_K3R3;

      // use default camera ID
      cameras_[0].set_identifier("CAMERA 0");
      cameras_[1].set_identifier("CAMERA 1");

      int_t num_values = 0;
      int_t i = 0;
      int_t j = 0;
      while (!dataFile.eof())
      {
        std::vector<std::string> tokens = tokenize_line(dataFile, " \t<>");
        if (tokens.size() == 0) continue;
        if (tokens[0] == "#") continue;
        if (tokens[0] == "TRANSFORM") continue;
        if (tokens.size() > 1) {
          assert(tokens[1] == "#"); // only one entry per line plus comments
        }
        if (num_values < 8)
          intrinsics_c0[param_order_int[num_values]] = strtod(tokens[0].c_str(), NULL);
        else if (num_values < 16)
          intrinsics_c1[param_order_int[num_values - 8]] = strtod(tokens[0].c_str(), NULL);
        else if (num_values < 22 && has_extrinsic)
          extrinsics_c1[param_order_int[num_values - 16]] = strtod(tokens[0].c_str(), NULL);
        else if (num_values < 25 && !has_extrinsic) {
          i = int((num_values - 16) / 3);
          j = (num_values - 16) % 3;
          user_trans_4x4_params_[i][j] = strtod(tokens[0].c_str(), NULL);
        }
        else if (num_values < 28 && !has_extrinsic)
          user_trans_4x4_params_[num_values - 25][3] = strtod(tokens[0].c_str(), NULL);
        else if (num_values < 28 && has_6_transform_)
          user_trans_6_params_[num_values - 22] = strtod(tokens[0].c_str(), NULL);
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "");
        }
        num_values++;
      }
      cameras_[0].set_intrinsics(intrinsics_c0);
      cameras_[0].set_extrinsics(extrinsics_c0);
      cameras_[1].set_intrinsics(intrinsics_c1);
      cameras_[1].set_extrinsics(extrinsics_c1);
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "Error, unrecognized calibration parameters file format: " << cal_file);
    }
  }

  if (num_cams_ != 0) {
    DEBUG_MSG("************************************************************************");
    DEBUG_MSG("System type: " << to_string(sys_type_));
    DEBUG_MSG("Number of Cams: " << num_cams_);
    DEBUG_MSG(" ");

    for (int_t i = 0; i < 10; i++) {
      if (cameras_[i].camera_filled()) {
        std::vector<scalar_t> intrinsics(Camera::MAX_CAM_INTRINSIC_PARAM, 0);
        std::vector<scalar_t> extrinsics(Camera::MAX_CAM_EXTRINSIC_PARAM, 0);
        cameras_[i].get_intrinsics(intrinsics);
        cameras_[i].get_extrinsics(extrinsics);

        DEBUG_MSG("******************* CAMERA: " << i << " ******************************");
        DEBUG_MSG("Camera_System::load_calibration_parameters(): identifier: " << cameras_[i].get_identifier());
        for (int_t j = 0; j < Camera::LENS_DISTORTION_MODEL; j++)
          DEBUG_MSG("Camera_System::load_calibration_parameters(): " << Camera::to_string(static_cast<Camera::Cam_Intrinsic_Param>(j)) << ": " << intrinsics[j]);
        int_t j = Camera::LENS_DISTORTION_MODEL;
        DEBUG_MSG("Camera_System::load_calibration_parameters(): " << Camera::to_string(static_cast<Camera::Cam_Intrinsic_Param>(j)) << ": " << Camera::to_string(static_cast<Camera::Lens_Distortion_Model>((int)intrinsics[j])));
        for (int_t j = 0; j < Camera::MAX_CAM_EXTRINSIC_PARAM; j++)
          DEBUG_MSG("Camera_System::load_calibration_parameters(): " << Camera::to_string(static_cast<Camera::Cam_Extrinsic_Param>(j)) << ": " << extrinsics[j]);
        DEBUG_MSG("Camera_System::load_calibration_parameters(): image height: " << cameras_[i].get_image_height());
        DEBUG_MSG("Camera_System::load_calibration_parameters(): image width: " << cameras_[i].get_image_width());
        DEBUG_MSG("Camera_System::load_calibration_parameters(): pixel depth: " << cameras_[i].get_pixel_depth());
        DEBUG_MSG("Camera_System::load_calibration_parameters(): lens: " << cameras_[i].get_camera_lens());
        DEBUG_MSG("Camera_System::load_calibration_parameters(): comments: " << cameras_[i].get_camera_comments());
        DEBUG_MSG(" ");
      }
    }
  }
  // 4x4 independent transformation
  if (has_4x4_transform_) {
    DEBUG_MSG("Camera_System::load_calibration_parameters(): 4x4 user transformation");
    for (int_t i = 0; i < 4; ++i)
      DEBUG_MSG("Camera_System::load_calibration_parameters(): " << user_trans_4x4_params_[i][0] <<
        " " << user_trans_4x4_params_[i][1] << " " << user_trans_4x4_params_[i][2] << " " << user_trans_4x4_params_[i][3]);
  }
  // 6 param independent transformation
  if (has_6_transform_) {
    DEBUG_MSG("Camera_System::load_calibration_parameters(): 6 parameter user transformation");
    DEBUG_MSG("Camera_System::load_calibration_parameters(): " << user_trans_6_params_[0] <<
      " " << user_trans_6_params_[1] << " " << user_trans_6_params_[2] << " " << user_trans_6_params_[3] <<
      " " << user_trans_6_params_[4] << " " << user_trans_6_params_[5]);
  }
  DEBUG_MSG("Camera_System::load_calibration_parameters(): end");
}

//*********************************save_calibration_file*********************************************
void
Camera_System::write_calibration_file(const std::string & cal_file,
  bool all_fields) {
  std::stringstream param_title;
  std::stringstream param_val;
  std::stringstream valid_fields;
  DEBUG_MSG(" ");
  DEBUG_MSG("*****************************save_calibration_parameters**************************");
  DEBUG_MSG("Camera_System::save_calibration_file() was called.");
  if (sys_type_ == UNKNOWN_SYSTEM) {
    DEBUG_MSG("Camera_System::save_calibration_file(): the system type has not been set file write canceled");
    return;
  }
  DEBUG_MSG("Calibration output file: " << cal_file);
  //do we want to check for .xml extension?
  // clear the files if they exist
  initialize_xml_file(cal_file);

  //write the header
  DEBUG_MSG("Camera_System::save_calibration_file(): writing headder");
  if (all_fields) DEBUG_MSG("Camera_System::save_calibration_file(): writing all parameters and comments");
  write_xml_comment(cal_file, "DICe formatted calibration file");

  if (all_fields) write_xml_comment(cal_file, "cal_file_ID must be in the file with a value of DICe_XML_Calibration_File");
  write_xml_string_param(cal_file, cal_file_ID, DICe_XML_Calibration_File, false);

  if (all_fields) {
    valid_fields=std::stringstream();
    valid_fields << "type of 3D system valid field values are: ";
    for (int_t n = 1; n < MAX_SYSTEM_TYPE_3D; n++) valid_fields << " " << to_string(static_cast<System_Type_3D>(n));
    write_xml_comment(cal_file, valid_fields.str());
  }
  write_xml_string_param(cal_file, system_type_3D, to_string(sys_type_), false);

  //camera parameters
  write_xml_comment(cal_file, "camera parameters (zero valued parameters don't need to be specified)");
  if (all_fields) {
    write_xml_comment(cal_file, "the file supports up to 16 cameras 0-15 each camera is a seperate sublist of parameters");
    write_xml_comment(cal_file, "the sublist must be named CAMERA {#} with {#} the number of the camera starting at 0");
    valid_fields = std::stringstream();
    valid_fields << "valid camera parameter field names are: ";
    for (int_t n = 0; n < Camera::MAX_CAM_INTRINSIC_PARAM; n++) valid_fields << " " << Camera::to_string(static_cast<Camera::Cam_Intrinsic_Param>(n));
    for (int_t n = 0; n < Camera::MAX_CAM_EXTRINSIC_PARAM; n++) valid_fields << " " << Camera::to_string(static_cast<Camera::Cam_Extrinsic_Param>(n));
    write_xml_comment(cal_file, valid_fields.str());

    write_xml_comment(cal_file, "CX,CY-image center (pix), FX,FY-pin hole distances (pix), FS-skew (deg)");
    write_xml_comment(cal_file, "K1-K6-lens distortion coefficients, P1-P2-tangential distortion(openCV), S1-S4 thin prism distortion(openCV), T1,T2-Scheimpfug correction (openCV)");
    write_xml_comment(cal_file, "be aware that openCV gives the values in the following order: (K1,K2,P1,P2[,K3[,K4,K5,K6[,S1,S2,S3,S4[,TX,TY]]]])");

    valid_fields = std::stringstream();
    valid_fields << "valid values for the lens distortion model are: ";
    for (int_t n = 0; n < Camera::MAX_LENS_DISTORTION_MODEL; n++) valid_fields << " " << Camera::to_string(static_cast<Camera::Lens_Distortion_Model>(n));
    write_xml_comment(cal_file, valid_fields.str());
    write_xml_comment(cal_file, "NONE no distortion model");
    write_xml_comment(cal_file, "OPENCV_DIS uses the model defined in openCV 3.4.1");
    write_xml_comment(cal_file, "VIC3D_DIS uses the model defined for VIC3D");
    write_xml_comment(cal_file, "K1R1_K2R2_K3R3 -> K1*R + K2*R^2 + K3*R^3");
    write_xml_comment(cal_file, "K1R2_K2R4_K3R6 -> K1*R^2 + K2*R^4 + K3*R^6");
    write_xml_comment(cal_file, "K1R3_K2R5_K3R7 -> K1*R^3 + K2*R^5 + K3*R^7");
    write_xml_comment(cal_file, "each camera may also have a sublist with the values for the 3x3 rotation matrix");
    write_xml_comment(cal_file, "if no matrix is given the rotation matrix is generated from the alph, beta, gamma values or given an identity matrix");

    write_xml_comment(cal_file, "additional camera fields:");
    write_xml_comment(cal_file, "CAMERA_ID: unique camera descripter, if not supplied CAMERA {#} is used");
    write_xml_comment(cal_file, "IMAGE_HEIGHT_WIDTH {h, w}");
    write_xml_comment(cal_file, "PIXEL_DEPTH");
    write_xml_comment(cal_file, "LENS");
    write_xml_comment(cal_file, "COMMENTS");
    write_xml_comment(cal_file, "any parameter with a value of 0 may simply be omitted from the calibration file");
  }

  for (int_t camera_index = 0; camera_index < MAX_NUM_CAMERAS_PER_SYSTEM; camera_index++) {
    //does the camera have any parameters
    if (cameras_[camera_index].camera_filled() || (camera_index==0 && all_fields)) {

      param_title = std::stringstream();
      param_title << "CAMERA " << camera_index;
      DEBUG_MSG("Camera_System::save_calibration_file(): writing camera parameters:" << param_title.str());
      write_xml_param_list_open(cal_file, param_title.str(), false);

      param_title = std::stringstream();
      param_title << "CAMERA_ID";
      param_val = std::stringstream();
      param_val << cameras_[camera_index].get_identifier();
      write_xml_string_param(cal_file, param_title.str(), param_val.str(), false);

      std::vector<scalar_t> intrinsics(Camera::MAX_CAM_INTRINSIC_PARAM, 0);
      cameras_[camera_index].get_intrinsics(intrinsics);
      for (int_t j = 0; j < Camera::MAX_CAM_INTRINSIC_PARAM; j++) {
        if (intrinsics[j] != 0||all_fields) {
          param_val = std::stringstream();
          if (j != Camera::LENS_DISTORTION_MODEL) {
            param_val << intrinsics[j];
            write_xml_real_param(cal_file, Camera::to_string(static_cast<Camera::Cam_Intrinsic_Param>(j)), param_val.str(), false);
          }
          else {
            param_val << Camera::to_string(static_cast<Camera::Lens_Distortion_Model>((int)intrinsics[j]));
            write_xml_string_param(cal_file, Camera::to_string(static_cast<Camera::Cam_Intrinsic_Param>(j)), param_val.str(), false);
          }
        }
      }

      std::vector<scalar_t> extrinsics(Camera::MAX_CAM_EXTRINSIC_PARAM, 0);
      cameras_[camera_index].get_extrinsics(extrinsics);
      for (int_t j = 0; j < Camera::MAX_CAM_EXTRINSIC_PARAM; j++) {
        if (extrinsics[j] != 0 || all_fields) {
          param_val = std::stringstream();
          param_val << extrinsics[j];
          write_xml_real_param(cal_file, Camera::to_string(static_cast<Camera::Cam_Extrinsic_Param>(j)), param_val.str(), false);
        }
      }

      if (cameras_[camera_index].camera_has_3x3_rotation() || all_fields) {
        std::vector<std::vector<scalar_t> > rot_mat;
        rot_mat.resize(3);
        rot_mat[0].assign(3, 0.0);
        rot_mat[1].assign(3, 0.0);
        rot_mat[2].assign(3, 0.0);
        cameras_[camera_index].get_3x3_rotation_matrix(rot_mat);
        DEBUG_MSG("Camera_System::save_calibration_file(): writing 3x3 rotation matrix");
        write_xml_comment(cal_file, "3x3 camera rotation matrix (world to cam transformation)");
        if (all_fields) write_xml_comment(cal_file, "this is a 3x3 matrix that combined with TX, TY and TZ transform world coodinates to this camera's coordinates");
        write_xml_param_list_open(cal_file, rotation_3x3_matrix, false);
        for (int_t i = 0; i < 3; i++) {
          param_val = std::stringstream();
          param_title = std::stringstream();
          param_title << "LINE " << i;
          param_val << "{ ";
          for (int_t j = 0; j < 2; j++)
            param_val << rot_mat[i][j] << ", ";
          param_val << rot_mat[i][2] << " }";
          write_xml_string_param(cal_file, param_title.str(), param_val.str(), false);
          if (all_fields && i == 0) write_xml_comment(cal_file, "R11 R12 R13");
          if (all_fields && i == 1) write_xml_comment(cal_file, "R21 R22 R23");
          if (all_fields && i == 2) write_xml_comment(cal_file, "R31 R32 R33");
        }
        write_xml_param_list_close(cal_file, false);
      }

      int_t img_width;
      int_t img_height;
      img_width = cameras_[camera_index].get_image_width();
      img_height = cameras_[camera_index].get_image_height();
      param_title = std::stringstream();
      param_val = std::stringstream();
      if ((img_width != 0 && img_height != 0) || all_fields) {
        param_title << "IMAGE_HEIGHT_WIDTH";
        param_val << "{ " << img_height << ", " << img_width << " }";
        write_xml_string_param(cal_file, param_title.str(), param_val.str(), false);
      }

      int_t pixel_depth;
      pixel_depth = cameras_[camera_index].get_pixel_depth();
      param_title = std::stringstream();
      param_val = std::stringstream();
      if (pixel_depth != 0 || all_fields) {
        param_title << "PIXEL_DEPTH";
        param_val << pixel_depth;
        write_xml_size_param(cal_file, param_title.str(), param_val.str(), false);
      }

      param_title = std::stringstream();
      param_val = std::stringstream();
      param_val << cameras_[camera_index].get_camera_lens();
      if (param_val.str() != "" || all_fields) {
        param_title << "LENS";
        write_xml_string_param(cal_file, param_title.str(), param_val.str(), false);
      }

      param_title = std::stringstream();
      param_val = std::stringstream();
      param_val << cameras_[camera_index].get_camera_comments();
      if (param_val.str() != "" || all_fields) {
        param_title << "COMMENTS";
        write_xml_string_param(cal_file, param_title.str(), param_val.str(), false);
      }

      write_xml_param_list_close(cal_file, false);
    }
  }

  if (has_6_transform_ || all_fields) {
    DEBUG_MSG("Camera_System::save_calibration_file(): writing user supplied 6 parameter transform");
    write_xml_comment(cal_file, "user supplied 6 parameter transform");
    if (all_fields) write_xml_comment(cal_file, "this is a user supplied transform with 6 parameters seperate from the transforms determined by the camera parameters");
    if (all_fields) write_xml_comment(cal_file, "can used for non-projection transformations between images - optional");
    param_val = std::stringstream();
    param_val << "{ ";
    for (int_t j = 0; j < 5; j++)
      param_val << user_trans_6_params_[j] << ", ";
    param_val << user_trans_6_params_[5] << " }";
    write_xml_string_param(cal_file, user_6_param_transform, param_val.str(), false);
  }

  if (has_4x4_transform_ || all_fields) {
    DEBUG_MSG("Camera_System::save_calibration_file(): writing user supplied 4x4 parameter transform");
    write_xml_comment(cal_file, "user supplied 4x4 parameter transform");
    if (all_fields) write_xml_comment(cal_file, "this is a user supplied 4x4 array transform seperate from the transforms determined by the camera parameters");
    if (all_fields) write_xml_comment(cal_file, "typically includes a combined rotation and translation array  - optional");
    write_xml_param_list_open(cal_file, user_4x4_param_transform, false);
    for (int_t i = 0; i < 4; i++) {
      param_val = std::stringstream();
      param_title = std::stringstream();
      param_title << "LINE " << i;
      param_val << "{ ";
      for (int_t j = 0; j < 3; j++)
        param_val << user_trans_4x4_params_[i][j] << ", ";
      param_val << user_trans_4x4_params_[i][3] << " }";
      write_xml_string_param(cal_file, param_title.str(), param_val.str(), false);
      if (all_fields && i == 0) write_xml_comment(cal_file, "R11 R12 R13 TX");
      if (all_fields && i == 1) write_xml_comment(cal_file, "R21 R22 R23 TY");
      if (all_fields && i == 2) write_xml_comment(cal_file, "R31 R32 R33 TZ");
      if (all_fields && i == 3) write_xml_comment(cal_file, "0   0   0   1");
    }
    write_xml_param_list_close(cal_file, false);
  }

  if (has_opencv_rot_trans_ || all_fields) {
    DEBUG_MSG("Camera_System::save_calibration_file(): writing openCV rotation matrix");
    write_xml_comment(cal_file, "openCV rotation translation matrix from from cam0 to cam 1");
    if (all_fields) write_xml_comment(cal_file, "this is the [R|t] rotation translation matrix from openCV");
    if (all_fields) write_xml_comment(cal_file, "this is the basic output from openCV stereo calibration for R and t");
    write_xml_param_list_open(cal_file, rotation_3x3_matrix, false);
    for (int_t i = 0; i < 3; i++) {
      param_val = std::stringstream();
      param_title = std::stringstream();
      param_title << "LINE " << i;
      param_val << "{ ";
      for (int_t j = 0; j < 3; j++)
        param_val << rotation_3x3_params_[i][j] << ", ";
      param_val << rotation_3x3_params_[i][3] << " }";
      write_xml_string_param(cal_file, param_title.str(), param_val.str(), false);
      if (all_fields && i == 0) write_xml_comment(cal_file, "R11 R12 R13 TX");
      if (all_fields && i == 1) write_xml_comment(cal_file, "R21 R22 R23 TY");
      if (all_fields && i == 2) write_xml_comment(cal_file, "R31 R32 R33 TZ");
    }
    write_xml_param_list_close(cal_file, false);
  }
  finalize_xml_file(cal_file);
}

void
Camera_System::pre_projection_(int_t num_pnts,
  int_t num_params,
  bool partials) {
  //prep the cameras if not already done
  if (!cameras_[source_cam_].camera_prepped())cameras_[source_cam_].prep_camera();
  if (!cameras_[target_cam_].camera_prepped())cameras_[target_cam_].prep_camera();

  //resize the vectors if necessary
  if ((int_t)img_x_.size() != num_pnts) {
    img_x_.assign(num_pnts, 0.0);
    img_y_.assign(num_pnts, 0.0);
    sen_x_.assign(num_pnts, 0.0);
    sen_y_.assign(num_pnts, 0.0);
    cam_x_.assign(num_pnts, 0.0);
    cam_y_.assign(num_pnts, 0.0);
    cam_z_.assign(num_pnts, 0.0);
    wld0_x_.assign(num_pnts, 0.0);
    wld0_y_.assign(num_pnts, 0.0);
    wld0_z_.assign(num_pnts, 0.0);
    wld1_x_.assign(num_pnts, 0.0);
    wld1_y_.assign(num_pnts, 0.0);
    wld1_z_.assign(num_pnts, 0.0);
  }
  if (partials && (int_t)img_dx_[0].size() != num_pnts){
    img_dx_.clear();
    img_dy_.clear();
    sen_dx_.clear();
    sen_dy_.clear();
    cam_dx_.clear();
    cam_dy_.clear();
    cam_dz_.clear();
    wld0_dx_.clear();
    wld0_dy_.clear();
    wld0_dz_.clear();
    wld1_dx_.clear();
    wld1_dy_.clear();
    wld1_dz_.clear();
    for (int_t j = 0; j < num_params; j++) {
      img_dx_[j].assign(num_pnts, 0.0);
      img_dy_[j].assign(num_pnts, 0.0);
      sen_dx_[j].assign(num_pnts, 0.0);
      sen_dy_[j].assign(num_pnts, 0.0);
      cam_dx_[j].assign(num_pnts, 0.0);
      cam_dy_[j].assign(num_pnts, 0.0);
      cam_dz_[j].assign(num_pnts, 0.0);
      wld0_dx_[j].assign(num_pnts, 0.0);
      wld0_dy_[j].assign(num_pnts, 0.0);
      wld0_dz_[j].assign(num_pnts, 0.0);
      wld1_dx_[j].assign(num_pnts, 0.0);
      wld1_dy_[j].assign(num_pnts, 0.0);
      wld1_dz_[j].assign(num_pnts, 0.0);
    }
  }
}

void
Camera_System::cross_projection_map(
  const scalar_t & img0_x,
  const scalar_t & img0_y,
  scalar_t & img1_x,
  scalar_t & img1_y,
  std::vector<scalar_t> & params) {

  pre_projection_(1, 3, false);

  img_x_[0] = img0_x;
  img_y_[0] = img0_y;
  cameras_[source_cam_].image_to_sensor(img_x_, img_y_, sen_x_, sen_y_);
  cameras_[source_cam_].sensor_to_cam(sen_x_, sen_y_, cam_x_, cam_y_, cam_z_, params);
  cameras_[source_cam_].cam_to_world(cam_x_, cam_y_, cam_z_, wld0_x_, wld0_y_, wld0_z_);
  cameras_[target_cam_].world_to_cam(wld0_x_, wld0_y_, wld0_z_, cam_x_, cam_y_, cam_z_);
  cameras_[target_cam_].cam_to_sensor(cam_x_, cam_y_, cam_z_, sen_x_, sen_y_);
  cameras_[target_cam_].sensor_to_image(sen_x_, sen_y_, img_x_, img_y_);
  img1_x = img_x_[0];
  img1_y = img_y_[0];
}

void
Camera_System::cross_projection_map(
  const scalar_t & img0_x,
  const scalar_t & img0_y,
  scalar_t & img1_x,
  scalar_t & img1_y,
  std::vector<scalar_t> & params,
  std::vector<scalar_t> & img1_dx,
  std::vector<scalar_t> & img1_dy) {

  pre_projection_(1, 3, true);
  img_x_[0] = img0_x;
  img_y_[0] = img0_y;

  cameras_[source_cam_].image_to_sensor(img_x_, img_y_, sen_x_, sen_y_);
  cameras_[source_cam_].sensor_to_cam(sen_x_, sen_y_, cam_x_, cam_y_, cam_z_, params, cam_dx_, cam_dy_, cam_dz_);
  cameras_[source_cam_].cam_to_world(cam_x_, cam_y_, cam_z_, wld0_x_, wld0_y_, wld0_z_, cam_dx_, cam_dy_, cam_dz_, wld0_dx_, wld0_dy_, wld0_dz_);
  cameras_[target_cam_].world_to_cam(wld0_x_, wld0_y_, wld0_z_, cam_x_, cam_y_, cam_z_, wld0_dx_, wld0_dy_, wld0_dz_, cam_dx_, cam_dy_, cam_dz_);
  cameras_[target_cam_].cam_to_sensor(cam_x_, cam_y_, cam_z_, sen_x_, sen_y_, cam_dx_, cam_dy_, cam_dz_, sen_dx_, sen_dy_);
  cameras_[target_cam_].sensor_to_image(sen_x_, sen_y_, img_x_, img_y_, sen_dx_, sen_dy_, img_dx_, img_dy_);
  img1_x = img_x_[0];
  img1_y = img_y_[0];
  for (int_t j = 0; j < 3; j++) {
    img1_dx[j] = img_dx_[j][0];
    img1_dy[j] = img_dy_[j][0];
  }
}

void
Camera_System::cross_projection_map(
  std::vector<scalar_t> & img0_x,
  std::vector<scalar_t> & img0_y,
  std::vector<scalar_t> & img1_x,
  std::vector<scalar_t> & img1_y,
  std::vector<scalar_t> & params) {

  pre_projection_(img0_x.size(), 3, false);

  cameras_[source_cam_].image_to_sensor(img0_x, img0_y, sen_x_, sen_y_);
  cameras_[source_cam_].sensor_to_cam(sen_x_, sen_y_, cam_x_, cam_y_, cam_z_, params);
  cameras_[source_cam_].cam_to_world(cam_x_, cam_y_, cam_z_, wld0_x_, wld0_y_, wld0_z_);
  cameras_[target_cam_].world_to_cam(wld0_x_, wld0_y_, wld0_z_, cam_x_, cam_y_, cam_z_);
  cameras_[target_cam_].cam_to_sensor(cam_x_, cam_y_, cam_z_, sen_x_, sen_y_);
  cameras_[target_cam_].sensor_to_image(sen_x_, sen_y_, img1_x, img1_y);
}

void
Camera_System::cross_projection_map(
  std::vector<scalar_t> & img0_x,
  std::vector<scalar_t> & img0_y,
  std::vector<scalar_t> & img1_x,
  std::vector<scalar_t> & img1_y,
  std::vector<scalar_t> & params,
  std::vector<std::vector<scalar_t> > & img1_dx,
  std::vector<std::vector<scalar_t> > & img1_dy) {

  pre_projection_(img0_x.size(), 3, true);

  cameras_[source_cam_].image_to_sensor(img0_x, img0_y, sen_x_, sen_y_);
  cameras_[source_cam_].sensor_to_cam(sen_x_, sen_y_, cam_x_, cam_y_, cam_z_, params, cam_dx_, cam_dy_, cam_dx_);
  cameras_[source_cam_].cam_to_world(cam_x_, cam_y_, cam_z_, wld0_x_, wld0_y_, wld0_z_, cam_dx_, cam_dy_, cam_dx_, wld0_dx_, wld0_dy_, wld0_dz_);
  cameras_[target_cam_].world_to_cam(wld0_x_, wld0_y_, wld0_z_, cam_x_, cam_y_, cam_z_, wld0_dx_, wld0_dy_, wld0_dz_, cam_dx_, cam_dy_, cam_dx_);
  cameras_[target_cam_].cam_to_sensor(cam_x_, cam_y_, cam_z_, sen_x_, sen_y_, cam_dx_, cam_dy_, cam_dx_, sen_dx_, sen_dy_);
  cameras_[target_cam_].sensor_to_image(sen_x_, sen_y_, img1_x, img1_y, sen_dx_, sen_dy_, img1_dx, img1_dy);
}

//*****Fixed projection parametes free 3D rigid body motion (single camera 3D)
void
Camera_System::fixed_proj_3DRB_map(
  const scalar_t & img0_x,
  const scalar_t & img0_y,
  scalar_t & img1_x,
  scalar_t & img1_y,
  std::vector<scalar_t> & proj_params,
  std::vector<scalar_t> & rigid_body_params) {

  pre_projection_(1, 6, false);
  img_x_[0] = img0_x;
  img_y_[0] = img0_y;

  cameras_[source_cam_].image_to_sensor(img_x_, img_y_, sen_x_, sen_y_);
  cameras_[source_cam_].sensor_to_cam(sen_x_, sen_y_, cam_x_, cam_y_, cam_z_, proj_params);
  cameras_[source_cam_].cam_to_world(cam_x_, cam_y_, cam_z_, wld0_x_, wld0_y_, wld0_z_);
  rot_trans_3D(wld0_x_, wld0_y_, wld0_z_, wld1_x_, wld1_y_, wld1_z_, rigid_body_params);
  cameras_[target_cam_].world_to_cam(wld1_x_, wld1_y_, wld1_z_, cam_x_, cam_y_, cam_z_);
  cameras_[target_cam_].cam_to_sensor(cam_x_, cam_y_, cam_z_, sen_x_, sen_y_);
  cameras_[target_cam_].sensor_to_image(sen_x_, sen_y_, img_x_, img_y_);
  img1_x = img_x_[0];
  img1_y = img_y_[0];
}

void
Camera_System::fixed_proj_3DRB_map(
  const scalar_t & img0_x,
  const scalar_t & img0_y,
  scalar_t & img1_x,
  scalar_t & img1_y,
  std::vector <scalar_t> & proj_params,
  std::vector <scalar_t> & rigid_body_params,
  std::vector<scalar_t> & img1_dx,
  std::vector<scalar_t> & img1_dy) {

  pre_projection_(1, 6, false);
  img_x_[0] = img0_x;
  img_y_[0] = img0_y;

  cameras_[source_cam_].image_to_sensor(img_x_, img_y_, sen_x_, sen_y_);
  cameras_[source_cam_].sensor_to_cam(sen_x_, sen_y_, cam_x_, cam_y_, cam_z_, proj_params);
  cameras_[source_cam_].cam_to_world(cam_x_, cam_y_, cam_z_, wld0_x_, wld0_y_, wld0_z_);
  rot_trans_3D(wld0_x_, wld0_y_, wld0_z_, wld1_x_, wld1_y_, wld1_z_, rigid_body_params, wld1_dx_, wld1_dy_, wld1_dz_);
  cameras_[target_cam_].world_to_cam(wld1_x_, wld1_y_, wld1_z_, cam_x_, cam_y_, cam_z_, wld1_dx_, wld1_dy_, wld1_dz_, cam_dx_, cam_dy_, cam_dx_);
  cameras_[target_cam_].cam_to_sensor(cam_x_, cam_y_, cam_z_, sen_x_, sen_y_, cam_dx_, cam_dy_, cam_dx_, sen_dx_, sen_dy_);
  cameras_[target_cam_].sensor_to_image(sen_x_, sen_y_, img_x_, img_y_, sen_dx_, sen_dy_, img_dx_, img_dy_);

  img1_x = img_x_[0];
  img1_y = img_y_[0];
  for (int_t j = 0; j < 6; j++) {
    img1_dx[j] = img_dx_[j][0];
    img1_dy[j] = img_dy_[j][0];
  }
}

void
Camera_System::fixed_proj_3DRB_map(
  std::vector<scalar_t> & img0_x,
  std::vector<scalar_t> & img0_y,
  std::vector<scalar_t> & img1_x,
  std::vector<scalar_t> & img1_y,
  std::vector<scalar_t> & proj_params,
  std::vector<scalar_t> & rigid_body_params) {

  pre_projection_(img0_x.size(), 6, false);

  cameras_[source_cam_].image_to_sensor(img0_x, img0_y, sen_x_, sen_y_);
  cameras_[source_cam_].sensor_to_cam(sen_x_, sen_y_, cam_x_, cam_y_, cam_z_, proj_params);
  cameras_[source_cam_].cam_to_world(cam_x_, cam_y_, cam_z_, wld0_x_, wld0_y_, wld0_z_);
  rot_trans_3D(wld0_x_, wld0_y_, wld0_z_, wld1_x_, wld1_y_, wld1_z_, rigid_body_params);
  cameras_[target_cam_].world_to_cam(wld1_x_, wld1_y_, wld1_z_, cam_x_, cam_y_, cam_z_);
  cameras_[target_cam_].cam_to_sensor(cam_x_, cam_y_, cam_z_, sen_x_, sen_y_);
  cameras_[target_cam_].sensor_to_image(sen_x_, sen_y_, img1_x, img1_y);
}

void
Camera_System::fixed_proj_3DRB_map(
  std::vector<scalar_t> & img0_x,
  std::vector<scalar_t> & img0_y,
  std::vector<scalar_t> & img1_x,
  std::vector<scalar_t> & img1_y,
  std::vector<scalar_t> & proj_params,
  std::vector<scalar_t> & rigid_body_params,
  std::vector<std::vector<scalar_t> > & img1_dx,
  std::vector<std::vector<scalar_t> > & img1_dy) {

  pre_projection_(img0_x.size(), 6, false);

  cameras_[source_cam_].image_to_sensor(img0_x, img0_y, sen_x_, sen_y_);
  cameras_[source_cam_].sensor_to_cam(sen_x_, sen_y_, cam_x_, cam_y_, cam_z_, proj_params);
  cameras_[source_cam_].cam_to_world(cam_x_, cam_y_, cam_z_, wld0_x_, wld0_y_, wld0_z_);
  rot_trans_3D(wld0_x_, wld0_y_, wld0_z_, wld1_x_, wld1_y_, wld1_z_, rigid_body_params, wld1_dx_, wld1_dy_, wld1_dz_);
  cameras_[target_cam_].world_to_cam(wld1_x_, wld1_y_, wld1_z_, cam_x_, cam_y_, cam_z_, wld1_dx_, wld1_dy_, wld1_dz_, cam_dx_, cam_dy_, cam_dx_);
  cameras_[target_cam_].cam_to_sensor(cam_x_, cam_y_, cam_z_, sen_x_, sen_y_, cam_dx_, cam_dy_, cam_dx_, sen_dx_, sen_dy_);
  cameras_[target_cam_].sensor_to_image(sen_x_, sen_y_, img1_x, img1_y, sen_dx_, sen_dy_, img1_dx, img1_dy);
}

void
Camera_System::pre_rot_trans_3D(std::vector<scalar_t> params,
  bool partials) {
  //if this is the first call to pre_rot_trans_3D assign the vectors
  if (rot_trans_3D_x_.size() != 4) {
    rot_trans_3D_x_.assign(4, 0.0);
    rot_trans_3D_y_.assign(4, 0.0);
    rot_trans_3D_z_.assign(4, 0.0);
    for (int_t i = 0; i < 6; i++) {
      rot_trans_3D_dx_[i].assign(4, 0.0);
      rot_trans_3D_dy_[i].assign(4, 0.0);
      rot_trans_3D_dz_[i].assign(4, 0.0);
    }
  }

  scalar_t cx, cy, cz, sx, sy, sz, tx, ty, tz;
  cx = cos(params[ANGLE_X]);
  cy = cos(params[ANGLE_Y]);
  cz = cos(params[ANGLE_Z]);
  sx = sin(params[ANGLE_X]);
  sy = sin(params[ANGLE_Y]);
  sz = sin(params[ANGLE_Z]);
  tx = params[TRANSLATION_X];
  ty = params[TRANSLATION_Y];
  tz = params[TRANSLATION_Z];

  rot_trans_3D_x_[0] = cy * cz;
  rot_trans_3D_x_[1] = sx * sy * cz - cx * sz;
  rot_trans_3D_x_[2] = cx * sy * cz + sx * sz;
  rot_trans_3D_x_[3] = tx;
  rot_trans_3D_y_[0] = cy * sz;
  rot_trans_3D_y_[1] = sx * sy * sz + cx * cz;
  rot_trans_3D_y_[2] = cx * sy * sz - sx * cz;
  rot_trans_3D_y_[3] = ty;
  rot_trans_3D_z_[0] = -sy;
  rot_trans_3D_z_[1] = sx * cy;
  rot_trans_3D_z_[2] = cx * cy;
  rot_trans_3D_z_[3] = tz;

  if (partials) {
    rot_trans_3D_dx_[ANGLE_X][0] = 0;
    rot_trans_3D_dx_[ANGLE_X][1] = cx * sy * cz + sx * sz;
    rot_trans_3D_dx_[ANGLE_X][2] = -sx * sy * cz + cx * sz;
    rot_trans_3D_dx_[ANGLE_X][3] = 0;
    rot_trans_3D_dy_[ANGLE_X][0] = 0;
    rot_trans_3D_dy_[ANGLE_X][1] = cx * sy * sz - sx * cz;
    rot_trans_3D_dy_[ANGLE_X][2] = - sx * sy * sz - cx * cz;
    rot_trans_3D_dy_[ANGLE_X][3] = 0;
    rot_trans_3D_dz_[ANGLE_X][0] = 0;
    rot_trans_3D_dz_[ANGLE_X][1] = cx * cy;
    rot_trans_3D_dz_[ANGLE_X][2] = -sx * cy;
    rot_trans_3D_dz_[ANGLE_X][3] = 0;

    rot_trans_3D_dx_[ANGLE_Y][0] = -sy * cz;
    rot_trans_3D_dx_[ANGLE_Y][1] = sx * cy * cz;
    rot_trans_3D_dx_[ANGLE_Y][2] = cx * cy * cz;
    rot_trans_3D_dx_[ANGLE_Y][3] = 0;
    rot_trans_3D_dy_[ANGLE_Y][0] = -sy * sz;
    rot_trans_3D_dy_[ANGLE_Y][1] = sx * cy * sz;
    rot_trans_3D_dy_[ANGLE_Y][2] = cx * cy * sz;
    rot_trans_3D_dy_[ANGLE_Y][3] = 0;
    rot_trans_3D_dz_[ANGLE_Y][0] = -cy;
    rot_trans_3D_dz_[ANGLE_Y][1] = -sx * sy;
    rot_trans_3D_dz_[ANGLE_Y][2] = -cx * sy;
    rot_trans_3D_dz_[ANGLE_Y][3] = 0;

    rot_trans_3D_dx_[ANGLE_Z][0] = -cy * sz;
    rot_trans_3D_dx_[ANGLE_Z][1] = -sx * sy * sz - cx * cz;
    rot_trans_3D_dx_[ANGLE_Z][2] = -cx * sy * sz + sx * cz;
    rot_trans_3D_dx_[ANGLE_Z][3] = 0;
    rot_trans_3D_dy_[ANGLE_Z][0] = cy * cz;
    rot_trans_3D_dy_[ANGLE_Z][1] = sx * sy * cz - cx * sz;
    rot_trans_3D_dy_[ANGLE_Z][2] = cx * sy * cz + sx * sz;
    rot_trans_3D_dy_[ANGLE_Z][3] = 0;
    rot_trans_3D_dz_[ANGLE_Z][0] = 0;
    rot_trans_3D_dz_[ANGLE_Z][1] = 0;
    rot_trans_3D_dz_[ANGLE_Z][2] = 0;
    rot_trans_3D_dz_[ANGLE_Z][3] = 0;

  }
}

void
Camera_System::rot_trans_3D(std::vector<scalar_t> & wld0_x,
  std::vector<scalar_t> & wld0_y,
  std::vector<scalar_t> & wld0_z,
  std::vector<scalar_t> & wld1_x,
  std::vector<scalar_t> & wld1_y,
  std::vector<scalar_t> & wld1_z,
  std::vector<scalar_t> & params) {
  //prep the rotation coefficients
  pre_rot_trans_3D(params, false);
  //transform the coordinates
  for (int_t i = 0; i < (int_t)wld0_x.size(); i++) {
    wld1_x[i] = rot_trans_3D_x_[0] * wld0_x[i] + rot_trans_3D_x_[1] * wld0_y[i] + rot_trans_3D_x_[2] * wld0_z[i] + rot_trans_3D_x_[3];
    wld1_y[i] = rot_trans_3D_y_[0] * wld0_x[i] + rot_trans_3D_y_[1] * wld0_y[i] + rot_trans_3D_y_[2] * wld0_z[i] + rot_trans_3D_y_[3];
    wld1_z[i] = rot_trans_3D_z_[0] * wld0_x[i] + rot_trans_3D_z_[1] * wld0_y[i] + rot_trans_3D_z_[2] * wld0_z[i] + rot_trans_3D_z_[3];
  }
}

void
Camera_System::rot_trans_3D(scalar_t & wld0_x,
  scalar_t & wld0_y,
  scalar_t & wld0_z,
  scalar_t & wld1_x,
  scalar_t & wld1_y,
  scalar_t & wld1_z,
  std::vector<scalar_t> & params) {
  //prep the rotation coefficients
  pre_rot_trans_3D(params, false);
  //transform the coordinates
  wld1_x = rot_trans_3D_x_[0] * wld0_x + rot_trans_3D_x_[1] * wld0_y + rot_trans_3D_x_[2] * wld0_z + rot_trans_3D_x_[3];
  wld1_y = rot_trans_3D_y_[0] * wld0_x + rot_trans_3D_y_[1] * wld0_y + rot_trans_3D_y_[2] * wld0_z + rot_trans_3D_y_[3];
  wld1_z = rot_trans_3D_z_[0] * wld0_x + rot_trans_3D_z_[1] * wld0_y + rot_trans_3D_z_[2] * wld0_z + rot_trans_3D_z_[3];
}


void
Camera_System::rot_trans_3D(std::vector<scalar_t> & wld0_x,
  std::vector<scalar_t> & wld0_y,
  std::vector<scalar_t> & wld0_z,
  std::vector<scalar_t> & wld1_x,
  std::vector<scalar_t> & wld1_y,
  std::vector<scalar_t> & wld1_z,
  std::vector<scalar_t> & params,
  std::vector < std::vector<scalar_t> > & wld1_dx,
  std::vector < std::vector<scalar_t> > & wld1_dy,
  std::vector < std::vector<scalar_t> > & wld1_dz) {
  //This transformation assumes all partials coming into the function are 0
  //prep the rotation coefficients
  pre_rot_trans_3D(params, true);
  //transform the coordinates
  for (int_t i = 0; i < (int_t)wld0_x.size(); i++) {
    wld1_x[i] = rot_trans_3D_x_[0] * wld0_x[i] + rot_trans_3D_x_[1] * wld0_y[i] + rot_trans_3D_x_[2] * wld0_z[i] + rot_trans_3D_x_[3];
    wld1_y[i] = rot_trans_3D_y_[0] * wld0_x[i] + rot_trans_3D_y_[1] * wld0_y[i] + rot_trans_3D_y_[2] * wld0_z[i] + rot_trans_3D_y_[3];
    wld1_z[i] = rot_trans_3D_z_[0] * wld0_x[i] + rot_trans_3D_z_[1] * wld0_y[i] + rot_trans_3D_z_[2] * wld0_z[i] + rot_trans_3D_z_[3];
  }
  //calculate the partials
  for (int_t j = 0; j < 3; j++) {
    for (int_t i = 0; i < (int_t)wld0_x.size(); i++) {
      wld1_dx[j][i] = rot_trans_3D_dx_[j][0] * wld0_x[i] + rot_trans_3D_dx_[j][1] * wld0_y[i] + rot_trans_3D_dx_[j][2] * wld0_z[i];
      wld1_dy[j][i] = rot_trans_3D_dy_[j][0] * wld0_x[i] + rot_trans_3D_dy_[j][1] * wld0_y[i] + rot_trans_3D_dy_[j][2] * wld0_z[i];
      wld1_dz[j][i] = rot_trans_3D_dz_[j][0] * wld0_x[i] + rot_trans_3D_dz_[j][1] * wld0_y[i] + rot_trans_3D_dz_[j][2] * wld0_z[i];
    }
    wld1_dx[j + 3].assign(wld0_x.size(), 0.0);
    wld1_dy[j + 3].assign(wld0_x.size(), 0.0);
    wld1_dz[j + 3].assign(wld0_x.size(), 0.0);
  }
  wld1_dx[3].assign(wld0_x.size(), 1.0);
  wld1_dy[4].assign(wld0_x.size(), 1.0);
  wld1_dz[5].assign(wld0_x.size(), 1.0);
}

void
Camera_System::rot_trans_3D(scalar_t & wld0_x,
  scalar_t & wld0_y,
  scalar_t & wld0_z,
  scalar_t & wld1_x,
  scalar_t & wld1_y,
  scalar_t & wld1_z,
  std::vector<scalar_t> & params,
  std::vector<scalar_t> & wld1_dx,
  std::vector<scalar_t> & wld1_dy,
  std::vector<scalar_t> & wld1_dz) {
  //This transformation assumes all partials coming into the function are 0
  //prep the rotation coefficients
  pre_rot_trans_3D(params, true);
  //transform the coordinates
  wld1_x = rot_trans_3D_x_[0] * wld0_x + rot_trans_3D_x_[1] * wld0_y + rot_trans_3D_x_[2] * wld0_z + rot_trans_3D_x_[3];
  wld1_y = rot_trans_3D_y_[0] * wld0_x + rot_trans_3D_y_[1] * wld0_y + rot_trans_3D_y_[2] * wld0_z + rot_trans_3D_y_[3];
  wld1_z = rot_trans_3D_z_[0] * wld0_x + rot_trans_3D_z_[1] * wld0_y + rot_trans_3D_z_[2] * wld0_z + rot_trans_3D_z_[3];
  //calculate the partials
  for (int_t j = 0; j < 3; j++) {
    wld1_dx[j] = rot_trans_3D_dx_[j][0] * wld0_x + rot_trans_3D_dx_[j][1] * wld0_y + rot_trans_3D_dx_[j][2] * wld0_z;
    wld1_dy[j] = rot_trans_3D_dy_[j][0] * wld0_x + rot_trans_3D_dy_[j][1] * wld0_y + rot_trans_3D_dy_[j][2] * wld0_z;
    wld1_dz[j] = rot_trans_3D_dz_[j][0] * wld0_x + rot_trans_3D_dz_[j][1] * wld0_y + rot_trans_3D_dz_[j][2] * wld0_z;
    wld1_dx[j + 3] = 0.0;
    wld1_dy[j + 3] = 0.0;
    wld1_dz[j + 3] = 0.0;
  }
  wld1_dx[3] = 1.0;
  wld1_dy[4] = 1.0;
  wld1_dz[5] = 1.0;
}

void
Camera_System::rot_trans_3D(std::vector<scalar_t> & wld0_x,
  std::vector<scalar_t> & wld0_y,
  std::vector<scalar_t> & wld0_z,
  std::vector<scalar_t> & wld1_x,
  std::vector<scalar_t> & wld1_y,
  std::vector<scalar_t> & wld1_z,
  std::vector<scalar_t> & params,
  std::vector < std::vector<scalar_t> > & wld0_dx,
  std::vector < std::vector<scalar_t> > & wld0_dy,
  std::vector < std::vector<scalar_t> > & wld0_dz,
  std::vector < std::vector<scalar_t> > & wld1_dx,
  std::vector < std::vector<scalar_t> > & wld1_dy,
  std::vector < std::vector<scalar_t> > & wld1_dz) {
  //This transformation assumes non zero incoming partials order is Zp, Theata, Phi, Ang X, Ang Y, Ang Z
  //prep the rotation coefficients
  pre_rot_trans_3D(params, true);
  //transform the coordinates
  for (int_t i = 0; i < (int_t)wld0_x.size(); i++) {
    wld1_x[i] = rot_trans_3D_x_[0] * wld0_x[i] + rot_trans_3D_x_[1] * wld0_y[i] + rot_trans_3D_x_[2] * wld0_z[i] + rot_trans_3D_x_[3];
    wld1_y[i] = rot_trans_3D_y_[0] * wld0_x[i] + rot_trans_3D_y_[1] * wld0_y[i] + rot_trans_3D_y_[2] * wld0_z[i] + rot_trans_3D_y_[3];
    wld1_z[i] = rot_trans_3D_z_[0] * wld0_x[i] + rot_trans_3D_z_[1] * wld0_y[i] + rot_trans_3D_z_[2] * wld0_z[i] + rot_trans_3D_z_[3];
  }
  //calculate the partials
  for (int_t j = 0; j < 3; j++) {
    for (int_t i = 0; i < (int_t)wld0_x.size(); i++) {
      wld1_dx[j][i] = rot_trans_3D_x_[0] * wld0_dx[j][i] + rot_trans_3D_x_[1] * wld0_dy[j][i] + rot_trans_3D_x_[2] * wld0_dz[j][i];
      wld1_dy[j][i] = rot_trans_3D_y_[0] * wld0_dx[j][i] + rot_trans_3D_y_[1] * wld0_dy[j][i] + rot_trans_3D_y_[2] * wld0_dz[j][i];
      wld1_dz[j][i] = rot_trans_3D_z_[0] * wld0_dx[j][i] + rot_trans_3D_z_[1] * wld0_dy[j][i] + rot_trans_3D_z_[2] * wld0_dz[j][i];
      wld1_dx[j+3][i] = rot_trans_3D_dx_[j][0] * wld0_x[i] + rot_trans_3D_dx_[j][1] * wld0_y[i] + rot_trans_3D_dx_[j][2] * wld0_z[i];
      wld1_dy[j+3][i] = rot_trans_3D_dy_[j][0] * wld0_x[i] + rot_trans_3D_dy_[j][1] * wld0_y[i] + rot_trans_3D_dy_[j][2] * wld0_z[i];
      wld1_dz[j+3][i] = rot_trans_3D_dz_[j][0] * wld0_x[i] + rot_trans_3D_dz_[j][1] * wld0_y[i] + rot_trans_3D_dz_[j][2] * wld0_z[i];
    }
    wld1_dx[j + 6].assign(wld0_x.size(), 0.0);
    wld1_dy[j + 6].assign(wld0_x.size(), 0.0);
    wld1_dz[j + 6].assign(wld0_x.size(), 0.0);
  }
  wld1_dx[6].assign(wld0_x.size(), 1.0);
  wld1_dy[7].assign(wld0_x.size(), 1.0);
  wld1_dz[8].assign(wld0_x.size(), 1.0);
}

void
Camera_System::rot_trans_3D(scalar_t & wld0_x,
  scalar_t & wld0_y,
  scalar_t & wld0_z,
  scalar_t & wld1_x,
  scalar_t & wld1_y,
  scalar_t & wld1_z,
  std::vector<scalar_t> & params,
  std::vector<scalar_t> & wld0_dx,
  std::vector<scalar_t> & wld0_dy,
  std::vector<scalar_t> & wld0_dz,
  std::vector<scalar_t> & wld1_dx,
  std::vector<scalar_t> & wld1_dy,
  std::vector<scalar_t> & wld1_dz) {
  //This transformation assumes all partials coming into the function are 0
  //prep the rotation coefficients
  pre_rot_trans_3D(params, true);
  //transform the coordinates
  wld1_x = rot_trans_3D_x_[0] * wld0_x + rot_trans_3D_x_[1] * wld0_y + rot_trans_3D_x_[2] * wld0_z + rot_trans_3D_x_[3];
  wld1_y = rot_trans_3D_y_[0] * wld0_x + rot_trans_3D_y_[1] * wld0_y + rot_trans_3D_y_[2] * wld0_z + rot_trans_3D_y_[3];
  wld1_z = rot_trans_3D_z_[0] * wld0_x + rot_trans_3D_z_[1] * wld0_y + rot_trans_3D_z_[2] * wld0_z + rot_trans_3D_z_[3];
  //calculate the partials
  for (int_t j = 0; j < 3; j++) {
    wld1_dx[j] = rot_trans_3D_x_[0] * wld0_dx[j] + rot_trans_3D_x_[1] * wld0_dy[j] + rot_trans_3D_x_[2] * wld0_dz[j];
    wld1_dy[j] = rot_trans_3D_y_[0] * wld0_dx[j] + rot_trans_3D_y_[1] * wld0_dy[j] + rot_trans_3D_y_[2] * wld0_dz[j];
    wld1_dz[j] = rot_trans_3D_z_[0] * wld0_dx[j] + rot_trans_3D_z_[1] * wld0_dy[j] + rot_trans_3D_z_[2] * wld0_dz[j];
    wld1_dx[j+3] = rot_trans_3D_dx_[j][0] * wld0_x + rot_trans_3D_dx_[j][1] * wld0_y + rot_trans_3D_dx_[j][2] * wld0_z;
    wld1_dy[j+3] = rot_trans_3D_dy_[j][0] * wld0_x + rot_trans_3D_dy_[j][1] * wld0_y + rot_trans_3D_dy_[j][2] * wld0_z;
    wld1_dz[j+3] = rot_trans_3D_dz_[j][0] * wld0_x + rot_trans_3D_dz_[j][1] * wld0_y + rot_trans_3D_dz_[j][2] * wld0_z;
    wld1_dx[j + 4] = 0.0;
    wld1_dy[j + 5] = 0.0;
    wld1_dz[j + 6] = 0.0;
  }
  wld1_dx[6] = 1.0;
  wld1_dy[7] = 1.0;
  wld1_dz[8] = 1.0;
}
} //end DICe namespace

