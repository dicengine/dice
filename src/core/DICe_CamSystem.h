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

#ifndef DICE_CAM_SYSTEM_H
#define DICE_CAM_SYSTEM_H

#include <DICe.h>
#include <DICe_Camera.h>
#include <DICe_Image.h>

#ifdef DICE_TPETRA
#include "DICe_MultiFieldTpetra.h"
#else
#include "DICe_MultiFieldEpetra.h"
#endif

#include <Teuchos_SerialDenseMatrix.hpp>
#include "DICe_ParameterUtilities.h"

#include <cassert>
#include <vector>


/// \namespace DICe_CamSystem holds the enumerated values and string representations for CamSystem members
namespace DICe_CamSystem {
  /// \enum System_Type_3D valid system types
  enum System_Type_3D {
    UNKNOWN_SYSTEM = 0,
    GENERIC_SYSTEM,
    OPENCV,
    VIC3D,
    DICE,
    //DON"T ADD ANY BELOW MAX
    MAX_SYSTEM_TYPE_3D,
    NO_SUCH_SYSTEM_TYPE_3D
  };

  /// \systemType3DStrings string representation of valid system types
  const static char * systemType3DStrings[] = {
    "UNKNOWN",
    "GENERIC_SYSTEM",
    "OPENCV",
    "VIC3D",
    "DICE"
  };

  /// \enum Rot_Trans_3D_Params rigid body rotation parameter index
  enum Rot_Trans_3D_Params {
    ANG_X = 0,
    ANG_Y,
    ANG_Z,
    T_X,
    T_Y,
    T_Z
  };

}

namespace DICe {

  /// \class DICe::CamSystem
  /// \brief A class for camera calibration parameters and computing projection transformations
  ///
  class DICE_LIB_DLL_EXPORT
    CamSystem {
  public:

    /// \brief constructor with no args
    CamSystem() {
    };


    /// \brief constructor
    /// \param param_file_name the name of the file to parse the calibration parameters from
    CamSystem(const std::string & param_file_name) {
      read_calibration_parameters(param_file_name);
    };

    // Pure virtual destructor
    virtual ~CamSystem() {}


    /// \brief read the calibration parameters 
    /// \param param_file_name File name of the cal parameters file
    void read_calibration_parameters(const std::string & param_file_name);

    /// \brief write the calibration parameters to an xml file
    /// \param calFile File name of the cal parameters file to write to
    void write_calibration_file(const std::string & calFile, bool allFields = false);

    /// \brief clear all the parameter values for the system and all cameras
    /// \param
    void clear_system();

    /// \brief set the system type
    /// \param system_type a DICe_CamSystem::System_Type_3D enum
    void set_System_Type(DICe_CamSystem::System_Type_3D system_type) {
      sys_type_ = (int_t)system_type;
    }
    /// \brief get the integer value for the current system type
    /// \param system_type integer system type value
    void get_System_Type(int_t & system_type) {
      system_type = sys_type_;
    }
    /// \brief get the string value for the current system type
    /// \param system_type string system type value
    void get_System_Type(std::string & system_type) {
      system_type = DICe_CamSystem::systemType3DStrings[sys_type_];
    }


    /// \brief set the source camera number
    /// \param cam_num number of the camera to set as the source 
    bool set_Source_Camera(int_t cam_num) {
      bool return_val = false;
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM) {
        if (Cameras_[cam_num].camera_valid(msg_)) {
          source_cam_ = cam_num;
          return_val = true;
        }
      }
      if (!return_val)source_cam_ = -1;
      return return_val;
    }
    /// \brief get the source camera number
    /// \param cam_num number of the camera currently set as the source 
    int_t get_Source_Camera() { return source_cam_; }


    /// \brief set the target camera number
    /// \param cam_num number of the camera to set as the target 
    bool set_Target_Camera(int_t cam_num) {
      bool return_val = false;
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM) {
        if (Cameras_[cam_num].camera_valid(msg_)) {
          target_cam_ = cam_num;
          return_val = true;
        }
      }
      if (!return_val)target_cam_ = -1;
      return return_val;
    }
    /// \brief get the target camera number
    /// \param cam_num number of the camera currently set as the target 
    int_t get_Target_Camera() { return target_cam_; }

    /// \brief get the numbers of valid cameras
    /// \param valid_cams array of valid camera numbers
    int_t get_Valid_Cameras(std::vector<int_t> & valid_cams) {
      valid_cams.clear();
      int_t return_val = 0;
      for (int_t i = 0; i < MAX_NUM_CAMERAS_PER_SYSTEM; i++) {
        if (Cameras_[i].camera_valid(msg_)) {
          valid_cams.push_back(i);
          return_val++;
        }
      }
      return return_val;
    }

    /// \brief returns the number of the first camera with a matching identifier
    /// \param cam_id camera identifying string
    int get_Camera_Num_From_ID(std::string cam_id) {
      stringToUpper(cam_id);
      std::string id_val;
      for (int_t i = 0; i < MAX_NUM_CAMERAS_PER_SYSTEM; i++) {
        id_val = Cameras_[i].get_Identifier();
        stringToUpper(id_val);
        if (id_val==cam_id)return i;
      }
      return -1;
    }

    /// \brief add a camera to the camera system with no extrinsic values
    /// returns the camera number for the new camera
    /// \param cam_id string descripter of the camera
    /// \param image_height height of the image in pixels
    /// \param image_width width of the image in pixels
    /// \param intrinsics array of intrinsic values ordered by DICe_Camera::Cam_Intrinsic_Params
    int_t add_Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics) {
      int_t cam_num = -1;
      for (int_t i = 0; i < MAX_NUM_CAMERAS_PER_SYSTEM; i++) {
        if (!Cameras_[i].camera_filled()) {
          cam_num = i;
          Cameras_[i].clear_camera();
          Cameras_[i].set_Identifier(cam_id);
          Cameras_[i].set_Intrinsics(intrinsics);
          Cameras_[i].set_Image_Height(image_height);
          Cameras_[i].set_Image_Width(image_width);
          return cam_num;
        }
      }
      return cam_num;
    }

    /// \brief add a camera to the camera system with extrinsic values
    /// returns the camera number for the new camera
    /// \param cam_id string descripter of the camera
    /// \param image_height height of the image in pixels
    /// \param image_width width of the image in pixels
    /// \param intrinsics array of intrinsic values ordered by DICe_Camera::Cam_Intrinsic_Params
    /// \param extrinsics array of extrinsic values ordered by DICe_Camera::Cam_Extrinsic_Params
    int_t add_Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics) {
      int_t cam_num = -1;
      for (int_t i = 0; i < MAX_NUM_CAMERAS_PER_SYSTEM; i++) {
        if (!Cameras_[i].camera_filled()) {
          cam_num = i;
          Cameras_[i].clear_camera();
          Cameras_[i].set_Identifier(cam_id);
          Cameras_[i].set_Intrinsics(intrinsics);
          Cameras_[i].set_Image_Height(image_height);
          Cameras_[i].set_Image_Width(image_width);
          Cameras_[i].set_Extrinsics(extrinsics);
          return cam_num;
        }
      }
      return cam_num;
    }

    /// \brief add a camera to the camera system setting all of the parameters
    /// returns the camera number for the new camera
    /// \param cam_id string descripter of the camera
    /// \param image_height height of the image in pixels
    /// \param image_width width of the image in pixels
    /// \param intrinsics array of intrinsic values ordered by DICe_Camera::Cam_Intrinsic_Params
    /// \param extrinsics array of extrinsic values ordered by DICe_Camera::Cam_Extrinsic_Params
    /// \param rotation_3x3_matrix matrix describing the angular relationship between the camera and the world coordinates
    /// \param camera_lens description of the camera lens
    /// \param camera_comments other comments about the camera
    /// \param pixel_depth pixel depth of the images
    int_t add_Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics,
      std::vector<std::vector<scalar_t>> & rotation_3x3_matrix, std::string camera_lens = "", std::string camera_comments = "",
      int_t pixel_depth = 0) {
      int_t cam_num = -1;
      for (int_t i = 0; i < MAX_NUM_CAMERAS_PER_SYSTEM; i++) {
        if (!Cameras_[i].camera_filled()) {
          cam_num = i;
          Cameras_[i].clear_camera();
          Cameras_[i].set_Identifier(cam_id);
          Cameras_[i].set_Intrinsics(intrinsics);
          Cameras_[i].set_Image_Height(image_height);
          Cameras_[i].set_Image_Width(image_width);
          Cameras_[i].set_Extrinsics(extrinsics);
          Cameras_[i].set_3x3_Rotation_Matrix(rotation_3x3_matrix);
          Cameras_[i].set_Camera_Lens(camera_lens);
          Cameras_[i].set_Camera_Comments(camera_comments);
          return cam_num;
        }
      }
      return cam_num;
    }

    /// \brief remove a camera from the system
    /// \cam_num number of the camera to remove
    void remove_Camera(int_t cam_num)
    {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].clear_camera();
    }


    //Pass through functions to allow access/changes to individual cameras

    ///sets a camera identifier
    /// \param cam_num number of the camera in the system
    /// \param cam_id string descripter of the camera
    void set_CamIdentifier(int_t cam_num, std::string cam_id) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Identifier(cam_id);
    }
    ///gets a camera identifier
    /// \param cam_num number of the camera in the system
    /// \param cam_id string descripter of the camera
    std::string get_CamIdentifier(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Identifier();
      return "";
    }

    ///sets a camera lens identifier
    /// \param cam_num number of the camera in the system
    /// \param camera_lens string descripter of the camera lens
    void set_Camera_Lens(int_t cam_num, std::string camera_lens) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Camera_Lens(camera_lens);
    }
    ///gets a camera lens identifier
    /// \param cam_num number of the camera in the system
    /// \param camera_lens string descripter of the camera lens
    std::string get_Camera_Lens(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Camera_Lens();
      return "";
    }

    ///sets a camera comment
    /// \param cam_num number of the camera in the system
    /// \param camera_comments comments about the camera
    void set_Camera_Comments(int_t cam_num, std::string camera_comments) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Camera_Comments(camera_comments);
    }
    ///gets a camera comment
    /// \param cam_num number of the camera in the system
    /// \param camera_comments comments about the camera
    std::string get_Camera_Comments(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Camera_Comments();
      return "";
    }

    ///sets the image height
    /// \param cam_num number of the camera in the system
    /// \param height height of the image from this camera
    void set_Camera_Image_Height(int_t cam_num, int_t height) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Image_Height(height);
    }
    ///gets the image height
    /// \param cam_num number of the camera in the system
    /// \param height height of the image from this camera
    int_t get_Camera_Image_Height(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Image_Height();
      return -1;
    }

    ///sets the image width
    /// \param cam_num number of the camera in the system
    /// \param width width of the image from this camera    
    void set_Camera_Image_Width(int_t cam_num, int_t width) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Image_Width(width);
    }
    ///sets the image width
    /// \param cam_num number of the camera in the system
    /// \param width width of the image from this camera    
    int_t get_Camera_Image_Width(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Image_Width();
      return -1;
    }

    ///sets the pixel depth
    /// \param cam_num number of the camera in the system
    /// \param pixel_depth pixel_depth of the image from this camera    
    void set_Camera_Pixel_Depth(int_t cam_num, int_t pixel_depth) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Pixel_Depth(pixel_depth);
    }
    ///gets the pixel depth
    /// \param cam_num number of the camera in the system
    /// \param pixel_depth pixel_depth of the image from this camera    
    int_t get_Camera_Pixel_Depth(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Pixel_Depth();
      return -1;
    }


    /// sets intrinsic values for the camera
    /// \param cam_num number of the camera in the system
    /// \param intrinsics array of intrinsic values ordered by DICe_Camera::Cam_Intrinsic_Params
    void set_Camera_Intrinsics(int_t cam_num, std::vector<scalar_t> & intrinsics) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Intrinsics(intrinsics);
    }
    /// gets intrinsic values for the camera
    /// \param cam_num number of the camera in the system
    /// \param intrinsics array of intrinsic values ordered by DICe_Camera::Cam_Intrinsic_Params
    void get_Camera_Intrinsics(int_t cam_num, std::vector<scalar_t> & intrinsics) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].get_Intrinsics(intrinsics);
    }

    /// sets extrinsic values for the camera
    /// \param cam_num number of the camera in the system
    /// \param extrinsics array of extrinsic values ordered by DICe_Camera::Cam_Extrinsic_Params
    void set_Camera_Extrinsics(int_t cam_num, std::vector<scalar_t> & extrinsics) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Extrinsics(extrinsics);
    }
    /// gets extrinsic values for the camera
    /// \param cam_num number of the camera in the system
    /// \param extrinsics array of extrinsic values ordered by DICe_Camera::Cam_Extrinsic_Params
    void get_Camera_Extrinsics(int_t cam_num, std::vector<scalar_t> & extrinsics) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].get_Extrinsics(extrinsics);
    }

    /// sets the 3x3 rotation matrix for the camera
    /// \param cam_num number of the camera in the system
    /// \param rotation_3x3_matrix matrix describing the angular relationship between the camera and the world coordinates
    void set_Camera_3x3_Rotation_Matrix(int_t cam_num, std::vector<std::vector<scalar_t>> & rotation_3x3_matrix) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_3x3_Rotation_Matrix(rotation_3x3_matrix);
    }
    /// gets the 3x3 rotation matrix for the camera
    /// \param cam_num number of the camera in the system
    /// \param rotation_3x3_matrix matrix describing the angular relationship between the camera and the world coordinates
    void get_3x3_Rotation_Matrix(int_t cam_num, std::vector<std::vector<scalar_t>> & rotation_3x3_matrix) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].get_3x3_Rotation_Matrix(rotation_3x3_matrix);
    }

    ///gets the 4x4 camera to world transformation matrix
    /// \param cam_num number of the camera in the system
    /// \param cam_world_trans 4x4 [R|T] transformation array
    void get_Camera_Cam_World_Trans_Matrix(int_t cam_num, std::vector<std::vector<scalar_t>> & cam_world_trans) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].get_Cam_World_Trans_Matrix(cam_world_trans);
    }

    ///gets the 4x4 world to camera transformation matrix
    /// \param cam_num number of the camera in the system
    /// \param cam_world_trans 4x4 [R|T] transformation array
    void get_Camera_World_Cam_Trans_Matrix(int_t cam_num, std::vector<std::vector<scalar_t>> & world_cam_trans) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].get_World_Cam_Trans_Matrix(world_cam_trans);
    }

    /// does the camera have enough values to be valid
    /// returns true or false
    /// \param cam_num number of the camera in the system
    /// \msg text string with reason camera is not ready
    bool camera_valid(int_t cam_num, std::string & msg) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].camera_valid(msg);
      return false;
    }

    /// does the camera have enough values to be valid
    /// returns true or false
    /// \param cam_num number of the camera in the system
    bool camera_valid(int_t cam_num) {
      std::string msg;
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].camera_valid(msg);
      return false;
    }

    /// were the cameras prepped after the camera values were extablished
    /// returns true or false
    bool cameras_prepped() {
      if (source_cam_ != -1 && target_cam_ != -1)
        return Cameras_[source_cam_].camera_prepped() && Cameras_[target_cam_].camera_prepped();
      return false;
    }

    /// prepares the tranfomation matricies for all of the valid cameras
    void prep_cameras() {
      if (source_cam_ != -1 && target_cam_ != -1) {
        Cameras_[source_cam_].prep_camera();
        Cameras_[target_cam_].prep_camera();
      }
    }

    //cross projection mapping and overloads
    /// \brief 3 parameter projection routine from the source camera to the target camera 
    /// \param img0_x x image location in the source camera
    /// \param img0_y y image location in the source camera
    /// \param img1_x projected x image location in the target camera
    /// \param img1_y projected y image location in the target camera
    /// \param params array of ZP, THETA, PHI values that govern the projection
    void cross_projection_map(const scalar_t & img0_x, const scalar_t & img0_y, scalar_t & img1_x, scalar_t & img1_y, std::vector<scalar_t> & params);
    /// \brief 3 parameter projection routine from the source camera to the target camera 
    /// \param img0_x array of x image location in the source camera
    /// \param img0_y array of y image location in the source camera
    /// \param img1_x array of projected x image location in the target camera
    /// \param img1_y array of projected y image location in the target camera
    /// \param params array of ZP, THETA, PHI values that govern the projection
    void cross_projection_map(std::vector<scalar_t> & img0_x, std::vector<scalar_t> & img0_y, std::vector<scalar_t> & img1_x,
      std::vector<scalar_t> & img1_y, std::vector<scalar_t> & params);
    /// \brief 3 parameter projection routine from the source camera to the target camera with partials
    /// \param img0_x x image location in the source camera
    /// \param img0_y y image location in the source camera
    /// \param img1_x projected x image location in the target camera
    /// \param img1_y projected y image location in the target camera
    /// \param params array of ZP, THETA, PHI values that govern the projection
    /// \param img1_dx array of the partials in the x location wrt (ZP, THETA, PHI)
    /// \param img1_dy array of the partials in the y location wrt (ZP, THETA, PHI)
    void cross_projection_map(const scalar_t & img0_x, const scalar_t & img0_y, scalar_t & img1_x, scalar_t & img1_y, std::vector<scalar_t> & params,
      std::vector<scalar_t> & img1_dx, std::vector<scalar_t> & img1_dy);
    /// \brief 3 parameter projection routine from the source camera to the target camera with partials
    /// \param img0_x array of x image location in the source camera
    /// \param img0_y array of y image location in the source camera
    /// \param img1_x array of projected x image location in the target camera
    /// \param img1_y array of projected y image location in the target camera
    /// \param params array of ZP, THETA, PHI values that govern the projection
    /// \param img1_dx array of arrays of the partials in the x location wrt (ZP, THETA, PHI)
    /// \param img1_dy array of arrays of the partials in the y location wrt (ZP, THETA, PHI)
    void cross_projection_map(std::vector<scalar_t> & img0_x, std::vector<scalar_t> & img0_y, std::vector<scalar_t> & img1_x,
      std::vector<scalar_t> & img1_y, std::vector<scalar_t> & params, std::vector<std::vector<scalar_t>> & img1_dx, std::vector<std::vector<scalar_t>> & img1_dy);


    /// fixed projection, 3D rigid body mapping
    /// points in space are established by projecting them using the 3 parameter projection, they are then allowed to rotate/translate in a rigid body manner to a second location
    /// which is then projected into the target camera. ZP, THETA and PHI are not allowed to vary
    /// \param img0_x x image location in the source camera
    /// \param img0_y y image location in the source camera
    /// \param img1_x projected x image location in the target camera
    /// \param img1_y projected y image location in the target camera
    /// \param proj_params fixed ZP, THETA, PHI values that govern the projection
    /// \param rigid_body_params 6 rotation/translation parameters to describe the rigid body motion
    void fixed_proj_3DRB_map(const scalar_t & img0_x, const scalar_t & img0_y, scalar_t & img1_x, scalar_t & img1_y, std::vector<scalar_t> & proj_params, std::vector<scalar_t> & rigid_body_params);
    /// fixed projection, 3D rigid body mapping
    /// points in space are established by projecting them using the 3 parameter projection, they are then allowed to rotate/translate in a rigid body manner to a second location
    /// which is then projected into the target camera. ZP, THETA and PHI are not allowed to vary
    /// \param img0_x array of x image location in the source camera
    /// \param img0_y array of y image location in the source camera
    /// \param img1_x array of projected x image location in the target camera
    /// \param img1_y array of projected y image location in the target camera
    /// \param proj_params fixed ZP, THETA, PHI values that govern the projection
    /// \param rigid_body_params 6 rotation/translation parameters to describe the rigid body motion
    void fixed_proj_3DRB_map(std::vector<scalar_t> & img0_x, std::vector<scalar_t> & img0_y, std::vector<scalar_t> & img1_x,
      std::vector<scalar_t> & img1_y, std::vector<scalar_t> & proj_params, std::vector<scalar_t> & rigid_body_params);
    /// fixed projection, 3D rigid body mapping with partials
    /// points in space are established by projecting them using the 3 parameter projection, they are then allowed to rotate/translate in a rigid body manner to a second location
    /// which is then projected into the target camera. ZP, THETA and PHI are not allowed to vary
    /// \param img0_x x image location in the source camera
    /// \param img0_y y image location in the source camera
    /// \param img1_x projected x image location in the target camera
    /// \param img1_y projected y image location in the target camera
    /// \param proj_params fixed ZP, THETA, PHI values that govern the projection
    /// \param rigid_body_params 6 rotation/translation parameters to describe the rigid body motion
    /// \param img1_dx array of the partials in the x location wrt rigid_body_parms
    /// \param img1_dy array of the partials in the y location wrt rigid_body_parms
    void fixed_proj_3DRB_map(const scalar_t & img0_x, const scalar_t & img0_y, scalar_t & img1_x, scalar_t & img1_y, std::vector<scalar_t> & proj_params, std::vector<scalar_t> & rigid_body_params,
      std::vector<scalar_t> & img1_dx, std::vector<scalar_t> & img1_dy);
    /// fixed projection, 3D rigid body mapping with partials
    /// points in space are established by projecting them using the 3 parameter projection, they are then allowed to rotate/translate in a rigid body manner to a second location
    /// which is then projected into the target camera. ZP, THETA and PHI are not allowed to vary
    /// \param img0_x array of x image location in the source camera
    /// \param img0_y array of y image location in the source camera
    /// \param img1_x array of projected x image location in the target camera
    /// \param img1_y array of projected y image location in the target camera
    /// \param proj_params fixed ZP, THETA, PHI values that govern the projection
    /// \param rigid_body_params 6 rotation/translation parameters to describe the rigid body motion
    /// \param img1_dx array of arrays of the partials in the x location wrt rigid_body_parms
    /// \param img1_dy array of arrays of the partials in the y location wrt rigid_body_parms
    void fixed_proj_3DRB_map(std::vector<scalar_t> & img0_x, std::vector<scalar_t> & img0_y, std::vector<scalar_t> & img1_x,
      std::vector<scalar_t> & img1_y, std::vector<scalar_t> & proj_params, std::vector<scalar_t> & rigid_body_params, std::vector<std::vector<scalar_t>> & img1_dx, 
      std::vector<std::vector<scalar_t>> & img1_dy);


    ///rigid body 3D rotation/translation transformation
    /// \param wld0_x initial x world position of the point
    /// \param wld0_y initial y world position of the point
    /// \param wld0_z initial z world position of the point
    /// \param wld1_x x world position of the point after the transformation
    /// \param wld1_y y world position of the point after the transformation
    /// \param wld1_z z world position of the point after the transformation
    /// \param params 6 rotation/translation parameters to describe the rigid body motion
    void rot_trans_3D(scalar_t & wld0_x, scalar_t & wld0_y, scalar_t & wld0_z,
      scalar_t & wld1_x, scalar_t & wld1_y, scalar_t & wld1_z, std::vector<scalar_t> & params);
    ///rigid body 3D rotation/translation transformation
    /// \param wld0_x array of initial x world position of the point
    /// \param wld0_y array of initial y world position of the point
    /// \param wld0_z array of initial z world position of the point
    /// \param wld1_x array of x world position of the point after the transformation
    /// \param wld1_y array of y world position of the point after the transformation
    /// \param wld1_z array of z world position of the point after the transformation
    /// \param params 6 rotation/translation parameters to describe the rigid body motion
    void rot_trans_3D(std::vector<scalar_t> & wld0_x, std::vector<scalar_t> & wld0_y, std::vector<scalar_t> & wld0_z,
      std::vector<scalar_t> & wld1_x, std::vector<scalar_t> & wld1_y, std::vector<scalar_t> & wld1_z, std::vector<scalar_t> & params);
    ///rigid body 3D rotation/translation transformation with no incoming partials and outgoing partials wrt params
    /// \param wld0_x initial x world position of the point
    /// \param wld0_y initial y world position of the point
    /// \param wld0_z initial z world position of the point
    /// \param wld1_x x world position of the point after the transformation
    /// \param wld1_y y world position of the point after the transformation
    /// \param wld1_z z world position of the point after the transformation
    /// \param params 6 rotation/translation parameters to describe the rigid body motion
    /// \param wld1_dx array of the partials in the x world location wrt params
    /// \param wld1_dy array of the partials in the y world location wrt params
    /// \param wld1_dy array of the partials in the z world location wrt params
    void rot_trans_3D(scalar_t & wld0_x, scalar_t & wld0_y, scalar_t & wld0_z, scalar_t & wld1_x, scalar_t & wld1_y, scalar_t & wld1_z, std::vector <scalar_t> & params,
      std::vector<scalar_t> & wld1_dx, std::vector<scalar_t> & wld1_dy, std::vector<scalar_t> & wld1_dz);
    ///rigid body 3D rotation/translation transformation with no incoming partials and outgoing partials wrt params
    /// \param wld0_x array of initial x world position of the point
    /// \param wld0_y array of initial y world position of the point
    /// \param wld0_z array of initial z world position of the point
    /// \param wld1_x array of x world position of the point after the transformation
    /// \param wld1_y array of y world position of the point after the transformation
    /// \param wld1_z array of z world position of the point after the transformation
    /// \param params 6 rotation/translation parameters to describe the rigid body motion
    /// \param wld1_dx array of arrays of the partials in the x world location wrt params
    /// \param wld1_dy array of arrays of the partials in the y world location wrt params
    /// \param wld1_dy array of arrays of the partials in the z world location wrt params
    void rot_trans_3D(std::vector<scalar_t> & wld0_x, std::vector<scalar_t> & wld0_y, std::vector<scalar_t> & wld0_z,
      std::vector<scalar_t> & wld1_x, std::vector<scalar_t> & wld1_y, std::vector<scalar_t> & wld1_z, std::vector<scalar_t> & params,
      std::vector < std::vector<scalar_t>> & wld1_dx, std::vector < std::vector<scalar_t>> & wld1_dy, std::vector < std::vector<scalar_t>> & wld1_dz);
    ///rigid body 3D rotation/translation transformation with incoming partials and outgoing partials wrt params
    /// \param wld0_x initial x world position of the point
    /// \param wld0_y initial y world position of the point
    /// \param wld0_z initial z world position of the point
    /// \param wld1_x x world position of the point after the transformation
    /// \param wld1_y y world position of the point after the transformation
    /// \param wld1_z z world position of the point after the transformation
    /// \param params 6 rotation/translation parameters to describe the rigid body motion
    /// \param wld0_dx array of the incoming partials in the x world location from other parameters
    /// \param wld0_dy array of the incoming partials in the y world location from other parameters
    /// \param wld0_dy array of the incoming partials in the z world location from other parameters
    /// \param wld1_dx array of the partials in the x world location wrt the other parameters and params
    /// \param wld1_dy array of the partials in the y world location wrt the other parameters and params
    /// \param wld1_dy array of the partials in the z world location wrt the other parameters and params
    void rot_trans_3D(scalar_t & wld0_x, scalar_t & wld0_y, scalar_t & wld0_z, scalar_t & wld1_x, scalar_t & wld1_y, scalar_t & wld1_z, std::vector <scalar_t> & params,
      std::vector<scalar_t> & wld0_dx, std::vector<scalar_t> & wld0_dy, std::vector<scalar_t> & wld0_dz,
      std::vector<scalar_t> & wld1_dx, std::vector<scalar_t> & wld1_dy, std::vector<scalar_t> & wld1_dz);
    ///rigid body 3D rotation/translation transformation with incoming partials and outgoing partials wrt params
    /// \param wld0_x array of initial x world position of the point
    /// \param wld0_y array of initial y world position of the point
    /// \param wld0_z array of initial z world position of the point
    /// \param wld1_x array of x world position of the point after the transformation
    /// \param wld1_y array of y world position of the point after the transformation
    /// \param wld1_z array of z world position of the point after the transformation
    /// \param params 6 rotation/translation parameters to describe the rigid body motion
    /// \param wld0_dx array of the incoming partials in the x world location from other parameters
    /// \param wld0_dy array of the incoming partials in the y world location from other parameters
    /// \param wld0_dy array of the incoming partials in the z world location from other parameters
    /// \param wld1_dx array of of arrays the partials in the x world location wrt the other parameters and params
    /// \param wld1_dy array of of arrays the partials in the y world location wrt the other parameters and params
    /// \param wld1_dy array of of arrays the partials in the z world location wrt the other parameters and params
    void rot_trans_3D(std::vector<scalar_t> & wld0_x, std::vector<scalar_t> & wld0_y, std::vector<scalar_t> & wld0_z,
      std::vector<scalar_t> & wld1_x, std::vector<scalar_t> & wld1_y, std::vector<scalar_t> & wld1_z, std::vector<scalar_t> & params,
      std::vector < std::vector<scalar_t>> & wld0_dx, std::vector < std::vector<scalar_t>> & wld0_dy, std::vector < std::vector<scalar_t>> & wld0_dz,
      std::vector < std::vector<scalar_t>> & wld1_dx, std::vector < std::vector<scalar_t>> & wld1_dy, std::vector < std::vector<scalar_t>> & wld1_dz);


  private:
    /// \brief calculates the fixed coefficients prior to projection transformations
    void pre_projection_(int_t num_pnts, int_t num_params, bool partials);

    /// \brief prepares coefficients for the 3D rotation/translation transformation
    void pre_rot_trans_3D(std::vector<scalar_t> params, bool partials);

    // these are for compatibility with triangulation and only support 2 camera systems
    //12 parameters that define a user supplied transforamation (independent from intrinsic and extrinsic parameters)
    std::vector<std::vector<scalar_t>> user_trans_4x4_params_;

    // 8 parameters that define a projective transform (independent from intrinsic and extrinsic parameters)
    std::vector<scalar_t> user_trans_6_params_;

    // 3x3 openCV rotation parameters from sterio calibration
    std::vector<std::vector<scalar_t>> rotation_3x3_params_;

    // identifier for calibration files
    std::string cal_file_ID_ = "DICe XML Calibration File";

    int_t num_cams_ = 0;
    int_t source_cam_ = -1;
    int_t target_cam_ = -1;
    std::string msg_;


    int_t sys_type_ = (int_t)DICe_CamSystem::UNKNOWN_SYSTEM;
    bool has_6_transform_ = false;
    bool has_4x4_transform_ = false;
    bool has_opencv_rot_trans_ = false;
    bool valid_cal_file_ = false;
    std::stringstream cal_file_error_;

    //create an array of generic cameras
    const int_t MAX_NUM_CAMERAS_PER_SYSTEM = 16;
    DICe::Camera Cameras_[16];

    //arrays for the transformations
    std::vector<scalar_t> img_x_;
    std::vector<scalar_t> img_y_;
    std::vector<scalar_t> sen_x_;
    std::vector<scalar_t> sen_y_;
    std::vector<scalar_t> cam_x_;
    std::vector<scalar_t> cam_y_;
    std::vector<scalar_t> cam_z_;
    std::vector<scalar_t> wld0_x_;
    std::vector<scalar_t> wld0_y_;
    std::vector<scalar_t> wld0_z_;
    std::vector<scalar_t> wld1_x_;
    std::vector<scalar_t> wld1_y_;
    std::vector<scalar_t> wld1_z_;

    std::vector<std::vector<scalar_t>>  img_dx_;
    std::vector<std::vector<scalar_t>>  img_dy_;
    std::vector<std::vector<scalar_t>>  sen_dx_;
    std::vector<std::vector<scalar_t>>  sen_dy_;
    std::vector<std::vector<scalar_t>>  cam_dx_;
    std::vector<std::vector<scalar_t>>  cam_dy_;
    std::vector<std::vector<scalar_t>>  cam_dz_;
    std::vector<std::vector<scalar_t>>  wld0_dx_;
    std::vector<std::vector<scalar_t>>  wld0_dy_;
    std::vector<std::vector<scalar_t>>  wld0_dz_;
    std::vector<std::vector<scalar_t>>  wld1_dx_;
    std::vector<std::vector<scalar_t>>  wld1_dy_;
    std::vector<std::vector<scalar_t>>  wld1_dz_;

    std::vector<scalar_t>  rot_trans_3D_x_;
    std::vector<scalar_t>  rot_trans_3D_y_;
    std::vector<scalar_t>  rot_trans_3D_z_;
    std::vector<std::vector<scalar_t>>  rot_trans_3D_dx_;
    std::vector<std::vector<scalar_t>>  rot_trans_3D_dy_;
    std::vector<std::vector<scalar_t>>  rot_trans_3D_dz_;

};

}// End DICe Namespace

#endif

