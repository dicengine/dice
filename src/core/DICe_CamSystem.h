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

namespace DICe_CamSystem {
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

  const static char * systemType3DStrings[] = {
    "UNKNOWN",
    "GENERIC_SYSTEM",
    "OPENCV",
    "VIC3D",
    "DICE"
  };

  enum Rot_Trans_3D_Params {
    CAM_SYS__ANG_X = 0,
    CAM_SYS__ANG_Y,
    CAM_SYS__ANG_Z,
    CAM_SYS__T_X,
    CAM_SYS__T_Y,
    CAM_SYS__T_Z
  };

}

namespace DICe {


  /*  Keep as template for declarations for projection matricies
  /// free function to estimate a 9 parameter affine projection
  /// \param proj_xl x coordinates of the points in the left image or coordinate system
  /// \param proj_yl y coordinates of the points in the left image or coordinate system
  /// \param proj_xr x coordinates of the points in the right image or coordinate system
  /// \param proj_yr y coordinates of the points in the right image or coordinate system
  /// return value affine_matrix storage for the affine parameters (must be 3x3)
  DICE_LIB_DLL_EXPORT
    Teuchos::SerialDenseMatrix<int_t, double>
    compute_affine_matrix(const std::vector<scalar_t> proj_xl,
      const std::vector<scalar_t> proj_yl,
      const std::vector<scalar_t> proj_xr,
      const std::vector<scalar_t> proj_yr);
  */

  /// \class DICe::CamSystem
  /// \brief A class for camera calibration parameters and computing projection transformations
  ///
  class DICE_LIB_DLL_EXPORT
    CamSystem {
  public:



    /// \brief Default constructor
    /// \param param_file_name the name of the file to parse the calibration parameters from
    CamSystem(const std::string & param_file_name) {
      load_calibration_parameters(param_file_name);
    };

    /// \brief constructor with no args
    CamSystem() {
    };

    /// Pure virtual destructor
    virtual ~CamSystem() {}


    /// \brief load the calibration parameters
    /// \param param_file_name File name of the cal parameters file
    void load_calibration_parameters(const std::string & param_file_name);

    /// \brief save the calibration parameters to an xml file
    /// \param calFile File name of the cal parameters file to write to
    void save_calibration_file(const std::string & calFile, bool allFields = false);

    /// \brief clear the values for all the cameras
    /// \param
    void clear_system();

    //set/get the undeformed camera number
    bool set_Undeformed_Camera(int_t cam_num) {
      bool return_val = false;
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM) {
        if (Cameras_[cam_num].camera_valid(msg_)) {
          undef_cam_ = cam_num;
          return_val = true;
        }
      }
      if (!return_val)undef_cam_ = -1;
      return return_val;
    }
    int_t get_Undeformed_Camera() { return undef_cam_; }

    //set/get the deformed camera number
    bool set_Deformed_Camera(int_t cam_num) {
      bool return_val = false;
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM) {
        if (Cameras_[cam_num].camera_valid(msg_)) {
          def_cam_ = cam_num;
          return_val = true;
        }
      }
      if (!return_val)def_cam_ = -1;
      return return_val;
    }
    int_t get_Deformed_Camera() { return def_cam_; }

    //get the numbers of valid cameras
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

    //get the number of the first camera with a matching identifier
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

    // minimun constructor for the entire camera
    int_t AddCamera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics) {
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
    // add a camera with extrinsic parameters
    int_t AddCamera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics) {
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
    // add a camera with extrinsic parameters and rotation matrix
    int_t AddCamera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics,
      std::vector<std::vector<scalar_t>> & rotation_3x3_matrix) {
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
          return cam_num;
        }
      }
      return cam_num;
    }
    // add all information for the camera
    int_t AddCamera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics,
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

    void RemoveCamera(int_t cam_num)
    {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].clear_camera();
    }

    //Pass through functions to allow access/changes to individual cameras

    ///gets/sets the camera identifier
    void set_CamIdentifier(int_t cam_num, std::string cam_id) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Identifier(cam_id);
    }
    std::string get_CamIdentifier(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Identifier();
      return "";
    }

    ///gets/sets the camera lens identifier
    void set_Camera_Lens(int_t cam_num, std::string camera_lens) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Camera_Lens(camera_lens);
    }
    std::string get_Camera_Lens(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Camera_Lens();
      return "";
    }

    ///gets/sets the camera comment
    void set_Camera_Comments(int_t cam_num, std::string camera_comments) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Camera_Comments(camera_comments);
    }
    std::string get_Camera_Comments(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Camera_Comments();
      return "";
    }

    ///gets/sets the image height
    void set_Camera_Image_Height(int_t cam_num, int_t height) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Image_Height(height);
    }
    int_t get_Camera_Image_Height(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Image_Height();
      return -1;
    }

    ///gets/sets the image width
    void set_Camera_Image_Width(int_t cam_num, int_t width) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Image_Width(width);
    }
    int_t get_Camera_Image_Width(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Image_Width();
      return -1;
    }

    ///gets/sets the pixel depth
    void set_Camera_Pixel_Depth(int_t cam_num, int_t pixel_depth) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Pixel_Depth(pixel_depth);
    }
    int_t get_Camera_Pixel_Depth(int_t cam_num) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].get_Pixel_Depth();
      return -1;
    }


    ///gets/sets intrinsic values
    void set_Camera_Intrinsics(int_t cam_num, std::vector<scalar_t> & intrinsics) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Intrinsics(intrinsics);
    }
    void get_Camera_Intrinsics(int_t cam_num, std::vector<scalar_t> & intrinsics) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].get_Intrinsics(intrinsics);
    }

    ///gets/sets extrinsic values
    void set_Camera_Extrinsics(int_t cam_num, std::vector<scalar_t> & extrinsics) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_Extrinsics(extrinsics);
    }
    void get_Camera_Extrinsics(int_t cam_num, std::vector<scalar_t> & extrinsics) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].get_Extrinsics(extrinsics);
    }

    ///gets/sets rotation matrix
    void set_Camera_3x3_Rotation_Matrix(int_t cam_num, std::vector<std::vector<scalar_t>> & rotation_3x3_matrix) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].set_3x3_Rotation_Matrix(rotation_3x3_matrix);
    }
    void get_3x3_Rotation_Matrix(int_t cam_num, std::vector<std::vector<scalar_t>> & rotation_3x3_matrix) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].get_3x3_Rotation_Matrix(rotation_3x3_matrix);
    }

    ///gets the camera to world transformation matrix
    void get_Camera_Cam_World_Trans_Matrix(int_t cam_num, std::vector<std::vector<scalar_t>> & cam_world_trans) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].get_Cam_World_Trans_Matrix(cam_world_trans);
    }

    ///gets the world to camera transformation matrix
    void get_Camera_World_Cam_Trans_Matrix(int_t cam_num, std::vector<std::vector<scalar_t>> & world_cam_trans) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        Cameras_[cam_num].get_World_Cam_Trans_Matrix(world_cam_trans);
    }

    /// does the camera have enough values to be valid
    bool camera_valid(int_t cam_num, std::string & msg) {
      if (cam_num > -1 && cam_num < MAX_NUM_CAMERAS_PER_SYSTEM)
        return Cameras_[cam_num].camera_valid(msg);
      return false;
    }

    //has the camera been prepped
    bool cameras_prepped() {
      if (undef_cam_ != -1 && def_cam_ != -1)
        return Cameras_[undef_cam_].camera_prepped() && Cameras_[def_cam_].camera_prepped();
      return false;
    }

    //prepares the tranfomation matricies
    void prep_cameras() {
      if (undef_cam_ != -1 && def_cam_ != -1) {
        Cameras_[undef_cam_].prep_camera();
        Cameras_[def_cam_].prep_camera();
      }
    }

    //cross projection mapping and overloads
    void cross_projection_map(scalar_t & img0_x, scalar_t & img0_y, scalar_t & img1_x, scalar_t & img1_y, std::vector<scalar_t> & params);
    void cross_projection_map(std::vector<scalar_t> & img0_x, std::vector<scalar_t> & img0_y, std::vector<scalar_t> & img1_x, 
      std::vector<scalar_t> & img1_y, std::vector<scalar_t> & params);
    void cross_projection_map(scalar_t & img0_x, scalar_t & img0_y, scalar_t & img1_x, scalar_t & img1_y, std::vector<scalar_t> & params, 
      std::vector<scalar_t> & img1_dx, std::vector<scalar_t> & img1_dy);
    void cross_projection_map(std::vector<scalar_t> & img0_x, std::vector<scalar_t> & img0_y, std::vector<scalar_t> & img1_x, 
      std::vector<scalar_t> & img1_y, std::vector<scalar_t> & params, std::vector<std::vector<scalar_t>> & img1_dx, std::vector<std::vector<scalar_t>> & img1_dy);

    //3D rotation/translation transformation and overloads
    void rot_trans_3D(std::vector<scalar_t> & wld0_x, std::vector<scalar_t> & wld0_y, std::vector<scalar_t> & wld0_z, 
      std::vector<scalar_t> & wld1_x, std::vector<scalar_t> & wld1_y, std::vector<scalar_t> & wld1_z, std::vector<scalar_t> & params);
    void rot_trans_3D(scalar_t & wld0_x, scalar_t & wld0_y, scalar_t & wld0_z,
      scalar_t & wld1_x, scalar_t & wld1_y, scalar_t & wld1_z, std::vector<scalar_t> & params);
    void rot_trans_3D(std::vector<scalar_t> & wld0_x, std::vector<scalar_t> & wld0_y, std::vector<scalar_t> & wld0_z,
      std::vector<scalar_t> & wld1_x, std::vector<scalar_t> & wld1_y, std::vector<scalar_t> & wld1_z, std::vector<scalar_t> & params,
      std::vector < std::vector<scalar_t>> & wld1_dx, std::vector < std::vector<scalar_t>> & wld1_dy, std::vector < std::vector<scalar_t>> & wld1_dz);
    void rot_trans_3D(scalar_t & wld0_x, scalar_t & wld0_y, scalar_t & wld0_z, scalar_t & wld1_x, scalar_t & wld1_y, scalar_t & wld1_z, std::vector <scalar_t> & params,
      std::vector<scalar_t> & wld1_dx, std::vector<scalar_t> & wld1_dy, std::vector<scalar_t> & wld1_dz);



  private:
    void pre_projection_(int_t num_pnts, int_t num_params, bool partials);

    //prepares coefficients for the 3D rotation/translation transformation
    void CamSystem::pre_rot_trans_3D(std::vector<scalar_t> params, bool partials);


    /// these are for compatibility with triangulation and only support 2 camera systems
    /// 12 parameters that define a user supplied transforamation (independent from intrinsic and extrinsic parameters)
    std::vector<std::vector<scalar_t>> user_trans_4x4_params_;

    /// 8 parameters that define a projective transform (independent from intrinsic and extrinsic parameters)
    std::vector<scalar_t> user_trans_6_params_;

    /// 3x3 openCV rotation parameters from sterio calibration
    std::vector<std::vector<scalar_t>> rotation_3x3_params_;


    std::string cal_file_ID_ = "DICe XML Calibration File";

    int_t num_cams_ = 0;
    int_t undef_cam_ = -1;
    int_t def_cam_ = -1;
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

