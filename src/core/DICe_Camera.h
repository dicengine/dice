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

#ifndef DICE_CAMERA_H
#define DICE_CAMERA_H
#include <DICe.h>
#include <DICe_Image.h>
#ifdef DICE_TPETRA
#include "DICe_MultiFieldTpetra.h"
#else
#include "DICe_MultiFieldEpetra.h"
#endif
#include <Teuchos_SerialDenseMatrix.hpp>
#include <cassert>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

/// \namespace holds the enumerated values and string representations for the Camera class members


namespace DICe {

/// \class DICe::Camera
/// \brief A class for camera calibration parameters and computing single camera projection based transformations
///
class DICE_LIB_DLL_EXPORT
Camera {
public:

  /// \enum Lens_Distortion_Model valid lens distortion model values
  enum Lens_Distortion_Model {
    NO_LENS_DISTORTION = 0,
    OPENCV_LENS_DISTORTION,
    VIC3D_LENS_DISTORTION,
    K1R1_K2R2_K3R3,
    K1R2_K2R4_K3R6,
    K1R3_K2R5_K3R7,
    //DON"T ADD ANY BELOW MAX
    MAX_LENS_DISTORTION_MODEL,
    NO_SUCH_LENS_DISTORTION_MODEL
  };

  static std::string to_string(Lens_Distortion_Model in){
    assert(in < MAX_LENS_DISTORTION_MODEL);
    const static char * lensDistStrings[] = {
      "NO_LENS_DISTORTION",
      "OPENCV_LENS_DISTORTION",
      "VIC3D_LENS_DISTORTION",
      "K1R1_K2R2_K3R3",
      "K1R2_K2R4_K3R6",
      "K1R3_K2R5_K3R7"
    };
    return lensDistStrings[in];
  };

  static Lens_Distortion_Model string_to_lens_distortion_model(std::string & in){
    // convert the string to uppercase
    std::transform(in.begin(), in.end(),in.begin(),::toupper);
    for(int_t i=0;i<MAX_LENS_DISTORTION_MODEL;++i){
      if(to_string(static_cast<Lens_Distortion_Model>(i))==in) return static_cast<Lens_Distortion_Model>(i);
    }
    std::cout << "Error: Lens_Distortion_Model " << in << " does not exist." << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"");
    return NO_SUCH_LENS_DISTORTION_MODEL; // prevent no return errors
  }

  /// \enum Cam_Intrinsic_Param valid intrinsic parameters index values
  enum Cam_Intrinsic_Param {
    CX = 0,
    CY,
    FX,
    FY,
    FS,
    K1,
    K2,
    K3,
    K4,
    K5,
    K6,
    P1,
    P2,
    S1,
    S2,
    S3,
    S4,
    T1,
    T2,
    LENS_DISTORTION_MODEL,
    //DON"T ADD ANY BELOW MAX
    MAX_CAM_INTRINSIC_PARAM,
    NO_SUCH_CAM_INTRINSIC_PARAM
  };

  static std::string to_string(Cam_Intrinsic_Param in){
    assert(in < MAX_CAM_INTRINSIC_PARAM);
    const static char * intrinsicParamStrings[] = {
      "CX",
      "CY",
      "FX",
      "FY",
      "FS",
      "K1",
      "K2",
      "K3",
      "K4",
      "K5",
      "K6",
      "P1",
      "P2",
      "S1",
      "S2",
      "S3",
      "S4",
      "T1",
      "T2",
      "LENS_DISTORTION_MODEL"
    };
    return intrinsicParamStrings[in];
  };

  static Cam_Intrinsic_Param string_to_cam_intrinsic_param(std::string & in){
    // convert the string to uppercase
    std::transform(in.begin(), in.end(),in.begin(),::toupper);
    for(int_t i=0;i<MAX_CAM_INTRINSIC_PARAM;++i){
      if(to_string(static_cast<Cam_Intrinsic_Param>(i))==in) return static_cast<Cam_Intrinsic_Param>(i);
    }
    std::cout << "Error: Cam_Intrinsic_Param " << in << " does not exist." << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"");
    return NO_SUCH_CAM_INTRINSIC_PARAM; // prevent no return errors
  }

  /// \enum Cam_Extrinsic_Param valid extrinsic parameters index values
  enum Cam_Extrinsic_Param {
    ALPHA = 0,
    BETA,
    GAMMA,
    TX,
    TY,
    TZ,
    //DON"T ADD ANY BELOW MAX
    MAX_CAM_EXTRINSIC_PARAM,
    NO_SUCH_CAM_EXTRINSIC_PARAM
  };

  static std::string to_string(Cam_Extrinsic_Param in){
    assert(in < MAX_CAM_EXTRINSIC_PARAM);
    const static char * extrinsicParamStrings[] = {
      "ALPHA",
      "BETA",
      "GAMMA",
      "TX",
      "TY",
      "TZ"
    };
    return extrinsicParamStrings[in];
  };

  static Cam_Extrinsic_Param string_to_cam_extrinsic_param(std::string & in){
    // convert the string to uppercase
    std::transform(in.begin(), in.end(),in.begin(),::toupper);
    for(int_t i=0;i<MAX_CAM_EXTRINSIC_PARAM;++i){
      if(to_string(static_cast<Cam_Extrinsic_Param>(i))==in) return static_cast<Cam_Extrinsic_Param>(i);
    }
    std::cout << "Error: Cam_Extrinsic_Param " << in << " does not exist." << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"");
    return NO_SUCH_CAM_EXTRINSIC_PARAM; // prevent no return errors
  }

  /// \brief constructor with no args
  Camera():
  camera_filled_(false),
  camera_prepped_(false),
  rot_3x3_matrix_filled_(false),
  image_height_(0),
  image_width_(0),
  pixel_depth_(0),
  camera_id_(""),
  camera_lens_(""),
  camera_comments_(""){
    clear_camera();
  };

  /// \brief minimun constructor to specify only the required camera parameters
  /// \param cam_id string identifier for the camera
  /// \param intrinsics intrinsic parameters ordered by DICe_Camera::Cam_Intrinsic_Params
  /// \param image_height height of the image in pixels
  /// \param image_width width of the image in pixels
  Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics) {
    clear_camera();
    set_identifier(cam_id);
    set_intrinsics(intrinsics);
    set_image_height(image_height);
    set_image_width(image_width);
  }

  /// \brief constructor to specify the required camera parameters and extrincis parameters
  /// \param cam_id string identifier for the camera
  /// \param intrinsics intrinsic parameters ordered by DICe_Camera::Cam_Intrinsic_Params
  /// \param image_height height of the image in pixels
  /// \param image_width width of the image in pixels
  /// \param extrinsics extrinsic parameters ordered by DICe_Camera::Cam_Extrinsic_Params
  Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics) {
    clear_camera();
    set_identifier(cam_id);
    set_intrinsics(intrinsics);
    set_image_height(image_height);
    set_image_width(image_width);
    set_extrinsics(extrinsics);
  }

  /// \brief constructor for the entire camera with extrinsic parameters and rotation matrix
  /// \param cam_id string identifier for the camera
  /// \param intrinsics intrinsic parameters ordered by DICe_Camera::Cam_Intrinsic_Params
  /// \param image_height height of the image in pixels
  /// \param image_width width of the image in pixels
  /// \param extrinsics extrinsic parameters ordered by DICe_Camera::Cam_Extrinsic_Params
  /// \param rotation_3x3_matrix [R] matrix transforming from the world coordinates to the camera coordinates
  Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics,
    std::vector<std::vector<scalar_t> > & rotation_3x3_matrix) {
    clear_camera();
    set_identifier(cam_id);
    set_intrinsics(intrinsics);
    set_image_height(image_height);
    set_image_width(image_width);
    set_extrinsics(extrinsics);
    set_3x3_rotation_matrix(rotation_3x3_matrix);
  }

  /// \brief full constructor for the entire camera
  /// \param cam_id string identifier for the camera
  /// \param intrinsics intrinsic parameters ordered by DICe_Camera::Cam_Intrinsic_Params
  /// \param image_height height of the image in pixels
  /// \param image_width width of the image in pixels
  /// \param extrinsics extrinsic parameters ordered by DICe_Camera::Cam_Extrinsic_Params
  /// \param rotation_3x3_matrix [R] matrix transforming from the world coordinates to the camera coordinates
  /// \param camera_lens camera lens descripter
  /// \param camera_comments camera comments
  Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics,
    std::vector<std::vector<scalar_t> > & rotation_3x3_matrix, std::string camera_lens = "", std::string camera_comments = "",
    int_t pixel_depth=0) {
    clear_camera();
    set_identifier(cam_id);
    set_intrinsics(intrinsics);
    set_image_height(image_height);
    set_image_width(image_width);
    set_extrinsics(extrinsics);
    set_3x3_rotation_matrix(rotation_3x3_matrix);
    set_camera_lens(camera_lens);
    set_camera_comments(camera_comments);
  }

  // Pure virtual destructor
  virtual ~Camera() {}

  ///sets the camera identifier
  /// \param cam_id string descripter for the camera
  void set_identifier(std::string cam_id) {
    camera_id_ = cam_id;
    camera_filled_ = true;
  }
  ///gets the camera identifier
  /// returns the camera identifier
  std::string get_identifier() { return camera_id_; }

  ///sets the camera lens identifier
  /// \param camera_lens string descripter for the camera lens
  void set_camera_lens(std::string camera_lens) {
    camera_lens_ = camera_lens;
    camera_filled_ = true;
  }
  ///gets the camera lens identifier
  /// returns the camera lens description
  std::string get_camera_lens() { return camera_lens_; }

  ///sets the camera comments
  /// \param camera_comments comments about the camera
  void set_camera_comments(std::string camera_comments) {
    camera_comments_ = camera_comments;
    camera_filled_ = true;
  }
  ///gets the camera comments
  /// returns the camera comments
  std::string get_camera_comments() { return camera_comments_; }

  /// \brief sets the image height
  /// \param height height of the image in pixels
  void set_image_height(int_t height) {
    if (image_height_ != height) camera_prepped_ = false;
    image_height_ = height;
    camera_filled_ = true;
  }
  /// \brief gets the image height
  /// returns the height of the image in pixels
  int_t get_image_height() { return image_height_; }

  /// \brief sets the image width
  /// \param width width of the image in pixels
  void set_image_width(int_t width) {
    if (image_width_ != width) camera_prepped_ = false;
    image_width_ = width;
    camera_filled_ = true;
  }
  /// \brief gets the image width
  /// returns the width of the image in pixels
  int_t get_image_width() { return image_width_; }

  ///sets the pixel depth
  /// \param pixel_depth depth of the pixel representation typically 8 for grayscale images
  void set_pixel_depth(int_t pixel_depth) {
    pixel_depth_ = pixel_depth;
    camera_filled_ = true;
  }
  ///sets the pixel depth
  /// returns the pixel depth
  int_t get_pixel_depth() { return pixel_depth_; }

  ///sets the intrinsic camera parameter values
  /// \param intrinsics intrinsic parameters ordered by DICe_Camera::Cam_Intrinsic_Params
  void set_intrinsics(std::vector<scalar_t> & intrinsics) {
    for (int_t i = 0; i < MAX_CAM_INTRINSIC_PARAM; i++) {
      if (intrinsics_[i] != intrinsics[i]) camera_prepped_ = false;
      intrinsics_[i] = intrinsics[i];
    }
    camera_filled_ = true;
  }
  ///gets the intrinsic camera parameter values
  /// \param intrinsics intrinsic parameters ordered by DICe_Camera::Cam_Intrinsic_Params
  void get_intrinsics(std::vector<scalar_t> & intrinsics) {
    for (int_t i = 0; i < MAX_CAM_INTRINSIC_PARAM; i++)
      intrinsics[i] = intrinsics_[i];
  }

  ///sets extrinsic camera parameter values
  /// \param extrinsics extrinsic parameters ordered by DICe_Camera::Cam_Extrinsic_Params
  void set_extrinsics(std::vector<scalar_t> & extrinsics) {
    for (int_t i = 0; i < MAX_CAM_EXTRINSIC_PARAM; i++) {
      if (extrinsics_[i] != extrinsics[i]) camera_prepped_ = false;
      extrinsics_[i] = extrinsics[i];
    }
    camera_filled_ = true;
  }
  ///gets extrinsic camera parameter values
  /// \param extrinsics extrinsic parameters ordered by DICe_Camera::Cam_Extrinsic_Params
  void get_extrinsics(std::vector<scalar_t> & extrinsics) {
    for (int_t i = 0; i < MAX_CAM_EXTRINSIC_PARAM; i++)
      extrinsics[i] = extrinsics_[i];
  }

  ///sets 3x3 rotation matrix [R]
  /// \param rotation_3x3_matrix [R] matrix transforming from the world coordinates to the camera coordinates
  void set_3x3_rotation_matrix(std::vector<std::vector<scalar_t> > & rotation_3x3_matrix) {
    for (int_t i = 0; i < 3; i++) {
      for (int_t j = 0; j < 3; j++) {
        if (rotation_3x3_matrix_[i][j] != rotation_3x3_matrix[i][j]) camera_prepped_ = false;
        rotation_3x3_matrix_[i][j] = rotation_3x3_matrix[i][j];
      }
    }
    camera_filled_ = true;
    rot_3x3_matrix_filled_ = true;
  }
  ///gets 3x3 rotation matrix [R]
  /// \param rotation_3x3_matrix [R] matrix transforming from the world coordinates to the camera coordinates
  void get_3x3_rotation_matrix(std::vector<std::vector<scalar_t> > & rotation_3x3_matrix) {
    for (int_t i = 0; i < 3; i++) {
      for (int_t j = 0; j < 3; j++) {
        rotation_3x3_matrix[i][j] = rotation_3x3_matrix_[i][j];
      }
    }
  }

  ///gets the camera to world transformation matrix
  /// \param cam_world_trans 4x4 [R|T] transformation matrix {cam(x,y,z)}=[R|T]{world(x,y,z)}
  void get_cam_world_trans_matrix(std::vector<std::vector<scalar_t> > & cam_world_trans) {
    for (int_t i = 0; i < 4; i++) {
      for (int_t j = 0; j < 4; j++) {
        cam_world_trans[i][j] = cam_world_trans_[i][j];
      }
    }
  }

  ///gets the world to camera transformation matrix
  /// \param cam_world_trans 4x4 [R|T] transformation matrix {world(x,y,z)}=[R|T]{cam(x,y,z)}
  void get_world_cam_trans_matrix(std::vector<std::vector<scalar_t> > & world_cam_trans) {
    for (int_t i = 0; i < 4; i++) {
      for (int_t j = 0; j < 4; j++) {
        world_cam_trans[i][j] = world_cam_trans_[i][j];
      }
    }
  }

  /// does the camera have enough values to be valid
  /// \paqam msg return message with reason for invalid
  bool camera_valid(std::string & msg) {
    return check_valid_(msg);
  }

  ///have any of the parameters/fields been filled
  bool camera_filled() {
    return camera_filled_;
  }

  ///has the 3x3 rotation matrix been filled
  bool camera_has_3x3_rotation() {
    return rot_3x3_matrix_filled_;
  }

  ///prepares the tranfomation matricies
  bool prep_camera();

  ///has the camera been prepped. cameras must be prepped to establish the transformation matricies
  bool camera_prepped(){
    return camera_prepped_;
  }

  /// \brief clear the parameter values for the camera
  /// \param
  void clear_camera();


  /// convert sensor locations to image locations: applies lens distortion scales for fx,fy and
  /// converts to image coordiates with cx, cy
  /// \param sen_x projected x sensor location
  /// \param sen_y projected y sensor location
  /// \param image_x x location after applied lens distortion
  /// \param image_y y location after applied lens distortion
  void sensor_to_image(
    std::vector<scalar_t> & sen_x,
    std::vector<scalar_t> & sen_y,
    std::vector<scalar_t> & image_x,
    std::vector<scalar_t> & image_y);
  /// convert sensor locations to image locations: applies lens distortion scales for fx,fy and
  /// converts to image coordiates with cx, cy with first partials
  /// \param sen_x projected x sensor location
  /// \param sen_y projected y sensor location
  /// \param image_x x location after applied lens distortion
  /// \param image_y y location after applied lens distortion
  /// \param sen_dx incoming location partials
  /// \param sen_dy incoming location partials
  /// \param image_dx outgoing location partials lens distortion fixed value
  /// \param image_dy outgoing location partials lens distortion fixed value
  void sensor_to_image(
    std::vector<scalar_t> & sen_x,
    std::vector<scalar_t> & sen_y,
    std::vector<scalar_t> & image_x,
    std::vector<scalar_t> & image_y,
    std::vector<std::vector<scalar_t> > & sen_dx,
    std::vector<std::vector<scalar_t> > & sen_dy,
    std::vector<std::vector<scalar_t> > & image_dx,
    std::vector<std::vector<scalar_t> > & image_dy);

  ///converts image coordinates to sensor coordinates (lens distortion^-1, fx,fy,cx,cy)
  /// \param image_x x location after applied lens distortion
  /// \param image_y y location after applied lens distortion
  /// \param sen_x projected x sensor location
  /// \param sen_y projected y sensor location
  /// \param integer_locs if all image points are integers setting this flag avoids interpolation overhead
  void image_to_sensor(
    std::vector<scalar_t> & image_x,
    std::vector<scalar_t> & image_y,
    std::vector<scalar_t> & sen_x,
    std::vector<scalar_t> & sen_y,
    bool integer_locs = true);

  ///projects sensor coordinates onto a plane in space described by zp,theta,phi
  /// \param sen_x x sensor location
  /// \param sen_y y sensor location
  /// \param cam_x projected x location in cam x,y,z space
  /// \param cam_y projected y location in cam x,y,z space
  /// \param cam_z projected z location in cam x,y,z space
  /// \param params projection parameters describing the plane in space (ZP, THETA, PHI)
  void sensor_to_cam(
    std::vector<scalar_t> & sen_x,
    std::vector<scalar_t> & sen_y,
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z,
    std::vector<scalar_t> & params);

  ///projects sensor coordinates onto a plane in space described by zp,theta,phi overloaded for first partials
  /// \param sen_x x sensor location
  /// \param sen_y y sensor location
  /// \param cam_x projected x location in cam x,y,z space
  /// \param cam_y projected y location in cam x,y,z space
  /// \param cam_z projected z location in cam x,y,z space
  /// \param params projection parameters describing the plane in space (ZP, THETA, PHI)
  /// \param cam_dx x location derivitives in cam x,y,z space partials wrt (ZP, THETA, PHI)
  /// \param cam_dy y locatoin derivitives in cam x,y,z space partials wrt (ZP, THETA, PHI)
  /// \param cam_dz z location derivitives in cam x,y,z space partials wrt (ZP, THETA, PHI)
  void sensor_to_cam(
    std::vector<scalar_t> & sen_x,
    std::vector<scalar_t> & sen_y,
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z,
    std::vector<scalar_t> & params,
    std::vector<std::vector<scalar_t> > & cam_dx,
    std::vector<std::vector<scalar_t> > & cam_dy,
    std::vector<std::vector<scalar_t> > & cam_dz);

  ///projects camera x,y,z locations back onto the sensor
  /// \param cam_x x location in cam x,y,z space
  /// \param cam_y y location in cam x,y,z space
  /// \param cam_z z location in cam x,y,z space
  /// \param sen_x projected x sensor location
  /// \param sen_y projected y sensor location
  void cam_to_sensor(
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z,
    std::vector<scalar_t> & sen_x,
    std::vector<scalar_t> & sen_y);

  ///projects camera x,y,z locations back onto the sensor overloaded with input and output first partials
  /// \param cam_x x location in cam x,y,z space
  /// \param cam_y y location in cam x,y,z space
  /// \param cam_z z location in cam x,y,z space
  /// \param sen_x projected x sensor location
  /// \param sen_y projected y sensor location
  /// \param cam_dx first partials of the cam locations from previous projections
  /// \param cam_dy first partials of the cam locations from previous projections
  /// \param cam_dz first partials of the cam locations from previous projections
  /// \param sen_dx first partials of the projected x sensor location
  /// \param sen_dy first partials of the projected y sensor location
  void cam_to_sensor(
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z,
    std::vector<scalar_t> & sen_x,
    std::vector<scalar_t> & sen_y,
    std::vector<std::vector<scalar_t> > & cam_dx,
    std::vector<std::vector<scalar_t> > & cam_dy,
    std::vector<std::vector<scalar_t> > & cam_dz,
    std::vector<std::vector<scalar_t> > & sen_dx,
    std::vector<std::vector<scalar_t> > & sen_dy);

  ///converts the camera x,y,z cooldinates to a world x,y,z coordinate system
  /// \param cam_x x location in cam x,y,z space
  /// \param cam_y y location in cam x,y,z space
  /// \param cam_z z location in cam x,y,z space
  /// \param wrld_x x location in the world x,y,z space
  /// \param wrld_y y location in the world x,y,z space
  /// \param wrld_z z location in the world x,y,z space
  void cam_to_world(
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z,
    std::vector<scalar_t> & wrld_x,
    std::vector<scalar_t> & wrld_y,
    std::vector<scalar_t> & wrld_z) {
    rot_trans_transform_(cam_world_trans_, cam_x, cam_y, cam_z, wrld_x, wrld_y, wrld_z);
  }

  ///converts the camera x,y,z cooldinates to a world x,y,z coordinate system overloaded with first partials
  /// \param cam_x x location in cam x,y,z space
  /// \param cam_y y location in cam x,y,z space
  /// \param cam_z z location in cam x,y,z space
  /// \param wrld_x x location in the world x,y,z space
  /// \param wrld_y y location in the world x,y,z space
  /// \param wrld_z z location in the world x,y,z space
  /// \param cam_dx first partials of the x cam locations from previous projections
  /// \param cam_dy first partials of the y cam locations from previous projections
  /// \param cam_dz first partials of the z cam locations from previous projections
  /// \param wrld_dx first partials of the x world locations
  /// \param wrld_dy first partials of the y world locations
  /// \param wrld_dz first partials of the z world locations
  void cam_to_world(
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z,
    std::vector<scalar_t> & wrld_x,
    std::vector<scalar_t> & wrld_y,
    std::vector<scalar_t> & wrld_z,
    std::vector<std::vector<scalar_t> > & cam_dx,
    std::vector<std::vector<scalar_t> > & cam_dy,
    std::vector<std::vector<scalar_t> > & cam_dz,
    std::vector<std::vector<scalar_t> > & wrld_dx,
    std::vector<std::vector<scalar_t> > & wrld_dy,
    std::vector<std::vector<scalar_t> > & wrld_dz) {
    rot_trans_transform_(cam_world_trans_, wrld_x, wrld_y, wrld_z, cam_x, cam_y, cam_z,
      cam_dx, cam_dy, cam_dz, wrld_dx, wrld_dy, wrld_dz);
  }

  ///converts world x,y,z coordinates to camera x,y,z coordinates
  /// \param wrld_x x location in the world x,y,z space
  /// \param wrld_y y location in the world x,y,z space
  /// \param wrld_z z location in the world x,y,z space
  /// \param cam_x x location in cam x,y,z space
  /// \param cam_y y location in cam x,y,z space
  /// \param cam_z z location in cam x,y,z space
  void world_to_cam(
    std::vector<scalar_t> & wrld_x,
    std::vector<scalar_t> & wrld_y,
    std::vector<scalar_t> & wrld_z,
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z) {
    rot_trans_transform_(world_cam_trans_, wrld_x, wrld_y, wrld_z, cam_x, cam_y, cam_z);
  }

  ///converts world x,y,z coordinates to camera x,y,z coordinates overloaded with first partials
  /// \param wrld_x x location in the world x,y,z space
  /// \param wrld_y y location in the world x,y,z space
  /// \param wrld_z z location in the world x,y,z space
  /// \param cam_x x location in cam x,y,z space
  /// \param cam_y y location in cam x,y,z space
  /// \param cam_z z location in cam x,y,z space
  /// \param wrld_dx first partials of the x world locations from previous transformations
  /// \param wrld_dy first partials of the x world locations from previous transformations
  /// \param wrld_dz first partials of the x world locations from previous transformations
  /// \param cam_dx first partials of the x cam locations
  /// \param cam_dy first partials of the y cam locations
  /// \param cam_dz first partials of the z cam locations
  void world_to_cam(
    std::vector<scalar_t> & wrld_x,
    std::vector<scalar_t> & wrld_y,
    std::vector<scalar_t> & wrld_z,
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z,
    std::vector<std::vector<scalar_t> > & wrld_dx,
    std::vector<std::vector<scalar_t> > & wrld_dy,
    std::vector<std::vector<scalar_t> > & wrld_dz,
    std::vector<std::vector<scalar_t> > & cam_dx,
    std::vector<std::vector<scalar_t> > & cam_dy,
    std::vector<std::vector<scalar_t> > & cam_dz) {
    rot_trans_transform_(world_cam_trans_, wrld_x, wrld_y, wrld_z, cam_x, cam_y, cam_z,
      wrld_dx, wrld_dy, wrld_dz, cam_dx, cam_dy, cam_dz);
  }


private:
  ///generic translation/rotation transform
  /// \param in_x input x location
  /// \param in_y input y location
  /// \param in_z input z location
  /// \param out_x transformed x location
  /// \param out_y transformed y location
  /// \param out_z transformed z location
  void rot_trans_transform_(
    std::vector<std::vector<scalar_t> > & RT_matrix,
    std::vector<scalar_t> & in_x,
    std::vector<scalar_t> & in_y,
    std::vector<scalar_t> & in_z,
    std::vector<scalar_t> & out_x,
    std::vector<scalar_t> & out_y,
    std::vector<scalar_t> & out_z);

  ///generic translation/rotation transform
  /// \param in_x input x location
  /// \param in_y input y location
  /// \param in_z input z location
  /// \param out_x transformed x location
  /// \param out_y transformed y location
  /// \param out_z transformed z location
  /// \param in_dx derivitives from previous transformations
  /// \param in_dy derivitives from previous transformations
  /// \param in_dz derivitives from previous transformations
  /// \param out_dx derivitives modifiec by the transform
  /// \param out_dy derivitives modifiec by the transform
  /// \param out_dz derivitives modifiec by the transform
  void rot_trans_transform_(
    std::vector<std::vector<scalar_t> > & RT_matrix,
    std::vector<scalar_t> & in_x,
    std::vector<scalar_t> & in_y,
    std::vector<scalar_t> & in_z,
    std::vector<scalar_t> & out_x,
    std::vector<scalar_t> & out_y,
    std::vector<scalar_t> & out_z,
    std::vector<std::vector<scalar_t> > & in_dx,
    std::vector<std::vector<scalar_t> > & in_dy,
    std::vector<std::vector<scalar_t> > & in_dz,
    std::vector<std::vector<scalar_t> > & out_dx,
    std::vector<std::vector<scalar_t> > & out_dy,
    std::vector<std::vector<scalar_t> > & out_dz);

  ///routine to check for a valid camera
  bool check_valid_(std::string & msg);
  ///creates the image values for the inverse lens distortion
  bool prep_lens_distortion_();
  ///creates the rotation/translation matricies and inverses
  bool prep_transforms_();

  ///has the camera been filled with any parameters
  bool camera_filled_;
  ///has the camera prep been run
  bool camera_prepped_;
  ///did the user supply a rotation matrix
  bool rot_3x3_matrix_filled_;

  // 18 member array of camera intrinsics
  // first index is the camera id, second index is:
  //   openCV_DIS - (cx cy fx fy k1 k2 p1 p2 [k3] [k4 k5 k6] [s1 s2 s3 s4] [tx ty])   4,5,8,12 or 14 distortion coef
  //		 Vic3D_DIS - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r1 k2r2 k3r3)
  //	    generic1 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r1 k2r2 k3r3)
  //	    generic2 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r2 k2r4 k3r6)
  //	    generic3 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r3 k2r5 k3r7)
  std::vector<scalar_t> intrinsics_;

  // 6 member array of camera extrinsics
  // index is alpha beta gamma tx ty tz (the Euler angles + translations)
  std::vector<scalar_t> extrinsics_;

  // 3x3 rotation matrix
  std::vector<std::vector<scalar_t> > rotation_3x3_matrix_;

  // Inverse lense distortion values for each pixel in an image
  std::vector<scalar_t> inv_lens_dis_x_;
  std::vector<scalar_t> inv_lens_dis_y_;

  // transformation coefficients
  std::vector<std::vector<scalar_t> > cam_world_trans_;
  std::vector<std::vector<scalar_t> > world_cam_trans_;

  //Other camera parameters
  int_t image_height_;
  int_t image_width_;
  int_t pixel_depth_;
  std::string camera_id_;
  std::string camera_lens_;
  std::string camera_comments_;
};

}// End DICe Namespace

#endif

