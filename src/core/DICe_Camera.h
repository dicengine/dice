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
#include <DICe_Matrix.h>

#include <cassert>
#include <vector>

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
      "T2"
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

  /// transform coordinates according to the rbm parameters and replace input values
  /// the rigid_body_paramers should be ordered according to three euler angles and three translations (x,y,z)
  /// rigid_body_params version
  static void transform_coordinates_in_place(std::vector<scalar_t> & x,
    std::vector<scalar_t> & y,
    std::vector<scalar_t> & z,
    const std::vector<scalar_t> & rigid_body_params);
  /// rotation matrix and translations version
  static void transform_coordinates_in_place(std::vector<scalar_t> & x,
    std::vector<scalar_t> & y,
    std::vector<scalar_t> & z,
    const Matrix<scalar_t,3> & R,
    const scalar_t & tx,
    const scalar_t & ty,
    const scalar_t & tz);
  /// transformation matrix version
  static void transform_coordinates_in_place(std::vector<scalar_t> & x,
    std::vector<scalar_t> & y,
    std::vector<scalar_t> & z,
    const Matrix<scalar_t,4> & T);
  /// helper struct for camera initialization (cleans up of the constructors for camera class so it only needs one)
  struct DICE_LIB_DLL_EXPORT Camera_Info{
    // array of camera intrinsics (the distortion coeffs are not stored in the intrinsic array)
    //   openCV_DIS - (cx cy fx fy k1 k2 p1 p2 [k3] [k4 k5 k6] [s1 s2 s3 s4] [tx ty])   4,5,8,12 or 14 distortion coef
    //     Vic3D_DIS - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r1 k2r2 k3r3)
    //      generic1 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r1 k2r2 k3r3)
    //      generic2 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r2 k2r4 k3r6)
    //      generic3 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r3 k2r5 k3r7)
    std::vector<scalar_t> intrinsics_;
    // extrinsic rotation_matrix [R] matrix transforming from the world coordinates to the camera coordinates
    // this matrix is the identity matrix if the world coordinates are the camera coordinates
    //
    // a camera always stores its extrinsic information as a rotation matrix and three translations
    // if the euler angles are specified, they are converted to a rotation matrix, the eulers are not stored
    Matrix<scalar_t,3> rotation_matrix_;
    // extrinsic translations x y and z (these are optional and zero if the world coordinates are the camera coordinates)
    scalar_t tx_;
    scalar_t ty_;
    scalar_t tz_;
    // lens distortion model
    Lens_Distortion_Model lens_distortion_model_;
    // image dimensions in pixels
    int_t image_width_;
    int_t image_height_;
    // optional
    int_t pixel_depth_;
    // camera information
    // identifier
    std::string id_;
    // lens type
    std::string lens_;
    // comments about the camera
    std::string comments_;
    // constructor
    Camera_Info():
      intrinsics_(MAX_CAM_INTRINSIC_PARAM,0),
      rotation_matrix_(Matrix<scalar_t,3>::identity()),
      tx_(0),
      ty_(0),
      tz_(0),
      lens_distortion_model_(NO_LENS_DISTORTION),
      image_width_(-1),
      image_height_(-1),
      pixel_depth_(-1){};
    // convert euler angles to rotation matrix
    static Matrix<scalar_t,3> eulers_to_rotation_matrix(const scalar_t & alpha,
      const scalar_t & beta,
      const scalar_t & gamma);
    // convert euler angles to rotation matrix
    static void rotation_matrix_to_eulers(const Matrix<scalar_t,3> & R,
      scalar_t & alpha,
      scalar_t & beta,
      scalar_t & gamma);
    // convert euler angles to rotation matrix partial derivatives
    static void eulers_to_rotation_matrix_partials(const scalar_t & alpha,
      const scalar_t & beta,
      const scalar_t & gamma,
      Matrix<scalar_t,3,4> & R_dx,
      Matrix<scalar_t,3,4> & R_dy,
      Matrix<scalar_t,3,4> & R_dz);
    // set the rotation matrix using the euler angles alpha, beta, and gamma
    void set_rotation_matrix(const scalar_t & alpha,
      const scalar_t & beta,
      const scalar_t & gamma);
    // set the intrinsic vector from an already constructed vector
    void set_intrinsics(const std::vector<scalar_t> & in){
      TEUCHOS_TEST_FOR_EXCEPTION(in.size()!=MAX_CAM_INTRINSIC_PARAM,std::runtime_error,"");
      for(size_t i=0;i<intrinsics_.size();++i) intrinsics_[i] = in[i];
    }
    // helper function to set the translations
    void set_extrinsic_translations(const scalar_t & tx, const scalar_t & ty, const scalar_t & tz){
      tx_ = tx; ty_ = ty; tz_ = tz;
    }
    // return true if the camera has all the parameters set needed to be used for analysis
    bool is_valid();
    // same as is_valid() only throws an exception if not valid
    void check_valid()const;
    // clears all the parameter values in a camera
    void clear();
    // difference with another camera info
    scalar_t diff(const Camera_Info & rhs) const;
    // comparison operator
    DICE_LIB_DLL_EXPORT
    friend bool operator==(const Camera_Info & lhs,const Camera_Info & rhs);
    // comparison operator
    DICE_LIB_DLL_EXPORT
    friend bool operator!=(const Camera_Info & lhs,const Camera_Info & rhs){
      return !(lhs==rhs);
    }
    // overaload the ostream operator for a camera info class
    DICE_LIB_DLL_EXPORT
    friend std::ostream & operator<<(std::ostream & os, const Camera_Info & info);
  };

  /// \brief constructor
  Camera(const Camera_Info & camera_info):
  camera_info_(camera_info){
    // check for valid camera_info
    camera_info_.check_valid();
    initialize();
  };

  /// Pure virtual destructor
  virtual ~Camera() {}

  ///sets the camera identifier
  /// \param cam_id string descripter for the camera
  void set_id(const std::string & cam_id) {
    camera_info_.id_ = cam_id;
  }
  ///gets the camera identifier
  /// returns the camera identifier
  std::string id() const { return camera_info_.id_; }

  ///sets the camera lens identifier
  /// \param camera_lens string descripter for the camera lens
  void set_lens(const std::string & camera_lens) {
    camera_info_.lens_ = camera_lens;
  }
  ///gets the camera lens identifier
  /// returns the camera lens description
  std::string lens() const { return camera_info_.lens_; }

  ///sets the camera comments
  /// \param camera_comments comments about the camera
  void set_comments(const std::string & camera_comments) {
    camera_info_.comments_ = camera_comments;
  }
  ///gets the camera comments
  /// returns the camera comments
  std::string comments() const { return camera_info_.comments_; }

  /// \brief gets the image height
  /// returns the height of the image in pixels
  int_t image_height() const { return camera_info_.image_height_; }

  /// \brief gets the image width
  /// returns the width of the image in pixels
  int_t image_width() const { return camera_info_.image_width_; }

  ///sets the pixel depth
  /// returns the pixel depth
  int_t pixel_depth() const { return camera_info_.pixel_depth_; }

  Lens_Distortion_Model lens_distortion_model()const{
    return camera_info_.lens_distortion_model_;
  }

  /// returns a 4x4 matrix with the upper left block the rotation matrix,
  /// the right side column vector as the translations and the lower right
  /// entry as 1.0
  Matrix<scalar_t,4> transformation_matrix()const;

  /// returns the facet parameters from R and T (if they describe the pose estimation, likely from 2d calibration)
  /// in 3d, R and T describe a coordinate transformation, not the pose estimation of the calibration plate so this is
  /// only a useful method for 2d calibration
  std::vector<scalar_t> get_facet_params();

  ///gets the intrinsic camera parameter values
  std::vector<scalar_t> * intrinsics() { return &camera_info_.intrinsics_;}

  /// get the x extrinsic translation
  scalar_t tx()const {return camera_info_.tx_;}
  /// get the y extrinsic translation
  scalar_t ty()const {return camera_info_.ty_;}
  /// get the z extrinsic translation
  scalar_t tz()const {return camera_info_.tz_;}

  ///gets 3x3 rotation matrix [R]
  /// \param rotation_3x3_matrix [R] matrix transforming from the world coordinates to the camera coordinates
  const Matrix<scalar_t,3> * rotation_matrix() const { return &camera_info_.rotation_matrix_; }

  ///gets the camera to world transformation matrix
  /// cam_world_trans 4x4 [R|T] transformation matrix {cam(x,y,z)}=[R|T]{world(x,y,z)}
  const Matrix<scalar_t,4> * cam_world_trans_matrix() const { return &cam_world_trans_; }

  ///gets the world to camera transformation matrix
  /// cam_world_trans 4x4 [R|T] transformation matrix {world(x,y,z)}=[R|T]{cam(x,y,z)}
  const Matrix<scalar_t,4> * world_cam_trans_matrix() const { return &world_cam_trans_; }

  ///prepares the tranfomation matricies
  void initialize();

  /// comparison operator
  DICE_LIB_DLL_EXPORT
  friend bool operator==(const Camera & lhs,const Camera & rhs){
    if(lhs.cam_world_trans_!=rhs.cam_world_trans_){
      DEBUG_MSG("camera cam_world_trans matrices do not match");
      return false;
    }
    if(lhs.world_cam_trans_!=rhs.world_cam_trans_){
      DEBUG_MSG("camera world_cam_trans matrices do not match");
      return false;
    }
    if(lhs.camera_info_!=rhs.camera_info_){
      return false;
    }
    return true;
  }

  /// comparison operator
  DICE_LIB_DLL_EXPORT
  friend bool operator!=(const Camera & lhs,const Camera & rhs){
    return !(lhs==rhs);
  }

  /// overaload the ostream operator for a camera class
  DICE_LIB_DLL_EXPORT
  friend std::ostream & operator<<(std::ostream & os, const Camera & camera);

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
    const std::vector<scalar_t> & sen_x,
    const std::vector<scalar_t> & sen_y,
    std::vector<scalar_t> & image_x,
    std::vector<scalar_t> & image_y,
    const std::vector<std::vector<scalar_t> > & sen_dx,
    const std::vector<std::vector<scalar_t> > & sen_dy,
    std::vector<std::vector<scalar_t> > & image_dx,
    std::vector<std::vector<scalar_t> > & image_dy);

  /// same as above without derivatives
  void sensor_to_image(
    const std::vector<scalar_t> & sen_x,
    const std::vector<scalar_t> & sen_y,
    std::vector<scalar_t> & image_x,
    std::vector<scalar_t> & image_y);

  ///converts image coordinates to sensor coordinates (lens distortion^-1, fx,fy,cx,cy)
  /// \param image_x x location after applied lens distortion
  /// \param image_y y location after applied lens distortion
  /// \param sen_x projected x sensor location
  /// \param sen_y projected y sensor location
  /// \param integer_locs if all image points are integers setting this flag avoids interpolation overhead
  void image_to_sensor(
    const std::vector<scalar_t> & image_x,
    const std::vector<scalar_t> & image_y,
    std::vector<scalar_t> & sen_x,
    std::vector<scalar_t> & sen_y,
    const bool integer_locs = true);

  /// helper function to convert image coordinates to world coordinates
  /// \param image_x x location after applied lens distortion
  /// \param image_y y location after applied lens distortion
  /// \param rigid_body_params projection parameters that describe a rotation and traslation in space of a planar facet
  /// \param world_x output world x coordinate
  /// \param world_y output world y coordinate
  /// \param world_z output world z coordinate
  void image_to_world(const std::vector<scalar_t> & image_x,
    const std::vector<scalar_t> & image_y,
    const std::vector<scalar_t> & rigid_body_params,
    std::vector<scalar_t> & world_x,
    std::vector<scalar_t> & world_y,
    std::vector<scalar_t> & world_z);
  void image_to_world(const std::vector<scalar_t> & image_x,
    const std::vector<scalar_t> & image_y,
    std::vector<scalar_t> & world_x,
    std::vector<scalar_t> & world_y,
    std::vector<scalar_t> & world_z){
    std::vector<scalar_t> dummy(6,0.0);
    image_to_world(image_x,image_y,dummy,world_x,world_y,world_z);
  }

  ///projects sensor coordinates onto a plane in space described by zp,theta,phi overloaded for first partials
  /// \param sen_x x sensor location
  /// \param sen_y y sensor location
  /// \param cam_x projected x location in cam x,y,z space
  /// \param cam_y projected y location in cam x,y,z space
  /// \param cam_z projected z location in cam x,y,z space
  /// \param facet_params projection parameters describing the plane in space (ZP, THETA, PHI)
  /// \param cam_dx x location derivitives in cam x,y,z space partials wrt (ZP, THETA, PHI)
  /// \param cam_dy y locatoin derivitives in cam x,y,z space partials wrt (ZP, THETA, PHI)
  /// \param cam_dz z location derivitives in cam x,y,z space partials wrt (ZP, THETA, PHI)
  void sensor_to_cam(
    const std::vector<scalar_t> & sen_x,
    const std::vector<scalar_t> & sen_y,
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z,
    const std::vector<scalar_t> & facet_params,
    std::vector<std::vector<scalar_t> > & cam_dx,
    std::vector<std::vector<scalar_t> > & cam_dy,
    std::vector<std::vector<scalar_t> > & cam_dz);

  /// same as above with no derivatives
  void sensor_to_cam(
    const std::vector<scalar_t> & sen_x,
    const std::vector<scalar_t> & sen_y,
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z,
    const std::vector<scalar_t> & facet_params){
    std::vector<std::vector<scalar_t> > dummy_vec;
    sensor_to_cam(sen_x,sen_y,cam_x,cam_y,cam_z,
        facet_params,dummy_vec,dummy_vec,dummy_vec);
  }

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
    const std::vector<scalar_t> & cam_x,
    const std::vector<scalar_t> & cam_y,
    const std::vector<scalar_t> & cam_z,
    std::vector<scalar_t> & sen_x,
    std::vector<scalar_t> & sen_y,
    const std::vector<std::vector<scalar_t> > & cam_dx,
    const std::vector<std::vector<scalar_t> > & cam_dy,
    const std::vector<std::vector<scalar_t> > & cam_dz,
    std::vector<std::vector<scalar_t> > & sen_dx,
    std::vector<std::vector<scalar_t> > & sen_dy);

  /// same as above without derivatives
  void cam_to_sensor(
    const std::vector<scalar_t> & cam_x,
    const std::vector<scalar_t> & cam_y,
    const std::vector<scalar_t> & cam_z,
    std::vector<scalar_t> & sen_x,
    std::vector<scalar_t> & sen_y){
    std::vector<std::vector<scalar_t> > dummy_vec;
    cam_to_sensor(cam_x,cam_y,cam_z,sen_x,sen_y,
      dummy_vec,dummy_vec,dummy_vec,dummy_vec,dummy_vec);
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
    const std::vector<scalar_t> & cam_x,
    const std::vector<scalar_t> & cam_y,
    const std::vector<scalar_t> & cam_z,
    std::vector<scalar_t> & wrld_x,
    std::vector<scalar_t> & wrld_y,
    std::vector<scalar_t> & wrld_z,
    std::vector<std::vector<scalar_t> > & cam_dx,
    std::vector<std::vector<scalar_t> > & cam_dy,
    std::vector<std::vector<scalar_t> > & cam_dz,
    std::vector<std::vector<scalar_t> > & wrld_dx,
    std::vector<std::vector<scalar_t> > & wrld_dy,
    std::vector<std::vector<scalar_t> > & wrld_dz) {
    // make sure this method isn't being called with regards to the rigid body motion parameters (which have 6)
    // and only allow the projection parameters, which have 3
    TEUCHOS_TEST_FOR_EXCEPTION(cam_dx.size()!=3,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(wrld_dx.size()!=3,std::runtime_error,"");
    rot_trans_transform(cam_world_trans_, cam_x, cam_y, cam_z, wrld_x, wrld_y, wrld_z,
      cam_dx, cam_dy, cam_dz, wrld_dx, wrld_dy, wrld_dz);
  }

  /// same as above without derivatives
  void cam_to_world(
    const std::vector<scalar_t> & cam_x,
    const std::vector<scalar_t> & cam_y,
    const std::vector<scalar_t> & cam_z,
    std::vector<scalar_t> & wrld_x,
    std::vector<scalar_t> & wrld_y,
    std::vector<scalar_t> & wrld_z) {
    rot_trans_transform(cam_world_trans_, cam_x, cam_y, cam_z, wrld_x, wrld_y, wrld_z);
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
    const std::vector<scalar_t> & wrld_x,
    const std::vector<scalar_t> & wrld_y,
    const std::vector<scalar_t> & wrld_z,
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z,
    const std::vector<std::vector<scalar_t> > & wrld_dx,
    const std::vector<std::vector<scalar_t> > & wrld_dy,
    const std::vector<std::vector<scalar_t> > & wrld_dz,
    std::vector<std::vector<scalar_t> > & cam_dx,
    std::vector<std::vector<scalar_t> > & cam_dy,
    std::vector<std::vector<scalar_t> > & cam_dz) {
    rot_trans_transform(world_cam_trans_, wrld_x, wrld_y, wrld_z, cam_x, cam_y, cam_z,
      wrld_dx, wrld_dy, wrld_dz, cam_dx, cam_dy, cam_dz);
  }

  /// same as above without derivatives
  void world_to_cam(
    const std::vector<scalar_t> & wrld_x,
    const std::vector<scalar_t> & wrld_y,
    const std::vector<scalar_t> & wrld_z,
    std::vector<scalar_t> & cam_x,
    std::vector<scalar_t> & cam_y,
    std::vector<scalar_t> & cam_z) {
    rot_trans_transform(world_cam_trans_, wrld_x, wrld_y, wrld_z, cam_x, cam_y, cam_z);
  }

  /// compare the intrinsics and extrinsic values for a camera
  scalar_t diff(const Camera & rhs) const{
    return camera_info_.diff(rhs.camera_info_);
  }

private:

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
  void rot_trans_transform(
    const Matrix<scalar_t,4> & RT_matrix,
    const std::vector<scalar_t> & in_x,
    const std::vector<scalar_t> & in_y,
    const std::vector<scalar_t> & in_z,
    std::vector<scalar_t> & out_x,
    std::vector<scalar_t> & out_y,
    std::vector<scalar_t> & out_z,
    const std::vector<std::vector<scalar_t> > & in_dx,
    const std::vector<std::vector<scalar_t> > & in_dy,
    const std::vector<std::vector<scalar_t> > & in_dz,
    std::vector<std::vector<scalar_t> > & out_dx,
    std::vector<std::vector<scalar_t> > & out_dy,
    std::vector<std::vector<scalar_t> > & out_dz);

  /// same as above with no partial derivatives
  void rot_trans_transform(
    const Matrix<scalar_t,4> & RT_matrix,
    const std::vector<scalar_t> & in_x,
    const std::vector<scalar_t> & in_y,
    const std::vector<scalar_t> & in_z,
    std::vector<scalar_t> & out_x,
    std::vector<scalar_t> & out_y,
    std::vector<scalar_t> & out_z){
    std::vector<std::vector<scalar_t> > dummy_vec;
    rot_trans_transform(RT_matrix,in_x,in_y,in_z,
        out_x,out_y,out_z,
        dummy_vec,dummy_vec,dummy_vec,dummy_vec,dummy_vec,dummy_vec);
  }

  ///creates the image values for the inverse lens distortion
  void prep_lens_distortion();

  ///creates the rotation/translation matricies and inverses
  void prep_transforms();

  /// struct that holds all the initialization information
  Camera_Info camera_info_;

  // Inverse lense distortion values for each pixel in an image
  std::vector<scalar_t> inv_lens_dis_x_;
  std::vector<scalar_t> inv_lens_dis_y_;

  // transformation coefficients
  Matrix<scalar_t,4> cam_world_trans_;
  Matrix<scalar_t,4> world_cam_trans_;

  scalar_t zero_ish_;
};

}// End DICe Namespace

#endif

