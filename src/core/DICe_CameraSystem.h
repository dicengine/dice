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

#include <Teuchos_SerialDenseMatrix.hpp>

#include <cassert>
#include <vector>

namespace DICe {

/// \class DICe::Camera_System
/// \brief A class for camera calibration parameters and computing projection transformations
/// notes: there is only one source camera per camera system, this determines how the coordinate systems
/// are organized. All other cameras are target cameras.
///
class DICE_LIB_DLL_EXPORT
Camera_System {
public:

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

  static std::string to_string(System_Type_3D in){
    assert(in < MAX_SYSTEM_TYPE_3D);
    const static char * systemTypeStrings[] = {
      "UNKNOWN_SYSTEM",
      "GENERIC_SYSTEM",
      "OPENCV",
      "VIC3D",
      "DICE"
    };
    return systemTypeStrings[in];
  };

  static System_Type_3D string_to_system_type_3d(std::string & in){
    // convert the string to uppercase
    std::transform(in.begin(), in.end(),in.begin(),::toupper);
    for(int_t i=0;i<MAX_SYSTEM_TYPE_3D;++i){
      if(to_string(static_cast<System_Type_3D>(i))==in) return static_cast<System_Type_3D>(i);
    }
    std::cout << "Error: System_Type_3D " << in << " does not exist." << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"");
    return NO_SUCH_SYSTEM_TYPE_3D; // prevent no return errors
  }

  /// \enum Coordinate_Transformation_Param rigid body rotation parameter index
  enum Coordinate_Transformation_Param {
    ANGLE_X = 0,
    ANGLE_Y,
    ANGLE_Z,
    TRANSLATION_X,
    TRANSLATION_Y,
    TRANSLATION_Z
  };

  /// \brief default constructor
  Camera_System();

  /// \brief constructor
  /// \param param_file_name the name of the file to parse the calibration parameters from
  Camera_System(const std::string & param_file_name);

  // Pure virtual destructor
  virtual ~Camera_System() {};

  /// \brief read the camera system parameters
  /// \param file File name of the parameters file
  void read_camera_system_file(const std::string & file);

  /// \brief write the camera system parameters to an xml file
  /// \param file File name of the parameters file to write to
  void write_camera_system_file(const std::string & file);

  /// \brief set the system type
  /// \param system_type a System_Type_3D enum
  void set_system_type(const System_Type_3D system_type) {
    if(system_type==OPENCV) extrinsics_relative_camera_to_camera_ = true;
    sys_type_ = system_type;
  }

  /// \brief get the integer value for the current system type
  /// \param system_type integer system type value
  System_Type_3D system_type() const {
    return sys_type_;
  }

  /// diffs all the cameras that make up a camera system
  scalar_t diff(const Camera_System & rhs) const{
    TEUCHOS_TEST_FOR_EXCEPTION(rhs.num_cameras()!=num_cameras(),std::runtime_error,"");
    scalar_t diff_ = 0.0;
    for(size_t i=0;i<num_cameras();++i)
      diff_ += camera(i)->diff(*(rhs.camera(i).get()));
    return diff_;
  }

  /// returns true if the extrinsic parameters are camera to camera relative rather
  /// than from world to camera (if this is the case, the first camera's extrinsics
  /// are world to camera 0, the second camera's extrinsics are the camera 0 to camera 1
  /// transformation) This is included for legacy reasons
  bool extrinsics_relative_camera_to_camera()const {return extrinsics_relative_camera_to_camera_;}

  /// \brief returns the number of the first camera with a matching identifier
  /// \param cam_id camera identifying string
  size_t get_camera_num_from_id(std::string & cam_id) const {
    std::transform(cam_id.begin(), cam_id.end(),cam_id.begin(),::toupper);
    std::string id_val;
    for (size_t i = 0; i < num_cameras(); i++) {
      id_val = cameras_[i]->id();
      std::transform(id_val.begin(), id_val.end(),id_val.begin(),::toupper);
      if (id_val==cam_id)return i;
    }
    return -1;
  }

  /// \brief return a pointer to the camera with this vector index
  Teuchos::RCP<Camera> camera(const size_t i)const{
    TEUCHOS_TEST_FOR_EXCEPTION(i>=num_cameras(),std::runtime_error,"");
    return cameras_[i];
  }

  /// return the number of cameras
  size_t num_cameras()const{
    return cameras_.size();
  }

  /// add a camera to the set
  void add_camera(Teuchos::RCP<Camera> camera_ptr){
    cameras_.push_back(camera_ptr);
  }

  /// calculate the fundamental matrix for a set of cameras
  Matrix<scalar_t,3> fundamental_matrix(const size_t source_cam_id=0,
    const size_t target_cam_id=1);

  /// comparison operator
  DICE_LIB_DLL_EXPORT
  friend bool operator==(const Camera_System & lhs,const Camera_System & rhs);
  /// comparison operator
  DICE_LIB_DLL_EXPORT
  friend bool operator!=(const Camera_System & lhs,const Camera_System & rhs){
    return !(lhs==rhs);
  }

  /// overaload the ostream operator for a camera system class
  DICE_LIB_DLL_EXPORT
  friend std::ostream & operator<<(std::ostream & os, const Camera_System & camera_system);

  /// \brief 3 parameter projection routine from the one camera to another camera
  /// \param source_id the number id of the source camera
  /// \param target_id the camera id of the target camera.
  /// \param img_source_x array of x image location in the source camera
  /// \param img_source_y array of y image location in the source camera
  /// \param img_targe_x array of projected x image location in the target camera
  /// \param img_target_y array of projected y image location in the target camera
  /// \param facet_params array of ZP, THETA, PHI values that govern the projection
  /// \param img_target_dx array of arrays of the partials in the x location wrt (ZP, THETA, PHI)
  /// \param img_target_dy array of arrays of the partials in the y location wrt (ZP, THETA, PHI)
  /// \param rigid_body_params 6 rotation/translation parameters to describe the rigid body motion
  /// if the rigid_body_params argument is specified
  /// points in space are established by projecting them using the 3 parameter projection,
  /// they are then transformed in a rigid body manner to a second location
  /// which is then projected into the target camera. ZP, THETA and PHI are considered fixed
  void camera_to_camera_projection(
    const size_t source_id,
    const size_t target_id,
    const std::vector<scalar_t> & img_source_x,
    const std::vector<scalar_t> & img_source_y,
    std::vector<scalar_t> & img_target_x,
    std::vector<scalar_t> & img_target_y,
    const std::vector<scalar_t> & facet_params,
    std::vector<std::vector<scalar_t> > & img_target_dx,
    std::vector<std::vector<scalar_t> > & img_target_dy,
    const std::vector<scalar_t> & rigid_body_params = std::vector<scalar_t>());

  /// same as above, only do not compute the partials
  void camera_to_camera_projection(
    const size_t source_id,
    const size_t target_id,
    const std::vector<scalar_t> & img_source_x,
    const std::vector<scalar_t> & img_source_y,
    std::vector<scalar_t> & img_target_x,
    std::vector<scalar_t> & img_target_y,
    const std::vector<scalar_t> & facet_params,
    const std::vector<scalar_t> & rigid_body_params = std::vector<scalar_t>()){
    std::vector<std::vector<scalar_t> > dummy_vec; // dummy vec of size zero to flag no deriv calcs
    camera_to_camera_projection(source_id,target_id,
      img_source_x,img_source_y,
      img_target_x,img_target_y,
      facet_params,dummy_vec,dummy_vec,rigid_body_params);
  }

  ///rigid body 3D rotation/translation transformation with no incoming partials and outgoing partials wrt params
  /// \param source_x array of initial x world position of the point
  /// \param source_y array of initial y world position of the point
  /// \param source_z array of initial z world position of the point
  /// \param targe_x array of x world position of the point after the transformation
  /// \param target_y array of y world position of the point after the transformation
  /// \param target_z array of z world position of the point after the transformation
  /// \param rigid_body_params 6 rotation/translation parameters to describe the rigid body motion (3 angles 3 translations)
  /// \param target_dx array of arrays of the partials in the x world location wrt params
  /// \param target_dy array of arrays of the partials in the y world location wrt params
  /// \param target_dy array of arrays of the partials in the z world location wrt params
  void rot_trans_3D(const std::vector<scalar_t> & source_x,
    const std::vector<scalar_t> & source_y,
    const std::vector<scalar_t> & source_z,
    std::vector<scalar_t> & target_x,
    std::vector<scalar_t> & target_y,
    std::vector<scalar_t> & target_z,
    const std::vector<scalar_t> & rigid_body_params,
    std::vector < std::vector<scalar_t> > & target_dx,
    std::vector < std::vector<scalar_t> > & target_dy,
    std::vector < std::vector<scalar_t> > & target_dz);

  /// same as above with no partials
  void rot_trans_3D(const std::vector<scalar_t> & source_x,
    const std::vector<scalar_t> & source_y,
    const std::vector<scalar_t> & source_z,
    std::vector<scalar_t> & target_x,
    std::vector<scalar_t> & target_y,
    std::vector<scalar_t> & target_z,
    const std::vector<scalar_t> & rigid_body_params){
    std::vector < std::vector<scalar_t> > dummy_vec;
    rot_trans_3D(source_x,source_y,source_z,
      target_x,target_y,target_z,
      rigid_body_params,dummy_vec,dummy_vec,dummy_vec);

  }

//  ///rigid body 3D rotation/translation transformation with incoming partials and outgoing partials wrt params
//  /// \param wld0_x array of initial x world position of the point
//  /// \param wld0_y array of initial y world position of the point
//  /// \param wld0_z array of initial z world position of the point
//  /// \param wld1_x array of x world position of the point after the transformation
//  /// \param wld1_y array of y world position of the point after the transformation
//  /// \param wld1_z array of z world position of the point after the transformation
//  /// \param params 6 rotation/translation parameters to describe the rigid body motion
//  /// \param wld0_dx array of the incoming partials in the x world location from other parameters
//  /// \param wld0_dy array of the incoming partials in the y world location from other parameters
//  /// \param wld0_dy array of the incoming partials in the z world location from other parameters
//  /// \param wld1_dx array of of arrays the partials in the x world location wrt the other parameters and params
//  /// \param wld1_dy array of of arrays the partials in the y world location wrt the other parameters and params
//  /// \param wld1_dy array of of arrays the partials in the z world location wrt the other parameters and params
//  void rot_trans_3D(const std::vector<scalar_t> & wld0_x,
//    const std::vector<scalar_t> & wld0_y,
//    const std::vector<scalar_t> & wld0_z,
//    std::vector<scalar_t> & wld1_x,
//    std::vector<scalar_t> & wld1_y,
//    std::vector<scalar_t> & wld1_z,
//    const std::vector<scalar_t> & params,
//    const std::vector < std::vector<scalar_t> > & wld0_dx,
//    const std::vector < std::vector<scalar_t> > & wld0_dy,
//    const std::vector < std::vector<scalar_t> > & wld0_dz,
//    std::vector < std::vector<scalar_t> > & wld1_dx,
//    std::vector < std::vector<scalar_t> > & wld1_dy,
//    std::vector < std::vector<scalar_t> > & wld1_dz);

private:

  /// sets the number of cameras that can be in an input file
  const size_t max_num_cameras_allowed_;

  /// these are for compatibility with triangulation and only support 2 camera systems
  /// 12 parameters that define a user supplied transforamation (independent from intrinsic and extrinsic parameters)
  Matrix<scalar_t,4> user_4x4_trans_;

  /// 8 parameters that define a projective transform (independent from intrinsic and extrinsic parameters)
  std::vector<scalar_t> user_6x1_trans_;

  // 3x4 openCV rotation parameters from stereo calibration
  //Matrix<scalar_t,3,4> opencv_3x4_trans_;

  /// defines the camera system type (OPENCV VIC3D...)
  System_Type_3D sys_type_;

  /// the user defined a 6 parameter transform to modify the world coordinate system
  bool has_6_transform_;

  /// the user defined a 4x transform to another coordinate system to use for the world coordinates
  bool has_4x4_transform_;

  /// the standard convention is to have the camera extrinsics be from world to camera
  /// when this flag is true, the extrinsics are the transformation from the left camera to the right instead
  bool extrinsics_relative_camera_to_camera_;

  /// cameras are stored in a vector
  std::vector<Teuchos::RCP<DICe::Camera> > cameras_;

};

}// End DICe Namespace

#endif

