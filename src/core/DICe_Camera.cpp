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

#include <DICe_Camera.h>
#include <DICe_LocalShapeFunction.h>

#include <fstream>
#include <math.h>

namespace DICe {

bool
Camera::Camera_Info::is_valid(){
  try{check_valid();}
  catch(...){
    return false;
  }
  return true;
};

bool
operator==(const Camera::Camera_Info & lhs,const Camera::Camera_Info & rhs){
  bool is_equal = true;
  const scalar_t tol = 1.0E-5;
  std::ios_base::fmtflags f( std::cout.flags() );
  std::cout.precision(6);
  std::cout << std::scientific;
  if(lhs.intrinsics_.size()!=rhs.intrinsics_.size()){
    DEBUG_MSG("intrinsic vectors are different sizes");
    std::cout.flags(f);
    return false;
  }
  for(size_t i=0;i<lhs.intrinsics_.size();++i){
    if(std::abs(lhs.intrinsics_[i]-rhs.intrinsics_[i])>tol){
      DEBUG_MSG("Camera_Info intrinsics (index " << i <<  ")  are not equivalent, lhs " << lhs.intrinsics_[i] << " rhs " << rhs.intrinsics_[i]);
      is_equal = false;
    }
  }
  if(std::abs(lhs.tx_-rhs.tx_)>tol){
    DEBUG_MSG("lhs.tx " << lhs.tx_ << " rhs.tx " << rhs.tx_);
    is_equal = false;
  }
  if(std::abs(lhs.ty_-rhs.ty_)>tol){
    DEBUG_MSG("lhs.ty " << lhs.ty_ << " rhs.ty " << rhs.ty_);
    is_equal = false;
  }
  if(std::abs(lhs.tz_-rhs.tz_)>tol){
    DEBUG_MSG("lhs.tz " << lhs.tz_ << " rhs.tz " << rhs.tz_);
    is_equal = false;
  }
  if(lhs.image_width_!=rhs.image_width_){
    DEBUG_MSG("lhs.image_width " << lhs.image_width_ << " rhs.image_width " << rhs.image_width_ );
    is_equal = false;
  }
  if(lhs.image_height_!=rhs.image_height_){
    DEBUG_MSG("lhs.image_height " << lhs.image_height_ << " rhs.image_height " << rhs.image_height_ );
    is_equal = false;
  }
  if(lhs.pixel_depth_!=rhs.pixel_depth_){
    DEBUG_MSG("lhs.pixel_depth " << lhs.pixel_depth_ << " rhs.pixel_depth " << rhs.pixel_depth_);
    is_equal = false;
  }
  if(lhs.lens_distortion_model_!=rhs.lens_distortion_model_){
    DEBUG_MSG("lhs.lens_distortion_model " << Camera::to_string(lhs.lens_distortion_model_) << " rhs.lens_distortion_model " << Camera::to_string(rhs.lens_distortion_model_));
    is_equal = false;
  }
  if(lhs.rotation_matrix_!=rhs.rotation_matrix_){
    DEBUG_MSG("rotation matrices don't match");
#ifdef DICE_DEBUG_MSG
    std::cout << lhs.rotation_matrix_ << std::endl;
    std::cout << rhs.rotation_matrix_ << std::endl;
#endif
    is_equal = false;
  }
  std::cout.flags( f );
  // dont compare the string fields
  return is_equal;
}

void
Camera::Camera_Info::set_rotation_matrix(const scalar_t & alpha,
  const scalar_t & beta,
  const scalar_t & gamma){
  const scalar_t cx = std::cos(alpha*DICE_PI/180.0); // input as degrees, need radians
  const scalar_t sx = std::sin(alpha*DICE_PI/180.0);
  const scalar_t cy = std::cos(beta*DICE_PI/180.0);
  const scalar_t sy = std::sin(beta*DICE_PI/180.0);
  const scalar_t cz = std::cos(gamma*DICE_PI/180.0);
  const scalar_t sz = std::sin(gamma*DICE_PI/180.0);
  rotation_matrix_(0,0) = cy*cz;
  rotation_matrix_(0,1) = sx*sy*cz-cx*sz;
  rotation_matrix_(0,2) = cx*sy*cz+sx*sz;
  rotation_matrix_(1,0) = cy*sz;
  rotation_matrix_(1,1) = sx*sy*sz+cx*cz;
  rotation_matrix_(1,2) = cx*sy*sz-sx*cz;
  rotation_matrix_(2,0) = -sy;
  rotation_matrix_(2,1) = sx*cy;
  rotation_matrix_(2,2) = cx*cy;
}

void
Camera::Camera_Info::check_valid()const{ // throws an exception for an invalid camera
  bool is_valid = false;
  bool has_intrinsics = false;
  for(size_t i=0;i<MAX_CAM_INTRINSIC_PARAM;++i)
    if(intrinsics_[i]!=0) has_intrinsics = true;
  if(!has_intrinsics)
    std::cout << "Camera_Info: error, all intrinsic values are 0" << std::endl;
  bool has_rotation_matrix = !rotation_matrix_.all_values_are_zero();
  if(!has_rotation_matrix)
    std::cout << "Camera_Info: error, all rotation matrix values are 0" << std::endl;
  //extrinsics can all be zero and that's valid
  if(image_height_<=0)
    std::cout << "Camera_Info: error, invalid image height: " << image_height_ << std::endl;
  if(image_width_<=0)
    std::cout << "Camera_Info: error, invalid image width: " << image_width_ << std::endl;
  is_valid = image_height_>0&&
      image_width_>0&&
      has_intrinsics&&
      has_rotation_matrix;
  TEUCHOS_TEST_FOR_EXCEPTION(!is_valid,std::runtime_error,"");
};

void
Camera::Camera_Info::clear(){
  std::fill(intrinsics_.begin(), intrinsics_.end(), 0);
  rotation_matrix_ = Matrix<scalar_t,3>::identity();
  tx_ = 0;
  ty_ = 0;
  tz_ = 0;
  image_height_ = -1;
  image_width_ = -1;
  pixel_depth_ = -1;
  id_.clear();
  lens_.clear();
  comments_.clear();
}

void
Camera::initialize() {
  //run the pre-run functions
  // TODO need to add more error handling into the functions
  inv_lens_dis_x_.assign(camera_info_.image_height_*camera_info_.image_width_,0);
  inv_lens_dis_y_.assign(camera_info_.image_height_*camera_info_.image_width_,0);
  prep_lens_distortion();
  prep_transforms();
}

void
Camera::prep_transforms() {
  camera_info_.check_valid();
  //clear the world to camera coordinate transform values
  world_cam_trans_.put_value(0);
  //create the 4x4 transformation matrix
  for (size_t i = 0; i < rotation_matrix()->rows(); i++) {
    for (size_t j = 0; j < rotation_matrix()->cols(); j++) {
      world_cam_trans_(i,j) = (*rotation_matrix())(i,j);
    }
  }
  world_cam_trans_(0,3) = tx();
  world_cam_trans_(1,3) = ty();
  world_cam_trans_(2,3) = tz();
  world_cam_trans_(3,3) = 1.0;
  cam_world_trans_ = world_cam_trans_.inv();
}


void
Camera::prep_lens_distortion() {
  camera_info_.check_valid();
  //pre-run lens distortion function
  scalar_t del_img_x;
  scalar_t del_img_y;
  const scalar_t end_crit = 0.0001;
  std::stringstream msg_output;
  bool end_loop;

  //get the needed intrinsic values
  const scalar_t fx = (*intrinsics())[FX];
  const scalar_t fy = (*intrinsics())[FY];
  //const scalar_t fs = (*intrinsics())[FS];
  const scalar_t cx = (*intrinsics())[CX];
  const scalar_t cy = (*intrinsics())[CY];

  //set the image size
  const size_t image_size = image_height() * image_width();

  //initialize the arrays
  assert(inv_lens_dis_x_.size()==image_size);
  assert(inv_lens_dis_y_.size()==image_size);
  std::vector<scalar_t> image_x(image_size, 0.0);
  std::vector<scalar_t> image_y(image_size, 0.0);
  std::vector<scalar_t> targ_x(image_size, 0.0);
  std::vector<scalar_t> targ_y(image_size, 0.0);
  std::vector<scalar_t> params(1, 0.0);

  for (size_t i = 0; i < image_size; i++) {
    //set the target value for x,y
    targ_x[i] = (scalar_t)(i % image_width());
    targ_y[i] = (scalar_t)(i / image_width());
    //generate the initial guss for the inverted sensor position
    inv_lens_dis_x_[i] = (targ_x[i] - cx) / fx;
    inv_lens_dis_y_[i] = (targ_y[i] - cy) / fy;
  }

  //iterate until the inverted point is near the target location
  const size_t max_its = 60;
  for (size_t j = 0; j < max_its; j++) {
    TEUCHOS_TEST_FOR_EXCEPTION(j==max_its-1,std::runtime_error,"error: max iterations reached in inverse distortion prep loop");
    end_loop = true;
    DEBUG_MSG("Inverse distortion prep iteration  " << j);
    //do the projection
    sensor_to_image(inv_lens_dis_x_, inv_lens_dis_y_, image_x, image_y);
    //apply the correction
    for (size_t i = 0; i < image_size; i++) {
      del_img_x = targ_x[i] - image_x[i];
      del_img_y = targ_y[i] - image_y[i];
      if ((abs(del_img_x) > end_crit) || (abs(del_img_y) > end_crit)) end_loop = false;
      inv_lens_dis_x_[i] += (del_img_x) / fx;
      inv_lens_dis_y_[i] += (del_img_y) / fy;
    }
    if (end_loop) break;
  }
}


void
Camera::image_to_sensor(
  const std::vector<scalar_t> & image_x,
  const std::vector<scalar_t> & image_y,
  std::vector<scalar_t> & sen_x,
  std::vector<scalar_t> & sen_y,
  const bool integer_locs) {
  camera_info_.check_valid();

  const size_t vec_size = image_x.size();
  TEUCHOS_TEST_FOR_EXCEPTION(image_y.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(sen_x.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(sen_y.size()!=vec_size,std::runtime_error,"");

  //transformation from distorted image locations to undistorted sensor locations
  int_t index;
  int_t index00;
  int_t index10;
  int_t index01;
  int_t index11;
  int_t x_base, y_base;
  scalar_t dx, dy, x, y;
  const int_t img_h = image_height();
  const int_t img_w = image_width();
  assert(inv_lens_dis_x_.size()>0);
  assert(inv_lens_dis_x_.size()==inv_lens_dis_y_.size());
  //if we are at an integer pixel location it is a simple lookup of the pre-calculated values
  if (integer_locs) {
    for (size_t i = 0; i < vec_size; i++) {
      assert((int_t)image_y[i]>=0.0&&(int_t)image_x[i]>=0.0
        &&(int_t)image_y[i]<img_h&&(int_t)image_x[i]<img_w);
      index = static_cast<int_t>(image_y[i]) * img_w + static_cast<int_t>(image_x[i]);
      assert(index>=0&&index<(int_t)inv_lens_dis_x_.size());
      sen_x[i] = inv_lens_dis_x_[index];
      sen_y[i] = inv_lens_dis_y_[index];
    }
  }
  else
  {
    //if not at an interger pixel location use linear interpolation to get the value
    for (size_t i = 0; i < vec_size; i++) {
      x = image_x[i];
      y = image_y[i];
      x_base = static_cast<int_t>(floor(x));
      y_base = static_cast<int_t>(floor(y));
      // make sure the base coordinates are inside the image:
      if(x_base<0) x_base=0;
      if(x_base>img_w-2)x_base=img_w-2;
      if(y_base<0) y_base=0;
      if(y_base>img_h-2)y_base=img_h-2;
      index00 = y_base * img_w + x_base;
      index10 = y_base * img_w + x_base + 1;
      index01 = (y_base + 1) * img_w + x_base;
      index11 = (y_base + 1) * img_w + x_base + 1;
      assert(index00>=0&&index00<(int_t)inv_lens_dis_x_.size());
      assert(index01>=0&&index01<(int_t)inv_lens_dis_x_.size());
      assert(index00>=0&&index10<(int_t)inv_lens_dis_x_.size());
      assert(index00>=0&&index11<(int_t)inv_lens_dis_x_.size());
      dx = x - x_base;
      dy = y - y_base;
      //quick linear interpolation to get the sensor location
      sen_x[i] = inv_lens_dis_x_[index00] * (1 - dx)*(1 - dy) + inv_lens_dis_x_[index10] * dx*(1 - dy)
          + inv_lens_dis_x_[index01] * (1 - dx)*dy + inv_lens_dis_x_[index11] * dx*dy;
      sen_y[i] = inv_lens_dis_y_[index00] * (1 - dx)*(1 - dy) + inv_lens_dis_y_[index10] * dx*(1 - dy)
          + inv_lens_dis_y_[index01] * (1 - dx)*dy + inv_lens_dis_y_[index11] * dx*dy;
    }
  }
}

void
Camera::sensor_to_image(
  const std::vector<scalar_t> & sen_x,
  const std::vector<scalar_t> & sen_y,
  std::vector<scalar_t> & image_x,
  std::vector<scalar_t> & image_y) {
  camera_info_.check_valid();
  //converts sensor locations to image locations by applying lens distortions
  const size_t vec_size = image_x.size();
  TEUCHOS_TEST_FOR_EXCEPTION(image_y.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(sen_x.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(sen_y.size()!=vec_size,std::runtime_error,"");

  //scaling with fx and fy and shifting by cx, cy.
  const scalar_t fx = (*intrinsics())[FX];
  const scalar_t fy = (*intrinsics())[FY];
  const scalar_t fs = (*intrinsics())[FS];
  const scalar_t cx = (*intrinsics())[CX];
  const scalar_t cy = (*intrinsics())[CY];
  const scalar_t k1 = (*intrinsics())[K1];
  const scalar_t k2 = (*intrinsics())[K2];
  const scalar_t k3 = (*intrinsics())[K3];
  const scalar_t k4 = (*intrinsics())[K4];
  const scalar_t k5 = (*intrinsics())[K5];
  const scalar_t k6 = (*intrinsics())[K6];
  const scalar_t p1 = (*intrinsics())[P1];
  const scalar_t p2 = (*intrinsics())[P2];
  const scalar_t s1 = (*intrinsics())[S1];
  const scalar_t s2 = (*intrinsics())[S2];
  const scalar_t s3 = (*intrinsics())[S3];
  const scalar_t s4 = (*intrinsics())[S4];
  const scalar_t t1 = (*intrinsics())[T1];
  const scalar_t t2 = (*intrinsics())[T2];
  scalar_t x_sen, y_sen, rad, dis_coef, rad_sqr;
  scalar_t x_temp, y_temp;

  //use the appropriate lens distortion model (only 8 parameter openCV has been tested)
  switch (lens_distortion_model()) {

    case NO_LENS_DISTORTION:
      for (size_t i = 0; i < vec_size; i++) {
        image_x[i] = sen_x[i] * fx + cx;
        image_y[i] = sen_y[i] * fy + cy;
      }
      break;

    case K1R1_K2R2_K3R3:
      for (size_t i = 0; i < vec_size; i++) {
        x_sen = sen_x[i];
        y_sen = sen_y[i];
        rad = sqrt(x_sen * x_sen + y_sen * y_sen);
        assert(rad!=0.0);
        dis_coef = k1 * rad + k2 * pow(rad, 2) + k3 * pow(rad, 3);
        x_sen = (rad + dis_coef)*x_sen / rad;
        y_sen = (rad + dis_coef)*y_sen / rad;
        image_x[i] = x_sen * fx + cx;
        image_y[i] = y_sen * fy + cy;
      }
      break;

    case K1R2_K2R4_K3R6:
      for (size_t i = 0; i < vec_size; i++) {
        x_sen = sen_x[i];
        y_sen = sen_y[i];
        rad = sqrt(x_sen * x_sen + y_sen * y_sen);
        assert(rad!=0.0);
        dis_coef = k1 * pow(rad, 2) + k2 * pow(rad, 4) + k3 * pow(rad, 6);
        x_sen = (rad + dis_coef)*x_sen / rad;
        y_sen = (rad + dis_coef)*y_sen / rad;
        image_x[i] = x_sen * fx + cx;
        image_y[i] = y_sen * fy + cy;
      }
      break;

    case K1R3_K2R5_K3R7:
      for (size_t i = 0; i < vec_size; i++) {
        x_sen = sen_x[i];
        y_sen = sen_y[i];
        rad = sqrt(x_sen * x_sen + y_sen * y_sen);
        assert(rad!=0.0);
        dis_coef = k1 * pow(rad, 3) + k2 * pow(rad, 5) + k3 * pow(rad, 7);
        x_sen = (rad + dis_coef)*x_sen / rad;
        y_sen = (rad + dis_coef)*y_sen / rad;
        image_x[i] = x_sen * fx + cx;
        image_y[i] = y_sen * fy + cy;
      }
      break;

    case VIC3D_LENS_DISTORTION:  //I believe it is K1R1_K2R2_K3R3 but need to confirm
      for (size_t i = 0; i < vec_size; i++) {
        x_sen = sen_x[i];
        y_sen = sen_y[i];
        rad_sqr = x_sen * x_sen + y_sen * y_sen;
        rad = sqrt((double)rad_sqr);
        assert(rad!=0.0);
        dis_coef = k1 * rad + k2 * pow(rad, 2) + k3 * pow(rad, 3);
        x_sen = (rad + dis_coef)*x_sen / rad;
        y_sen = (rad + dis_coef)*y_sen / rad;
        image_x[i] = x_sen * fx + y_sen * fs + cx;
        image_y[i] = y_sen * fy + cy;
      }
      break;

    case OPENCV_LENS_DISTORTION:
      //equations from: https://docs.opencv.org/3.4.1/d9/d0c/group__calib3d.html
    {
      const bool has_denom = (k4 != 0 || k5 != 0 || k6 != 0);
      const bool has_tangential = (p1 != 0 || p2 != 0);
      const bool has_prism = (s1 != 0 || s2 != 0 || s3 != 0 || s4 != 0);
      const bool has_Scheimpfug = (t1 != 0 || t2 != 0);

      for (size_t i = 0; i < vec_size; i++) {
        x_sen = sen_x[i];
        y_sen = sen_y[i];
        rad_sqr = x_sen * x_sen + y_sen * y_sen;
        dis_coef = 1 + k1 * rad_sqr + k2 * pow(rad_sqr, 2) + k3 * pow(rad_sqr, 3);
        image_x[i] = x_sen * dis_coef;
        image_y[i] = y_sen * dis_coef;
      }
      if (has_denom) {
        for (size_t i = 0; i < vec_size; i++) {
          x_sen = sen_x[i];
          y_sen = sen_y[i];
          rad_sqr = x_sen * x_sen + y_sen * y_sen;
          dis_coef = 1 / (1 + k4 * rad_sqr + k5 * pow(rad_sqr, 2) + k6 * pow(rad_sqr, 3));
          image_x[i] = image_x[i] * dis_coef;
          image_y[i] = image_y[i] * dis_coef;
        }
      }
      if (has_tangential) {
        for (size_t i = 0; i < vec_size; i++) {
          x_sen = sen_x[i];
          y_sen = sen_y[i];
          rad_sqr = x_sen * x_sen + y_sen * y_sen;
          dis_coef = 2 * p1*x_sen*y_sen + p2 * (rad_sqr + 2 * x_sen*x_sen);
          image_x[i] = image_x[i] + dis_coef;
          dis_coef = p1 * (rad_sqr + 2 * y_sen*y_sen) + 2 * p2*x_sen*y_sen;
          image_y[i] = image_y[i] + dis_coef;
        }
      }
      if (has_prism) {
        for (size_t i = 0; i < vec_size; i++) {
          x_sen = sen_x[i];
          y_sen = sen_y[i];
          rad_sqr = x_sen * x_sen + y_sen * y_sen;
          dis_coef = s1 * rad_sqr + s2 * rad_sqr*rad_sqr;
          image_x[i] = image_x[i] + dis_coef;
          dis_coef = s3 * rad_sqr + s4 * rad_sqr*rad_sqr;
          image_y[i] = image_y[i] + dis_coef;
        }
      }
      if (has_Scheimpfug) {
        scalar_t R11, R12, R13, R22, R23, R31, R32, R33;
        scalar_t S11, S12, S13, S21, S22, S23, S31, S32, S33;
        scalar_t norm;

        //calculate the Scheimpflug factors
        //assuming t1, t2 in radians need to find out
        const scalar_t c_t1 = cos(t1);
        const scalar_t s_t1 = sin(t1);
        const scalar_t c_t2 = cos(t2);
        const scalar_t s_t2 = sin(t2);
        R11 = c_t2;
        R12 = s_t1 * s_t2;
        R13 = -s_t2 * c_t1;
        R22 = c_t1;
        R23 = s_t1;
        R31 = s_t2;
        R32 = -c_t2 * s_t1;
        R33 = c_t2 * c_t1;
        S11 = R11 * R33 - R13 * R31;
        S12 = R12 * R33 - R13 * R32;
        S13 = R33 * R13 - R13 * R33;
        S21 = -R23 * R31;
        S22 = R33 * R22 - R23 * R32;
        S23 = R33 * R23 - R23 * R33;
        S31 = R31;
        S32 = R32;
        S33 = R33;

        for (size_t i = 0; i < vec_size; i++) {
          x_temp = image_x[i];
          y_temp = image_y[i];
          norm = 1 / (S31 * x_temp + S32 * y_temp + S33);
          image_x[i] = (x_temp * S11 + y_temp * S12 + S13)*norm;
          image_y[i] = (x_temp * S21 + y_temp * S22 + S23)*norm;
        }
      }
      for (size_t i = 0; i < vec_size; i++) {
        image_x[i] = image_x[i] * fx + cx;
        image_y[i] = image_y[i] * fy + cy;
      }
    }
    break;
    default:
      //raise exception if it gets here?
      for (size_t i = 0; i < vec_size; i++) {
        image_x[i] = sen_x[i] * fx + cx;
        image_y[i] = sen_y[i] * fy + cy;
      }
      break;
  }
}

void
Camera::sensor_to_image(
  const std::vector<scalar_t> & sen_x,
  const std::vector<scalar_t> & sen_y,
  std::vector<scalar_t> & image_x,
  std::vector<scalar_t> & image_y,
  const std::vector<std::vector<scalar_t> > & sen_dx,
  const std::vector<std::vector<scalar_t> > & sen_dy,
  std::vector<std::vector<scalar_t> > & image_dx,
  std::vector<std::vector<scalar_t> > & image_dy) {
  camera_info_.check_valid();
  //overload for derivitives
  //assume the lens distortion is mostly a translation of the subset
  //and does not effect the derivities. The scaling factors and skew will.
  sensor_to_image(sen_x, sen_y, image_x, image_y);
  // dims on sen and image vectors get checked in sensor_to_image

  const scalar_t fx = (*intrinsics())[FX];
  const scalar_t fy = (*intrinsics())[FY];
  const scalar_t fs = (*intrinsics())[FS];

  const size_t vec_size = sen_x.size();
  const size_t num_params = sen_dx.size();
  TEUCHOS_TEST_FOR_EXCEPTION(num_params!=3&&num_params!=6,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(sen_dx.size()!=sen_dy.size(),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(sen_dx.size()!=image_dy.size(),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(sen_dx.size()!=image_dy.size(),std::runtime_error,"");
  for (size_t i = 0; i < num_params; i++) {
    TEUCHOS_TEST_FOR_EXCEPTION(image_dx[i].size()!=vec_size,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(image_dy[i].size()!=vec_size,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(sen_dx[i].size()!=vec_size,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(sen_dy[i].size()!=vec_size,std::runtime_error,"");
  }
  for (size_t i = 0; i < vec_size; i++) {
    for (size_t j = 0; j < num_params; j++) {
      image_dx[j][i] = sen_dx[j][i] * fx + sen_dy[j][i] * fs;
      image_dy[j][i] = sen_dy[j][i] * fy;
    }
  }
}

void
Camera::sensor_to_cam(
  const std::vector<scalar_t> & sen_x,
  const std::vector<scalar_t> & sen_y,
  std::vector<scalar_t> & cam_x,
  std::vector<scalar_t> & cam_y,
  std::vector<scalar_t> & cam_z,
  const std::vector<scalar_t> & params,
  std::vector<std::vector<scalar_t> > & cam_dx,
  std::vector<std::vector<scalar_t> > & cam_dy,
  std::vector<std::vector<scalar_t> > & cam_dz){

  TEUCHOS_TEST_FOR_EXCEPTION(params.size()!=3,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(params.size()!=3,std::runtime_error,"");
  const size_t vec_size = sen_x.size();
  TEUCHOS_TEST_FOR_EXCEPTION(sen_y.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(cam_x.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(cam_y.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(cam_z.size()!=vec_size,std::runtime_error,"");

  // for this method, the derivatives are explicitly with respect to the
  // projection shape function parmeters so the derivative multi-dimensional
  // vectors must be sized appropriately. (i.e. don't allow user to pass through RBM derivatives).
  const bool has_derivatives = cam_dx.size()>0;
  if(has_derivatives){
    TEUCHOS_TEST_FOR_EXCEPTION(cam_dx.size()!=cam_dy.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(cam_dx.size()!=cam_dz.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(cam_dx.size()!=3,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(cam_dy.size()!=3,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(cam_dz.size()!=3,std::runtime_error,"");
    for(size_t i=0;i<cam_dx.size();++i){
      TEUCHOS_TEST_FOR_EXCEPTION(cam_dx[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(cam_dy[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(cam_dz[i].size()!=vec_size,std::runtime_error,"");
    }
  }

  //overloaded for first derivitives
  scalar_t zp = params[Projection_Shape_Function::ZP];
  scalar_t theta = params[Projection_Shape_Function::THETA];
  scalar_t phi = params[Projection_Shape_Function::PHI];
  scalar_t cos_theta, cos_phi, cos_xi, sin_theta, sin_phi;
  scalar_t cos_xi_dtheta, cos_xi_dphi;
  scalar_t x_sen, y_sen, denom, denom2;
  scalar_t denom_dzp, denom_dtheta, denom_dphi;
  scalar_t uxcam, uycam, uzcam;
  scalar_t uxcam_dzp, uxcam_dtheta, uxcam_dphi;
  scalar_t uycam_dzp, uycam_dtheta, uycam_dphi;
  scalar_t uzcam_dzp, uzcam_dtheta, uzcam_dphi;

  cos_theta = cos(theta);
  cos_phi = cos(phi);
  sin_theta = sin(theta);
  sin_phi = sin(phi);
  cos_xi = sqrt(1 - cos_theta * cos_theta - cos_phi * cos_phi);
  //cos_xi_dzp = 0;
  assert(cos_xi!=0.0);
  cos_xi_dtheta = cos_theta * sin_theta / cos_xi;
  cos_xi_dphi = cos_phi * sin_phi / cos_xi;

  for (size_t i = 0; i < vec_size; i++) {
    x_sen = sen_x[i];
    y_sen = sen_y[i];
    denom = (cos_xi + y_sen * cos_phi + x_sen * cos_theta);
    uxcam = x_sen * zp * cos_xi;
    uycam = y_sen * zp * cos_xi;
    uzcam = zp * cos_xi;

    if(has_derivatives){
      //factors for the derivitives
      denom2 = denom * denom;
      denom_dzp = 0;
      denom_dtheta = cos_xi_dtheta - x_sen * sin_theta;
      denom_dphi = cos_xi_dphi - y_sen * sin_phi;

      uxcam_dzp = x_sen * cos_xi;
      uxcam_dtheta = x_sen * zp * cos_xi_dtheta;
      uxcam_dphi = x_sen * zp * cos_xi_dphi;

      uycam_dzp = y_sen * cos_xi;
      uycam_dtheta = y_sen * zp * cos_xi_dtheta;
      uycam_dphi = y_sen * zp * cos_xi_dphi;

      uzcam_dzp = cos_xi;
      uzcam_dtheta = zp * cos_xi_dtheta;
      uzcam_dphi = zp * cos_xi_dphi;
    }
    assert(denom!=0.0);
    //calculate the positions
    cam_x[i] = uxcam / denom;
    cam_y[i] = uycam / denom;
    cam_z[i] = uzcam / denom;

    if(has_derivatives){
      //first derivities
      cam_dx[Projection_Shape_Function::ZP][i] = (denom * uxcam_dzp - uxcam * denom_dzp) / denom2;
      cam_dx[Projection_Shape_Function::THETA][i] = (denom * uxcam_dtheta - uxcam * denom_dtheta) / denom2;
      cam_dx[Projection_Shape_Function::PHI][i] = (denom * uxcam_dphi - uxcam * denom_dphi) / denom2;

      cam_dy[Projection_Shape_Function::ZP][i] = (denom * uycam_dzp - uycam * denom_dzp) / denom2;
      cam_dy[Projection_Shape_Function::THETA][i] = (denom * uycam_dtheta - uycam * denom_dtheta) / denom2;
      cam_dy[Projection_Shape_Function::PHI][i] = (denom * uycam_dphi - uycam * denom_dphi) / denom2;

      cam_dz[Projection_Shape_Function::ZP][i] = (denom * uzcam_dzp - uzcam * denom_dzp) / denom2;
      cam_dz[Projection_Shape_Function::THETA][i] = (denom * uzcam_dtheta - uzcam * denom_dtheta) / denom2;
      cam_dz[Projection_Shape_Function::PHI][i] = (denom * uzcam_dphi - uzcam * denom_dphi) / denom2;
    }
  }
}

void
Camera::cam_to_sensor(
  const std::vector<scalar_t> & cam_x,
  const std::vector<scalar_t> & cam_y,
  const std::vector<scalar_t> & cam_z,
  std::vector<scalar_t> & sen_x,
  std::vector<scalar_t> & sen_y,
  const std::vector<std::vector<scalar_t> > & cam_dx,
  const std::vector<std::vector<scalar_t> > & cam_dy,
  const std::vector<std::vector<scalar_t> > & cam_dz,
  std::vector<std::vector<scalar_t> > & sen_dx,
  std::vector<std::vector<scalar_t> > & sen_dy)
{
  //overloaded for first derivitives
  const size_t vec_size = cam_x.size();
  TEUCHOS_TEST_FOR_EXCEPTION(sen_y.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(sen_x.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(cam_y.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(cam_z.size()!=vec_size,std::runtime_error,"");

  const size_t num_params = cam_dx.size();
  const bool has_derivatives = num_params>0;
  if(has_derivatives){
    TEUCHOS_TEST_FOR_EXCEPTION(num_params!=3&&num_params!=6,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(cam_dx.size()!=sen_dx.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(cam_dx.size()!=sen_dy.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(cam_dx.size()!=cam_dy.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(cam_dx.size()!=cam_dz.size(),std::runtime_error,"");
    for(size_t i=0;i<num_params;++i){
      TEUCHOS_TEST_FOR_EXCEPTION(cam_dx[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(cam_dy[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(cam_dz[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(sen_dx[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(sen_dy[i].size()!=vec_size,std::runtime_error,"");
    }
  }

  for (size_t i = 0; i < vec_size; i++) {
    assert(cam_x[i]!=0.0);
    sen_x[i] = cam_x[i] / cam_z[i];
    sen_y[i] = cam_y[i] / cam_z[i];
    if(has_derivatives){
      for (size_t j = 0; j < num_params; j++) {
        assert(cam_z[j]!=0.0);
        sen_dx[j][i] = (cam_dx[j][i] * cam_z[i] - cam_x[i] * cam_dz[j][i]) / (cam_z[j] * cam_z[j]);
        sen_dy[j][i] = (cam_dy[j][i] * cam_z[i] - cam_y[i] * cam_dz[j][i]) / (cam_z[j] * cam_z[j]);
      }
    }
  }
}

void
Camera::rot_trans_transform(
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
  std::vector<std::vector<scalar_t> > & out_dz)
{
  const size_t vec_size = in_x.size();
  TEUCHOS_TEST_FOR_EXCEPTION(in_y.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(in_z.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(out_x.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(out_y.size()!=vec_size,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(out_z.size()!=vec_size,std::runtime_error,"");
  const size_t num_params = in_dx.size();
  const bool has_partials = num_params > 0;
  if(has_partials){
    // the size has to be 3 (for shape function parameters, or 6 for rigid body motion parameters)
    TEUCHOS_TEST_FOR_EXCEPTION((num_params!=3)&&(num_params!=6),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(in_dx.size()!=in_dy.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(in_dx.size()!=in_dz.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(in_dx.size()!=out_dx.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(in_dx.size()!=out_dy.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(in_dx.size()!=out_dz.size(),std::runtime_error,"");
    for(size_t i=0;i<num_params;++i){
      TEUCHOS_TEST_FOR_EXCEPTION(in_dx[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(in_dy[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(in_dz[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(out_dx[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(out_dy[i].size()!=vec_size,std::runtime_error,"");
      TEUCHOS_TEST_FOR_EXCEPTION(out_dz[i].size()!=vec_size,std::runtime_error,"");
    }
  }

  const scalar_t RT00 = RT_matrix(0,0);
  const scalar_t RT01 = RT_matrix(0,1);
  const scalar_t RT02 = RT_matrix(0,2);
  const scalar_t RT03 = RT_matrix(0,3);
  const scalar_t RT10 = RT_matrix(1,0);
  const scalar_t RT11 = RT_matrix(1,1);
  const scalar_t RT12 = RT_matrix(1,2);
  const scalar_t RT13 = RT_matrix(1,3);
  const scalar_t RT20 = RT_matrix(2,0);
  const scalar_t RT21 = RT_matrix(2,1);
  const scalar_t RT22 = RT_matrix(2,2);
  const scalar_t RT23 = RT_matrix(2,3);

  for (size_t i = 0; i < vec_size; i++) {
    out_x[i] = RT00 * in_x[i] + RT01 * in_y[i] + RT02 * in_z[i] + RT03;
    out_y[i] = RT10 * in_x[i] + RT11 * in_y[i] + RT12 * in_z[i] + RT13;
    out_z[i] = RT20 * in_x[i] + RT21 * in_y[i] + RT22 * in_z[i] + RT23;
    if(has_partials){
      for(size_t j=0;j<num_params;++j){
        out_dx[j][i] = RT00 * in_dx[j][i] + RT01 * in_dy[j][i] + RT02 * in_dz[j][i];
        out_dy[j][i] = RT10 * in_dx[j][i] + RT11 * in_dy[j][i] + RT12 * in_dz[j][i];
        out_dz[j][i] = RT20 * in_dx[j][i] + RT21 * in_dy[j][i] + RT22 * in_dz[j][i];
      }
    }
  }
}

}// end DICe namespace
