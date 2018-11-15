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

namespace DICe {


	/// \class DICe::Camera
	/// \brief A class for camera calibration parameters and computing single camera projection transformations
	///
	class DICE_LIB_DLL_EXPORT
		Camera {
	public:
		/// \brief Default constructor
		/// \param param_file_name the name of the file to parse the calibration parameters from

		/// \brief constructor with no args
		Camera() {
			clear_camera();
		};

		// minimun constructor for the entire camera
		Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics) {
			clear_camera();
			set_Identifier(cam_id);
			set_Intrinsics(intrinsics);
			set_Image_Height(image_height);
			set_Image_Width(image_width);
		}

		// constructor for the entire camera with extrinsic parameters
		Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics) {
			clear_camera();
			set_Identifier(cam_id);
			set_Intrinsics(intrinsics);
			set_Image_Height(image_height);
			set_Image_Width(image_width);
			set_Extrinsics(extrinsics);
		}

		// constructor for the entire camera with extrinsic parameters and rotation matrix
		Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics,
			std::vector<std::vector<scalar_t>> & rotation_3x3_matrix) {
			clear_camera();
			set_Identifier(cam_id);
			set_Intrinsics(intrinsics);
			set_Image_Height(image_height);
			set_Image_Width(image_width);
			set_Extrinsics(extrinsics);
			set_3x3_Rotation_Matrix(rotation_3x3_matrix);
		}

		// full constructor for the entire camera
		Camera(std::string cam_id, int_t image_width, int_t image_height, std::vector<scalar_t> intrinsics, std::vector<scalar_t> extrinsics,
			std::vector<std::vector<scalar_t>> & rotation_3x3_matrix, std::string camera_lens = "", std::string camera_comments = "", 
			int_t pixel_depth=0) {
			clear_camera();
			set_Identifier(cam_id);
			set_Intrinsics(intrinsics);
			set_Image_Height(image_height);
			set_Image_Width(image_width);
			set_Extrinsics(extrinsics);
			set_3x3_Rotation_Matrix(rotation_3x3_matrix);
			set_Camera_Lens(camera_lens);
			set_Camera_Comments(camera_comments);
		}

		/// Pure virtual destructor
		virtual ~Camera() {}

		///gets/sets the camera identifier
		void set_Identifier(std::string cam_id) {
			camera_id_ = cam_id;
			camera_filled_ = true;
		}
		std::string get_Identifier() { return camera_id_; }

		///gets/sets the camera lens identifier
		void set_Camera_Lens(std::string camera_lens) {
			camera_lens_ = camera_lens;
			camera_filled_ = true;
		}
		std::string get_Camera_Lens() { return camera_lens_; }

		///gets/sets the camera comment
		void set_Camera_Comments(std::string camera_comments) {
			camera_comments_ = camera_comments;
			camera_filled_ = true;
		}
		std::string get_Camera_Comments() { return camera_comments_; }

		///gets/sets the image height
		void set_Image_Height(int_t height) {
			if (image_height_ != height) camera_prepped_ = false;
			image_height_ = height;
			camera_filled_ = true;
		}
		int_t get_Image_Height() { return image_height_; }

		///gets/sets the image width
		void set_Image_Width(int_t width) {
			if (image_width_ != width) camera_prepped_ = false;
			image_width_ = width;
			camera_filled_ = true;
		}
		int_t get_Image_Width() { return image_width_; }

		///gets/sets the pixel depth
		void set_Pixel_Depth(int_t pixel_depth) {
			pixel_depth_ = pixel_depth;
			camera_filled_ = true;
		}
		int_t get_Pixel_Depth() { return pixel_depth_; }

		///gets/sets intrinsic values
		void set_Intrinsics(std::vector<scalar_t> & intrinsics) {
			for (int_t i = 0; i < MAX_CAM_INTRINSIC_PARAMS; i++) {
				if (intrinsics_[i] != intrinsics[i]) camera_prepped_ = false;
				intrinsics_[i] = intrinsics[i];
			}
			camera_filled_ = true;
		}
		void get_Intrinsics(std::vector<scalar_t> & intrinsics) {
			for (int_t i = 0; i < MAX_CAM_INTRINSIC_PARAMS; i++)
				intrinsics[i] = intrinsics_[i];
		}

		///gets/sets extrinsic values
		void set_Extrinsics(std::vector<scalar_t> & extrinsics) {
			for (int_t i = 0; i < MAX_CAM_EXTRINSIC_PARAMS; i++) {
				if (extrinsics_[i] != extrinsics[i]) camera_prepped_ = false;
				extrinsics_[i] = extrinsics[i];
			}
			camera_filled_ = true;
		}
		void get_Extrinsics(std::vector<scalar_t> & extrinsics) {
			for (int_t i = 0; i < MAX_CAM_EXTRINSIC_PARAMS; i++)
				extrinsics[i] = extrinsics_[i];
		}

		///gets/sets rotation matrix
		void set_3x3_Rotation_Matrix(std::vector<std::vector<scalar_t>> & rotation_3x3_matrix) {
			for (int_t i = 0; i < 3; i++) {
				for (int_t j = 0; j < 3; j++) {
					if (rotation_3x3_matrix_[i][j] != rotation_3x3_matrix[i][j]) camera_prepped_ = false;
					rotation_3x3_matrix_[i][j] = rotation_3x3_matrix[i][j];
				}
			}
			camera_filled_ = true;
			rot_3x3_matrix_filled_ = true;
		}
		void get_3x3_Rotation_Matrix(std::vector<std::vector<scalar_t>> & rotation_3x3_matrix) {
			for (int_t i = 0; i < 3; i++) {
				for (int_t j = 0; j < 3; j++) {
					rotation_3x3_matrix[i][j] = rotation_3x3_matrix_[i][j];
				}
			}
		}
		
		///gets the camera to world transformation matrix
		void get_Cam_World_Trans_Matrix(std::vector<std::vector<scalar_t>> & cam_world_trans) {
			for (int_t i = 0; i < 4; i++) {
				for (int_t j = 0; j < 4; j++) {
					cam_world_trans[i][j] = cam_world_trans_[i][j];
				}
			}
		}

		///gets the world to camera transformation matrix
		void get_World_Cam_Trans_Matrix(std::vector<std::vector<scalar_t>> & world_cam_trans) {
			for (int_t i = 0; i < 4; i++) {
				for (int_t j = 0; j < 4; j++) {
					world_cam_trans[i][j] = world_cam_trans_[i][j];
				}
			}
		}

		/// does the camera have enough values to be valid
		bool camera_valid(std::string & msg) {
			return check_valid_(msg);
		}

		//have any of the parameters/fields been filled
		bool camera_filled() {
			return camera_filled_;
		}

		//has the 3x3 rotation matrix been filled
		bool camera_has_3x3_rotation() {
			return rot_3x3_matrix_filled_;
		}

		//has the camera been prepped
		bool camera_prepped(){
			return camera_prepped_;
		}

		/// \brief clear the values for all the cameras
		/// \param
		void clear_camera();

		//prepares the tranfomation matricies
		bool prep_camera();

		// convert sensor locations to image locations: applies lens distortion scales for fx,fy and 
		// converts to image coordiates with cx, cy
		void sensor_to_image(
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y,
			std::vector<scalar_t> & image_x,
			std::vector<scalar_t> & image_y);
		//overloaded with first partials
		void sensor_to_image(
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y,
			std::vector<scalar_t> & image_x,
			std::vector<scalar_t> & image_y,
			std::vector<std::vector<scalar_t>> & sen_dx,
			std::vector<std::vector<scalar_t>> & sen_dy,
			std::vector<std::vector<scalar_t>> & image_dx,
			std::vector<std::vector<scalar_t>> & image_dy);

		//converts image coordinates to sensor coordinates (lens distortion^-1, fx,fy,cx,cy)
		void image_to_sensor(
			std::vector<scalar_t> & image_x,
			std::vector<scalar_t> & image_y,
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y,
			bool integer_locs = true);

		//projects sensor coordinates onto a plane in space described by zp,theta,phi
		void sensor_to_cam(
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y,
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z,
			std::vector<scalar_t> & params);
		//overloaded with first partials
		void sensor_to_cam(
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y,
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z,
			std::vector<scalar_t> & params,
			std::vector<std::vector<scalar_t>> & cam_dx,
			std::vector<std::vector<scalar_t>> & cam_dy,
			std::vector<std::vector<scalar_t>> & cam_dz);

		//projects x,y,z locations back onto the sensor
		void cam_to_sensor(
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z,
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y);
		//overloaded with first partials
		void cam_to_sensor(
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z,
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y,
			std::vector<std::vector<scalar_t>> & cam_dx,
			std::vector<std::vector<scalar_t>> & cam_dy,
			std::vector<std::vector<scalar_t>> & cam_dz,
			std::vector<std::vector<scalar_t>> & sen_dx,
			std::vector<std::vector<scalar_t>> & sen_dy);

		//converts the camera x,y,z cooldinates to a world x,y,z coordinate system
		void cam_to_world(
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z,
			std::vector<scalar_t> & wrld_x,
			std::vector<scalar_t> & wrld_y,
			std::vector<scalar_t> & wrld_z) {
			rot_trans_transform_(cam_world_trans_, cam_x, cam_y, cam_z, wrld_x, wrld_y, wrld_z);
		}
		//overloaded for first partials
		void cam_to_world(
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z,
			std::vector<scalar_t> & wrld_x,
			std::vector<scalar_t> & wrld_y,
			std::vector<scalar_t> & wrld_z,
			std::vector<std::vector<scalar_t>> & cam_dx,
			std::vector<std::vector<scalar_t>> & cam_dy,
			std::vector<std::vector<scalar_t>> & cam_dz,
			std::vector<std::vector<scalar_t>> & wrld_dx,
			std::vector<std::vector<scalar_t>> & wrld_dy,
			std::vector<std::vector<scalar_t>> & wrld_dz) {
			rot_trans_transform_(cam_world_trans_, wrld_x, wrld_y, wrld_z, cam_x, cam_y, cam_z,
				cam_dx, cam_dy, cam_dz, wrld_dx, wrld_dy, wrld_dz);
		}

		//converts world x,y,z coordinates to camera x,y,z coordinates
		void world_to_cam(
			std::vector<scalar_t> & wrld_x,
			std::vector<scalar_t> & wrld_y,
			std::vector<scalar_t> & wrld_z,
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z) {
			rot_trans_transform_(world_cam_trans_, wrld_x, wrld_y, wrld_z, cam_x, cam_y, cam_z);
		}
		//overloaded for first partials
		void world_to_cam(
			std::vector<scalar_t> & wrld_x,
			std::vector<scalar_t> & wrld_y,
			std::vector<scalar_t> & wrld_z,
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z,
			std::vector<std::vector<scalar_t>> & wrld_dx,
			std::vector<std::vector<scalar_t>> & wrld_dy,
			std::vector<std::vector<scalar_t>> & wrld_dz,
			std::vector<std::vector<scalar_t>> & cam_dx,
			std::vector<std::vector<scalar_t>> & cam_dy,
			std::vector<std::vector<scalar_t>> & cam_dz) {
			rot_trans_transform_(world_cam_trans_, wrld_x, wrld_y, wrld_z, cam_x, cam_y, cam_z,
				wrld_dx, wrld_dy, wrld_dz, cam_dx, cam_dy, cam_dz);
		}


	private:
		//generic translation/rotation transform 
		void rot_trans_transform_(
			std::vector<std::vector<scalar_t>> & RT_matrix,
			std::vector<scalar_t> & in_x,
			std::vector<scalar_t> & in_y,
			std::vector<scalar_t> & in_z,
			std::vector<scalar_t> & out_x,
			std::vector<scalar_t> & out_y,
			std::vector<scalar_t> & out_z);
		//overloaded for first partials
		void rot_trans_transform_(
			std::vector<std::vector<scalar_t>> & RT_matrix,
			std::vector<scalar_t> & in_x,
			std::vector<scalar_t> & in_y,
			std::vector<scalar_t> & in_z,
			std::vector<scalar_t> & out_x,
			std::vector<scalar_t> & out_y,
			std::vector<scalar_t> & out_z,
			std::vector<std::vector<scalar_t>> & in_dx,
			std::vector<std::vector<scalar_t>> & in_dy,
			std::vector<std::vector<scalar_t>> & in_dz,
			std::vector<std::vector<scalar_t>> & out_dx,
			std::vector<std::vector<scalar_t>> & out_dy,
			std::vector<std::vector<scalar_t>> & out_dz);

		//routine to check for a valid camera
		bool check_valid_(std::string & msg);
		//creates the image values for the inverse lens distortion
		bool prep_lens_distortion_();
		//creates the rotation/translation matricies and inverses
		bool prep_transforms_();

		//has the camera been filled with any parameters
		bool camera_filled_ = false;
		//has the camera prep been run
		bool camera_prepped_ = false;
		//did the user supply a rotation matrix
		bool rot_3x3_matrix_filled_ = false;

		/// 18 member array of camera intrinsics
		/// first index is the camera id, second index is: 
		///   openCV_DIS - (cx cy fx fy k1 k2 p1 p2 [k3] [k4 k5 k6] [s1 s2 s3 s4] [tx ty])   4,5,8,12 or 14 distortion coef
		///		 Vic3D_DIS - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r1 k2r2 k3r3)
		///	    generic1 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r1 k2r2 k3r3)
		///	    generic2 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r2 k2r4 k3r6)
		///	    generic3 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r3 k2r5 k3r7)
		std::vector<scalar_t> intrinsics_;

		/// 6 member array of camera extrinsics
		/// index is alpha beta gamma tx ty tz (the Euler angles + translations)
		std::vector<scalar_t> extrinsics_;

		/// 3x3 rotation matrix
		std::vector<std::vector<scalar_t>> rotation_3x3_matrix_;

		// Inverse lense distortion values for each pixel in an image
		std::vector<scalar_t> inv_lens_dis_x_;
		std::vector<scalar_t> inv_lens_dis_y_;

		// transformation coefficients
		std::vector<std::vector<scalar_t>> cam_world_trans_;
		std::vector<std::vector<scalar_t>> world_cam_trans_;

		//Other camera parameters
		int_t image_height_ = 0;
		int_t image_width_ = 0;
		int_t pixel_depth_ = 0;

		std::string camera_id_;
		std::string camera_lens_;
		std::string camera_comments_;
	};

}// End DICe Namespace

#endif

