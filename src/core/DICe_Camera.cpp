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
#include <DICe_Parser.h>
#include <fstream>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <DICe_XMLUtils.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Array.hpp>
#include <math.h>


namespace DICe {

	DICE_LIB_DLL_EXPORT
		void Camera::clear_camera() {
		//clear the intrinsic and extrinsic vectors and fill with zero values
		//const size_t int_size = MAX_CAM_INTRINSIC_PARAMS;
		//const size_t ext_size = MAX_CAM_EXTRINSIC_PARAMS;
		//const double new_value = 0.0;
		//std::cout << "jeff debug line " << extrinsics_.size() << std::endl;
		intrinsics_.assign(MAX_CAM_INTRINSIC_PARAMS, 0.0);
		extrinsics_.assign(MAX_CAM_EXTRINSIC_PARAMS, 0.0);

		//clear the other values
		camera_id_.clear();
		image_height_ = 0;
		image_width_ = 0;
		pixel_depth_ = 0;
		camera_lens_.clear();
		camera_comments_.clear();
		camera_filled_ = false;
	}

	bool Camera::prep_camera() {
		bool return_val = false;
		return_val = prep_lens_distortion_();

		return return_val;

	}

	bool Camera::prep_lens_distortion_() {
		int_t image_size;
		scalar_t del_img_x;
		scalar_t del_img_y;
		scalar_t end_crit = 0.00001;
		std::stringstream msg_output;
		bool end_loop;

		const scalar_t fx = intrinsics_[FX];
		const scalar_t fy = intrinsics_[FY];
		const scalar_t fs = intrinsics_[FS];
		const scalar_t cx = intrinsics_[CX];
		const scalar_t cy = intrinsics_[CY];


		image_size = image_height_ * image_width_;

		//initialize the arrays
		inv_lens_dis_x_.assign(image_size, 0.0);
		inv_lens_dis_y_.assign(image_size, 0.0);
		std::vector<scalar_t> image_x(image_size, 0.0);
		std::vector<scalar_t> image_y(image_size, 0.0);
		std::vector<scalar_t> targ_x(image_size, 0.0);
		std::vector<scalar_t> targ_y(image_size, 0.0);
		std::vector<scalar_t> params(1, 0.0);

		for (int_t i = 0; i < image_size; i++) {
			msg_output = std::stringstream();
			targ_x[i] = (scalar_t)(i % image_width_);
			targ_y[i] = (scalar_t)(i / image_width_);
			inv_lens_dis_x_[i] = (targ_x[i] - cx) / fx;
			inv_lens_dis_y_[i] = (targ_y[i] - cy) / fy;
		}

		for (int_t j = 0; j < 60; j++){
			end_loop = true;
			DEBUG_MSG(" ");
			DEBUG_MSG("Invers distortion prep iteration  " << j);
			sensor_to_image(inv_lens_dis_x_, inv_lens_dis_y_, image_x, image_y);
			for (int_t i = 0; i < image_size; i++) {
				del_img_x = targ_x[i] - image_x[i];
				del_img_y = targ_y[i] - image_y[i];
				if ((abs(del_img_x) > end_crit) || (abs(del_img_y) > end_crit)) end_loop = false;
				inv_lens_dis_x_[i] += (del_img_x) / fx;
				inv_lens_dis_y_[i] += (del_img_y) / fy;
			}
/*
			int_t i;
			for (int_t k = 0; k < 480; k=k+4) {
				i = k * image_width_ + k;
				del_img_x = targ_x[i] - image_x[i];
				del_img_y = targ_y[i] - image_y[i];
				msg_output = std::stringstream();
				msg_output << i << " (" << targ_x[i] << ", " << targ_y[i] << ") " << " (" << image_x[i] << ", " << image_y[i] << ") " << " (" << inv_lens_dis_x_[i] << ", " << inv_lens_dis_x_[i] << ") " << " (" << del_img_x << ", " << del_img_y << ") ";
				DEBUG_MSG(msg_output.str());
			}
*/
			if (end_loop) break;
		}

		return false;
	}


	DICE_LIB_DLL_EXPORT
		void Camera::image_to_sensor(
			std::vector<scalar_t> & image_x,
			std::vector<scalar_t> & image_y,
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y,
			bool integer_locs) {
		
		int_t vect_size = 0;
		int_t index;
		int_t index00;
		int_t index10;
		int_t index01;
		int_t index11;
		int_t x_base, y_base;
		scalar_t dx, dy, x, y;
		vect_size = (int_t)(sen_x.size());

		if (integer_locs) {
			for (int_t i = 0; i < vect_size; i++) {
				index = image_y[i] * image_width_ + image_x[i];
				sen_x[i] = inv_lens_dis_x_[index];
				sen_y[i] = inv_lens_dis_y_[index];
			}
		}
		else
		{
			for (int_t i = 0; i < vect_size; i++) {
				x = image_x[i];
				y = image_y[i];
				x_base = floor(x);
				y_base = floor(y);
				index00 = y_base * image_width_ + x_base;
				index10 = y_base * image_width_ + x_base+1;
				index01 = (y_base + 1) * image_width_ + x_base;
				index11 = (y_base + 1) * image_width_ + x_base + 1;
				dx = x - x_base;
				dy = y - y_base;
				//quick linear interpolation to get the sensor location
				sen_x[i] = inv_lens_dis_x_[index00] * (1 - dx)*(1 - dy) + inv_lens_dis_x_[index10] * dx*(1 - dy) + inv_lens_dis_x_[index01] * (1 - dx)*dy + inv_lens_dis_x_[index11] * dx*dy;
				sen_y[i] = inv_lens_dis_y_[index00] * (1 - dx)*(1 - dy) + inv_lens_dis_y_[index10] * dx*(1 - dy) + inv_lens_dis_y_[index01] * (1 - dx)*dy + inv_lens_dis_y_[index11] * dx*dy;
			}
		}
	}


	DICE_LIB_DLL_EXPORT
		void Camera::sensor_to_image(
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y,
			std::vector<scalar_t> & image_x,
			std::vector<scalar_t> & image_y) {

		const scalar_t fx = intrinsics_[FX];
		const scalar_t fy = intrinsics_[FY];
		const scalar_t fs = intrinsics_[FS];
		const scalar_t cx = intrinsics_[CX];
		const scalar_t cy = intrinsics_[CY];
		const scalar_t k1 = intrinsics_[K1];
		const scalar_t k2 = intrinsics_[K2];
		const scalar_t k3 = intrinsics_[K3];
		const scalar_t k4 = intrinsics_[K4];
		const scalar_t k5 = intrinsics_[K5];
		const scalar_t k6 = intrinsics_[K6];
		const scalar_t p1 = intrinsics_[P1];
		const scalar_t p2 = intrinsics_[P2];
		const scalar_t s1 = intrinsics_[S1];
		const scalar_t s2 = intrinsics_[S2];
		const scalar_t s3 = intrinsics_[S3];
		const scalar_t s4 = intrinsics_[S4];
		const scalar_t t1 = intrinsics_[T1];
		const scalar_t t2 = intrinsics_[T2];
		const int_t lens_dis_type = (int_t)intrinsics_[LD_MODEL];
		scalar_t x_sen, y_sen, rad, dis_coef, rad_sqr;
		scalar_t x_temp, y_temp;
		int_t vect_size=0;

		vect_size = (int_t)(sen_x.size());


		switch (lens_dis_type) {

		case NONE:
			for (int_t i = 0; i < vect_size; i++) {
				image_x[i] = sen_x[i] * fx + cx;
				image_y[i] = sen_y[i] * fy + cy;
			}
		break;

		case K1R1_K2R2_K3R3:
			for (int_t i = 0; i < vect_size; i++) {
				x_sen = sen_x[i];
				y_sen = sen_y[i];
				rad = sqrt(x_sen * x_sen + y_sen * y_sen);
				dis_coef = k1 * rad + k2 * pow(rad, 2) + k3 * pow(rad, 3);
				x_sen = (rad + dis_coef)*x_sen / rad;
				y_sen = (rad + dis_coef)*y_sen / rad;
				image_x[i] = x_sen * fx + cx;
				image_y[i] = y_sen * fy + cy;
			}
		break;

		case K1R2_K2R4_K3R6:
			for (int_t i = 0; i < vect_size; i++) {
				x_sen = sen_x[i];
				y_sen = sen_y[i];
				rad = sqrt(x_sen * x_sen + y_sen * y_sen);
				dis_coef = k1 * pow(rad, 2) + k2 * pow(rad, 4) + k3 * pow(rad, 6);
				x_sen = (rad + dis_coef)*x_sen / rad;
				y_sen = (rad + dis_coef)*y_sen / rad;
				image_x[i] = x_sen * fx + cx;
				image_y[i] = y_sen * fy + cy;
			}
		break;

		case K1R3_K2R5_K3R7:
			for (int_t i = 0; i < vect_size; i++) {
				x_sen = sen_x[i];
				y_sen = sen_y[i];
				rad = sqrt(x_sen * x_sen + y_sen * y_sen);
				dis_coef = k1 * pow(rad, 3) + k2 * pow(rad, 5) + k3 * pow(rad, 7);
				x_sen = (rad + dis_coef)*x_sen / rad;
				y_sen = (rad + dis_coef)*y_sen / rad;
				image_x[i] = x_sen * fx + cx;
				image_y[i] = y_sen * fy + cy;
			}
		break;

		case VIC3D_DIS:  //I believe it is K1R1_K2R2_K3R3 but need to confirm
			for (int_t i = 0; i < vect_size; i++) {
				x_sen = sen_x[i];
				y_sen = sen_y[i];
				rad_sqr = x_sen * x_sen + y_sen * y_sen;
				rad = sqrt((double)rad_sqr);
				dis_coef = k1 * rad + k2 * pow(rad, 2) + k3 * pow(rad, 3);
				x_sen = (rad + dis_coef)*x_sen / rad;
				y_sen = (rad + dis_coef)*y_sen / rad;
				image_x[i] = x_sen * fx + y_sen * fs + cx;
				image_y[i] = y_sen * fy + cy;
			}
		break;

		case OPENCV_DIS:
			{
				const bool has_denom = (k4 != 0 || k5 != 0 || k6 != 0);
				const bool has_tangential = (p1 != 0 || p2 != 0);
				const bool has_prism = (s1 != 0 || s2 != 0 || s3 != 0 || s4 != 0);
				const bool has_Scheimpfug = (t1 != 0 || t2 != 0);

				for (int_t i = 0; i < vect_size; i++) {
					x_sen = sen_x[i];
					y_sen = sen_y[i];
					rad_sqr = x_sen * x_sen + y_sen * y_sen;
					dis_coef = 1 + k1 * rad_sqr + k2 * pow(rad_sqr, 2) + k3 * pow(rad_sqr, 3);
					image_x[i] = x_sen * dis_coef;
					image_y[i] = y_sen * dis_coef;
				}
				if (has_denom) {
					for (int_t i = 0; i < vect_size; i++) {
						x_sen = sen_x[i];
						y_sen = sen_y[i];
						rad_sqr = x_sen * x_sen + y_sen * y_sen;
						dis_coef = 1 / (1 + k4 * rad_sqr + k5 * pow(rad_sqr, 2) + k6 * pow(rad_sqr, 3));
						image_x[i] = image_x[i] * dis_coef;
						image_y[i] = image_y[i] * dis_coef;
					}
				}
				if (has_tangential) {
					for (int_t i = 0; i < vect_size; i++) {
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
					for (int_t i = 0; i < vect_size; i++) {
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
					scalar_t R11, R12, R13, R21, R22, R23, R31, R32, R33;
					scalar_t S11, S12, S13, S21, S22, S23, S31, S32, S33;
					scalar_t norm;

					//calculate the Scheimpfug factors
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

					for (int_t i = 0; i < vect_size; i++) {
						x_temp = image_x[i];
						y_temp = image_y[i];
						norm = 1 / (S31 * x_temp + S32 * y_temp + S33);
						image_x[i] = (x_temp * S11 + y_temp * S12 + S13)*norm;
						image_y[i] = (x_temp * S21 + y_temp * S22 + S23)*norm;
					}
				}
				for (int_t i = 0; i < vect_size; i++) {
					image_x[i] = image_x[i] * fx + cx;
					image_y[i] = image_y[i] * fy + cy;
				}
			}
		break;
		default:
			//raise exception if it gets here?
			for (int_t i = 0; i < vect_size; i++) {
				image_x[i] = sen_x[i] * fx + cx;
				image_y[i] = sen_y[i] * fy + cy;
			}
		}
	}

	DICE_LIB_DLL_EXPORT
		void Camera::sensor_to_cam(
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y,
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z,
			std::vector<scalar_t> & params)
	{
		scalar_t zp = params[ZP];
		scalar_t theta = params[THETA];
		scalar_t phi = params[PHI];
		scalar_t cos_theta, cos_phi, cos_xi;
		scalar_t x_sen, y_sen, denom;
		int_t vect_size;
		cos_theta = cos(theta * DICE_PI / 180.0);
		cos_phi = cos(phi * DICE_PI / 180.0);
		cos_xi = sqrt(1 - cos_theta * cos_theta - cos_phi * cos_phi);

		vect_size = (int_t)(sen_x.size());

		for (int_t i = 0; i < vect_size; i++) {
			x_sen = sen_x[i];
			y_sen = sen_y[i];
			denom = 1 / (cos_xi + y_sen * cos_phi + x_sen * cos_theta);
			cam_x[i] = x_sen * zp * cos_xi * denom;
			cam_y[i] = y_sen * zp * cos_xi * denom;
			cam_z[i] = zp * cos_xi * denom;
		}
	}

	DICE_LIB_DLL_EXPORT
		void Camera::cam_to_sensor(
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z,
			std::vector<scalar_t> & sen_x,
			std::vector<scalar_t> & sen_y)
	{
		int_t vect_size;
		vect_size = (int_t)(sen_x.size());

		for (int_t i = 0; i < vect_size; i++) {
			sen_x[i] = cam_x[i] / cam_z[i];
			sen_y[i] = cam_y[i] / cam_z[i];
		}
	}

	DICE_LIB_DLL_EXPORT
		void cam_to_world(
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z,
			std::vector<scalar_t> & wrld_x,
			std::vector<scalar_t> & wrld_y,
			std::vector<scalar_t> & wrld_z)
	{



	}

	DICE_LIB_DLL_EXPORT
		void world_to_cam(
			std::vector<scalar_t> & wrld_x,
			std::vector<scalar_t> & wrld_y,
			std::vector<scalar_t> & wrld_z,
			std::vector<scalar_t> & cam_x,
			std::vector<scalar_t> & cam_y,
			std::vector<scalar_t> & cam_z)
	{





	}



	DICE_LIB_DLL_EXPORT
		bool Camera::check_valid_(std::string & msg) {
		bool is_valid = true;
		std::stringstream message;
		message = std::stringstream();
		if (intrinsics_[FX] <= 0) {
			message << "fx must be greater than 0" << "\n";
			is_valid = false;
		}
		if (intrinsics_[FY] <= 0) {
			message << "fy must be greater than 0" << "\n";
			is_valid = false;
		}
		if (image_height_ <= 0) {
			message << "image height must be greater than 0" << "\n";
			is_valid = false;
		}
		if (image_width_ <= 0) {
			message << "image width must be greater than 0" << "\n";
			is_valid = false;
		}
		msg = message.str();
		return is_valid;
	}

}  //end DICe namespace