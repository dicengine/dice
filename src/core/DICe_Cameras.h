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

#ifndef DICE_CAMERAS_H
#define DICE_CAMERAS_H

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

	/// \class DICe::Cameras
	/// \brief A class for camera calibration parameters and computing projection transformations
	///
	class DICE_LIB_DLL_EXPORT
		Cameras {
	public:
		/// \brief Default constructor
		/// \param param_file_name the name of the file to parse the calibration parameters from
		Cameras(const std::string & param_file_name) {
			load_calibration_parameters(param_file_name);
		};

		/// \brief constructor with no args
		Cameras() {
		};

		/// Pure virtual destructor
		virtual ~Cameras() {}



		/// returns a pointer to all of the camera intrinsics
		std::vector<std::vector<scalar_t> > * intrinsics() {
			return &intrinsics_;
		}

		/// returns a pointer to the calibration extrinsics
		std::vector<std::vector<scalar_t> > * extrinsics() {
			return &extrinsics_;
		}

		/*
		/// triangulate the optimal point in 3D.
		/// returns the the max value of the psuedo matrix
		/// global coordinates are always defined with camera 0 as the origin
		/// unless another transformation is requested by specifying a transformation file
		/// \param x0 sensor x coordinate of the point in camera 0
		/// \param y0 sensor y coordinate of the point in camera 0
		/// \param x1 sensor x coordinate of the point in camera 1
		/// \param y1 sensor y coordinate of the point in camera 1
		/// \param xc_out global x position in camera 0 coords
		/// \param yc_out global y position in camera 0 coords
		/// \param zc_out global z position in camera 0 coords
		/// \param xw_out global x position in world coords
		/// \param yw_out global y position in world coords
		/// \param zw_out global z position in world coords
		/// \param correct_lens_distortion correct for lens distortion
		scalar_t triangulate(const scalar_t & x0,
			const scalar_t & y0,
			const scalar_t & x1,
			const scalar_t & y1,
			scalar_t & xc_out,
			scalar_t & yc_out,
			scalar_t & zc_out,
			scalar_t & xw_out,
			scalar_t & yw_out,
			scalar_t & zw_out,
			const bool correct_lens_distortion = false);

		//  /// project camera 0 coordinates to sensor 1 coordinates
		//  /// \param xc camera 0 x coordinate
		//  /// \param yc camera 0 y coordinate
		//  /// \param zc camera 0 z coordinate
		//  /// \param xs1_out [out] computed sensor 1 coordinate
		//  /// \param ys1_out [out] computed sensor 1 coordinate
		//  void project_camera_0_to_sensor_1(const scalar_t & xc,
		//    const scalar_t & yc,
		//    const scalar_t & zc,
		//    scalar_t & xs2_out,
		//    scalar_t & ys2_out);

			/// convert Cardan-Bryant angles and offsets to T matrix format
			/// \param alpha euler angle in degrees
			/// \param beta euler angle in degrees
			/// \param gamma euler angle in degrees
			/// \param tx offset x
			/// \param ty offset y
			/// \param tz offset z
		void convert_CB_angles_to_T(const scalar_t & alpha,
			const scalar_t & beta,
			const scalar_t & gamma,
			const scalar_t & tx,
			const scalar_t & ty,
			const scalar_t & tz,
			Teuchos::SerialDenseMatrix<int_t, double> & T_out);

		/// computes and returns in place the inverse of a transform
		/// \param T_out the matrix to invert
		void invert_transform(Teuchos::SerialDenseMatrix<int_t, double> & T_out);

		/// correct the lens distortion with a radial model
		/// \param x_s x sensor coordinate to correct, modified in place
		/// \param y_s y sensor coordinate to correct, modified in place
		/// \param camera_id either 0 or 1
		void correct_lens_distortion_radial(scalar_t & x_s,
			scalar_t & y_s,
			const int_t camera_id);

		/// estimate the projective transform from the left to right image
		/// \param left_img pointer to the left image
		/// \param right_img pointer to the right image
		/// \param output_projected_image true if an image should be output using the solution projection parameters
		/// \param use_nonlinear_projection true if an additional 12 nonlinear parameter projection model should be used
		/// \param processor_id the id of the process calling this routine
		/// returns 0 if successful -1 if the linear projection failed -2 if the nonlinear projection fails, -3 for other error
		int_t estimate_projective_transform(Teuchos::RCP<Image> left_img,
			Teuchos::RCP<Image> right_img,
			const bool output_projected_image = false,
			const bool use_nonlinear_projection = false,
			const int_t processor_id = 0);

		/// determine the corresponding right sensor coordinates given the left
		/// using the projective transform
		/// \param xl left x sensor coord
		/// \param yl left y sensor coord
		/// \param xr [out] right x sensor coord
		/// \param yr [out] right y sensor coord
		void project_left_to_right_sensor_coords(const scalar_t & xl,
			const scalar_t & yl,
			scalar_t & xr,
			scalar_t & yr);

		/// set the warp parameter vector of the triangulation
		/// \param params the projective parameters
		void set_warp_params(Teuchos::RCP<std::vector<scalar_t> > & params) {
			TEUCHOS_TEST_FOR_EXCEPTION(params->size() != 12, std::runtime_error, "Error, params vector is the wrong size");
			warp_params_ = params;
		}

		/// return a pointer to the warp parameters
		Teuchos::RCP<std::vector<scalar_t> > warp_params()const {
			return warp_params_;
		}

		/// set the projective parameter vector of the triangulation
		/// \param params the projective parameters
		void set_projective_params(Teuchos::RCP<std::vector<scalar_t> > & params) {
			TEUCHOS_TEST_FOR_EXCEPTION(params->size() != 9, std::runtime_error, "Error, params vector is the wrong size");
			projective_params_ = params;
		}

		/// return a pointer to the projective parameters
		Teuchos::RCP<std::vector<scalar_t> > projective_params()const {
			return projective_params_;
		}

		/// determine the best fit plane to the complete set of X Y Z coordinates (excluding any failed points)
		/// \param cx pointer to the x coordinates from the initial triangulation
		/// \param cy pointer to the y coordinates from the initial triangulation
		/// \param cz pointer to the z coordinates from the intiial triangulation
		/// \param sigma pointer to the sigma field
		void best_fit_plane(Teuchos::RCP<MultiField> & cx,
			Teuchos::RCP<MultiField> & cy,
			Teuchos::RCP<MultiField> & cz,
			Teuchos::RCP<MultiField> & sigma);


		/// returns the cosine of the angle between two vectors
		/// \param a vector a, must have three components
		/// \param b vector b, must have three components
		scalar_t cosine_of_two_vectors(const std::vector<scalar_t> & a,
			const std::vector<scalar_t> & b);

		/// set the transform to the identity matrix
		void clear_trans_extrinsics() {
			for (size_t i = 0; i < 4; ++i)
				for (size_t j = 0; j < 4; ++j)
					trans_extrinsics_[i][j] = i == j ? 1.0 : 0.0;
		}
*/

	private:
		/// \brief load the calibration parameters
		/// \param param_file_name File name of the cal parameters file
		void load_calibration_parameters(const std::string & param_file_name);

		/// 10 x 18 matrix of camera intrinsics (expanded to allow up to 10 cameras and the full openCV distortion)
		/// first index is the camera id, second index is: 
		///   openCV - (cx cy fx fy k1 k2 p1 p2 [k3] [k4 k5 k6] [s1 s2 s3 s4] [tx ty]) 4,5,8,12,14 distortion coef
		///		 Vic3D - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r1 k2r2 k3r3)
		///     DICe - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r2 k2r4 k3r6)
		///	generic1 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r1 k2r2 k3r3)
		///	generic2 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r2 k2r4 k3r6)
		///	generic3 - (cx cy fx fy fs k1 k2 k3) 3 distortion coef (k1r3 k2r5 k3r7)
		std::vector<std::vector<scalar_t> > intrinsics_;

		/// 10 x 6 matrix of camera extrinsics (expanded to allow up to 10 cameras)
		/// first index is the camera id, second index is alpha beta gamma tx ty tz (the Cardan Bryant angles + translations)
		std::vector<std::vector<scalar_t> > extrinsics_;

		/// 12 parameters for projection from X,Y,Z to xsen, ysen
		/// xsen=(P[0]X+P[1]Y+P[2]Z+P[3])/(P[8]X+P[9]Y+P[10]Z+P[11])
		/// ysen=(P[4]X+P[5]Y+P[6]Z+P[7])/(P[8]X+P[9]Y+P[10]Z+P[11])
		std::vector<std::vector<scalar_t> > proj_parms_;

		/// 4 x 4 matrix to convert from one coordinate syste to another coordinate system
		/// cam to cam, cam to world, world to cam
		/// R11 R12 R13 tx
		/// R21 R22 R23 ty
		/// R31 R32 R33 tz
		/// 0   0   0   1
		std::vector<std::vector<scalar_t> > transform_parms_;

		/// these are for compatibility with triangulation and only support 2 camera systems
		/// 12 parameters that define a user supplied transforamation (independent from intrinsic and extrinsic parameters)
		std::vector<std::vector<scalar_t> > user_trans_4x4_params_;

		/// 8 parameters that define a projective transform (independent from intrinsic and extrinsic parameters)
		std::vector<scalar_t> user_trans_6_params_;

		std::string cal_file_ID_ = "DICe XML Calibration File";

		int_t num_cams_ = 0;
		int_t sys_type_ = UNKNOWN;

	};

}// End DICe Namespace

#endif

