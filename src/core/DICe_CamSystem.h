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


#include <DICe_Camera.h>

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

	private:

		//create an array of generic cameras
		DICe::Camera Cameras_[10];

		/// these are for compatibility with triangulation and only support 2 camera systems
		/// 12 parameters that define a user supplied transforamation (independent from intrinsic and extrinsic parameters)
		std::vector<std::vector<scalar_t>> user_trans_4x4_params_;

		/// 8 parameters that define a projective transform (independent from intrinsic and extrinsic parameters)
		std::vector<scalar_t> user_trans_6_params_;

		/// 3x3 openCV rotation parameters from sterio calibration
		std::vector<std::vector<scalar_t>> openCV_rot_trans_3x4_params_;


		std::string cal_file_ID_ = "DICe XML Calibration File";

		int_t num_cams_ = 0;
		int_t sys_type_ = UNKNOWN;
		bool has_6_transform_ = false;
		bool has_4x4_transform_ = false;
		bool has_opencv_rot_trans_ = false;
		const int_t MAX_NUM_CAMERAS_PER_SYSTEM = 10;
		bool valid_cal_file_ = false;
		std::stringstream cal_file_error_;

	};

}// End DICe Namespace

#endif

