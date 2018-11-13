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
#include <cstdlib>
#include <ctime>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <Teuchos_oblackholestream.hpp>
#include <iostream>
#include <fstream>


using namespace DICe;

int main(int argc, char *argv[]) {
	bool all_passed = true;
	DICe::initialize(argc, argv);
	try {

		// only print output if args are given (for testing the output is quiet)
		int_t iprint = argc - 1;
		int_t errorFlag = 0;
		scalar_t errorTol = 1.0E-2;
		scalar_t strong_match = 1.0e-4;
		Teuchos::RCP<std::ostream> outStream;
		Teuchos::oblackholestream bhs; // outputs nothing
		if (iprint > 0)
			outStream = Teuchos::rcp(&std::cout, false);
		else
			outStream = Teuchos::rcp(&bhs, false);

		*outStream << "--- Starting test of the camera class ---" << std::endl;

		bool passed = true;
		DICe::Camera test_cam;
		//arrays to hold intrinsic and extrinsic values
		std::vector<scalar_t>intrinsic1(MAX_CAM_INTRINSIC_PARAMS, 0.0);
		std::vector<scalar_t>extrinsic1(MAX_CAM_EXTRINSIC_PARAMS, 0.0);
		std::vector<scalar_t>intrinsic2(MAX_CAM_INTRINSIC_PARAMS, 0.0);
		std::vector<scalar_t>extrinsic2(MAX_CAM_EXTRINSIC_PARAMS, 0.0);
		std::vector<std::vector<scalar_t>> rotate_3x3;
		std::vector<std::vector<scalar_t>> return_rotate_3x3;
		std::vector<std::vector<scalar_t>> rotate_trans_4x4;
		//projection parameter arrays
		std::vector<scalar_t> proj_params(3, 0.0);
		std::vector<scalar_t> proj_params2(3, 0.0);
		//image, cam, sensor, world and return arrays
		std::vector<scalar_t> img_x(1, 0.0);
		std::vector<scalar_t> img_y(1, 0.0);
		std::vector<scalar_t> sen_x(1, 0.0);
		std::vector<scalar_t> sen_y(1, 0.0);
		std::vector<scalar_t> cam_x(1, 0.0);
		std::vector<scalar_t> cam_y(1, 0.0);
		std::vector<scalar_t> cam_z(1, 0.0);
		std::vector<scalar_t> wld_x(1, 0.0);
		std::vector<scalar_t> wld_y(1, 0.0);
		std::vector<scalar_t> wld_z(1, 0.0);
		std::vector<scalar_t> ret_x(1, 0.0);
		std::vector<scalar_t> ret_y(1, 0.0);
		std::vector<scalar_t> ret_z(1, 0.0);
		std::vector<scalar_t> ret1_x(1, 0.0);
		std::vector<scalar_t> ret1_y(1, 0.0);
		std::vector<scalar_t> ret1_z(1, 0.0);
		std::vector<scalar_t> ret2_x(1, 0.0);
		std::vector<scalar_t> ret2_y(1, 0.0);
		std::vector<scalar_t> ret2_z(1, 0.0);
		//first partials
		std::vector<std::vector<scalar_t>> d1_dx;
		std::vector<std::vector<scalar_t>> d1_dy;
		std::vector<std::vector<scalar_t>> d1_dz;
		std::vector<std::vector<scalar_t>> d2_dx;
		std::vector<std::vector<scalar_t>> d2_dy;
		std::vector<std::vector<scalar_t>> d2_dz;
		std::vector<std::vector<scalar_t>> dn_dx;
		std::vector<std::vector<scalar_t>> dn_dy;
		std::vector<std::vector<scalar_t>> dn_dz;
		std::vector<std::vector<scalar_t>> der_dels;
		std::vector<std::vector<scalar_t>> der_aves;



		std::string test_str = "big";
		std::string return_str;
		int_t test_val = 1024;
		int_t return_val;
		int_t image_height;
		int_t image_width;
		scalar_t params_delta = 0.001;


		//*********************Test the basic get/set functions************************************************

		//fill one pair of arrays with random numbers
		for (int_t i = 0; i < MAX_CAM_INTRINSIC_PARAMS; i++) intrinsic1[i] = (rand() % 10000) / 1000.0;
		for (int_t i = 0; i < MAX_CAM_EXTRINSIC_PARAMS; i++) extrinsic1[i] = (rand() % 10000) / 1000.0;

		//clear and prep the local rotation matrices
		rotate_3x3.clear();
		rotate_trans_4x4.clear();
		for (int_t i = 0; i < 3; ++i) rotate_3x3.push_back(std::vector<scalar_t>(3, 0.0));
		for (int_t i = 0; i < 3; ++i) return_rotate_3x3.push_back(std::vector<scalar_t>(3, 0.0));
		for (int_t i = 0; i < 4; ++i) rotate_trans_4x4.push_back(std::vector<scalar_t>(4, 0.0));

		//fill the intrinsic and extrinsic values in the camera
		test_cam.set_Intrinsics(intrinsic1);
		test_cam.set_Extrinsics(extrinsic1);
		//get the values into the second set of arrays
		test_cam.get_Intrinsics(intrinsic2);
		test_cam.get_Extrinsics(extrinsic2);

		//Test setting getting and clearing the camera parameters
		passed = true;
		for (int_t i = 0; i < MAX_CAM_INTRINSIC_PARAMS; i++) {
			if (intrinsic1[i] != intrinsic2[i])
				passed = false;
		}
		for (int_t i = 0; i < MAX_CAM_EXTRINSIC_PARAMS; i++) {
			if (extrinsic1[i] != extrinsic2[i])
				passed = false;
		}

		test_cam.set_Camera_Comments(test_str);
		return_str = test_cam.get_Camera_Comments();
		if (return_str != test_str) passed = false;

		test_cam.set_Camera_Lens(test_str);
		return_str = test_cam.get_Camera_Lens();
		if (return_str != test_str) passed = false;

		test_cam.set_Identifier(test_str);
		return_str = test_cam.get_Identifier();
		if (return_str != test_str) passed = false;

		test_cam.set_Image_Height(test_val);
		return_val = test_cam.get_Image_Height();
		if (return_val != test_val) passed = false;

		test_cam.set_Image_Width(test_val);
		return_val = test_cam.get_Image_Width();
		if (return_val != test_val) passed = false;

		test_cam.set_Pixel_Depth(test_val);
		return_val = test_cam.get_Pixel_Depth();
		if (return_val != test_val) passed = false;


		//clear the camera then check that all fields have been reset
		test_cam.clear_camera();

		test_str = "";
		test_val = 0;

		test_cam.get_Intrinsics(intrinsic2);
		test_cam.get_Extrinsics(extrinsic2);
		for (int_t i = 0; i < MAX_CAM_INTRINSIC_PARAMS; i++) {
			if (intrinsic2[i] != 0.0)
				passed = false;
		}
		for (int_t i = 0; i < MAX_CAM_EXTRINSIC_PARAMS; i++) {
			if (extrinsic2[i] != 0.0)
				passed = false;
		}

		return_str = test_cam.get_Camera_Comments();
		if (return_str != test_str) passed = false;

		return_str = test_cam.get_Camera_Lens();
		if (return_str != test_str) passed = false;

		return_str = test_cam.get_Identifier();
		if (return_str != test_str) passed = false;

		return_val = test_cam.get_Image_Height();
		if (return_val != test_val) passed = false;

		return_val = test_cam.get_Image_Width();
		if (return_val != test_val) passed = false;

		return_val = test_cam.get_Pixel_Depth();
		if (return_val != test_val) passed = false;

		if (passed) *outStream << "passed: parameter set, get and clear" << std::endl;
		else *outStream << "failed: parameter set, get and clear" << std::endl;

		all_passed = all_passed && passed;

		//*********************Test rotation matrix creation************************************************

		passed = true;

		for (int_t dataset = 0; dataset < 2; dataset++) {
			if (dataset == 0) {
				//values from the checkerboard calibration test
				//Fill parameters with reasonable values for testing
				//From checkerboard calibration in TestStereoCalib
				intrinsic1[CX] = 3.138500331437E+02;
				intrinsic1[CY] = 2.423001711052E+02;
				intrinsic1[FX] = 5.341694772442E+02;
				intrinsic1[FY] = 5.288189029375E+02;
				intrinsic1[FS] = 0.0;
				intrinsic1[K1] = -2.632408663590E-01;
				intrinsic1[K2] = -1.196920656310E-02;
				intrinsic1[K3] = 0;
				intrinsic1[K4] = 0;
				intrinsic1[K5] = 0;
				intrinsic1[K6] = -1.959487138239E-01;
				intrinsic1[S1] = 0;
				intrinsic1[S2] = 0;
				intrinsic1[S3] = 0;
				intrinsic1[S4] = 0;
				intrinsic1[P1] = 0;
				intrinsic1[P2] = 0;
				intrinsic1[T1] = 0;
				intrinsic1[T2] = 0;
				intrinsic1[LD_MODEL] = OPENCV_DIS;

				extrinsic1[ALPHA] = 0;
				extrinsic1[BETA] = 0;
				extrinsic1[GAMMA] = 0;
				extrinsic1[TX] = -3.342839793459E+00;
				extrinsic1[TY] = 4.670078367815E-02;
				extrinsic1[TZ] = 3.252131958135E-03;

				rotate_3x3[0][0] = 9.999741331604E-01;
				rotate_3x3[0][1] = 4.700768261799E-03;
				rotate_3x3[0][2] = 5.443876178856E-03;
				rotate_3x3[1][0] = -4.764218549160E-03;
				rotate_3x3[1][1] = 9.999201838000E-01;
				rotate_3x3[1][2] = 1.170163454318E-02;
				rotate_3x3[2][0] = -5.388434997075E-03;
				rotate_3x3[2][1] = -1.172726767475E-02;
				rotate_3x3[2][2] = 9.999167145123E-01;

				image_height = 480;
				image_width = 640;

			}
			else {
				//values from the dot calibration test
				//Fill parameters with reasonable values for testing
				//From checkerboard calibration in TestStereoCalib
				intrinsic1[CX] = 1.224066475835E+03;
				intrinsic1[CY] = 1.024007702260E+03;
				intrinsic1[FX] = 1.275326315913E+04;
				intrinsic1[FY] = 1.271582348356E+04;
				intrinsic1[FS] = 0.0;
				intrinsic1[K1] = -7.006520231053E-02;
				intrinsic1[K2] = 8.573035432280E+01;
				intrinsic1[K3] = 0;
				intrinsic1[K4] = 0;
				intrinsic1[K5] = 0;
				intrinsic1[K6] = -5.423416904733E-01;
				intrinsic1[S1] = 0;
				intrinsic1[S2] = 0;
				intrinsic1[S3] = 0;
				intrinsic1[S4] = 0;
				intrinsic1[P1] = 0;
				intrinsic1[P2] = 0;
				intrinsic1[T1] = 0;
				intrinsic1[T2] = 0;
				intrinsic1[LD_MODEL] = OPENCV_DIS;

				extrinsic1[ALPHA] = 0;
				extrinsic1[BETA] = 0;
				extrinsic1[GAMMA] = 0;
				extrinsic1[TX] = -8.823158862228E+01;
				extrinsic1[TY] = 5.771721469879E-01;
				extrinsic1[TZ] = 2.396269011734E+01;

				rotate_3x3[0][0] = 8.838522041011E-01;
				rotate_3x3[0][1] = 1.380068199293E-02;
				rotate_3x3[0][2] = 4.675626401693E-01;
				rotate_3x3[1][0] = -1.330392252901E-02;
				rotate_3x3[1][1] = 9.999019740475E-01;
				rotate_3x3[1][2] = -4.364394718824E-03;
				rotate_3x3[2][0] = -4.675770385198E-01;
				rotate_3x3[2][1] = -2.362937250472E-03;
				rotate_3x3[2][2] = 8.839491668510E-01;

				image_height = 2000;
				image_width = 2400;
			}

			//creat a compatible Mat for the decomposition of the projection matrix
			cv::Mat opencv_projection = cv::Mat(3, 4, cv::DataType<double>::type);

			//create the projection matrix
			opencv_projection.at<double>(0, 0) = intrinsic1[FX] * rotate_3x3[0][0] + intrinsic1[CX] * rotate_3x3[2][0];
			opencv_projection.at<double>(1, 0) = intrinsic1[FY] * rotate_3x3[1][0] + intrinsic1[CY] * rotate_3x3[2][0];
			opencv_projection.at<double>(2, 0) = rotate_3x3[2][0];

			opencv_projection.at<double>(0, 1) = intrinsic1[FX] * rotate_3x3[0][1] + intrinsic1[CX] * rotate_3x3[2][1];
			opencv_projection.at<double>(1, 1) = intrinsic1[FY] * rotate_3x3[1][1] + intrinsic1[CY] * rotate_3x3[2][1];
			opencv_projection.at<double>(2, 1) = rotate_3x3[2][1];

			opencv_projection.at<double>(0, 2) = intrinsic1[FX] * rotate_3x3[0][2] + intrinsic1[CX] * rotate_3x3[2][2];
			opencv_projection.at<double>(1, 2) = intrinsic1[FY] * rotate_3x3[1][2] + intrinsic1[CY] * rotate_3x3[2][2];
			opencv_projection.at<double>(2, 2) = rotate_3x3[2][2];

			opencv_projection.at<double>(0, 3) = intrinsic1[FX] * extrinsic1[TX] + intrinsic1[CX] * extrinsic1[TZ];
			opencv_projection.at<double>(1, 3) = intrinsic1[FY] * extrinsic1[TY] + intrinsic1[CY] * extrinsic1[TZ];
			opencv_projection.at<double>(2, 3) = extrinsic1[TZ];

			//create the return matrices
			cv::Mat cam, rot, trans, rotx, roty, rotz, euler;
			//call the decomposer
			cv::decomposeProjectionMatrix(opencv_projection, cam, rot, trans, rotx, roty, rotz, euler);

			//fill the euler angles
			extrinsic1[ALPHA] = euler.at<double>(0);
			extrinsic1[BETA] = euler.at<double>(1);
			extrinsic1[GAMMA] = euler.at<double>(2);

			DEBUG_MSG("Euler: " << euler);

			//clear the camera
			test_cam.clear_camera();
			//fill the intrinsic and extrinsic values in the camera
			test_cam.set_Intrinsics(intrinsic1);
			test_cam.set_Extrinsics(extrinsic1);
			//calculate the rotation matrix from euler angles
			test_cam.prep_camera();
			test_cam.get_3x3_Rotation_Matrix(return_rotate_3x3);


			for (int_t i = 0; i < 3; i++) {
				for (int_t j = 0; j < 3; j++) {
					if (abs(return_rotate_3x3[i][j] - rot.at<double>(i, j)) > strong_match) passed = false;
				}
			}
		}

		if (passed) *outStream << "passed: create rotation matrix from Eulers" << std::endl;
		else *outStream << "failed: create rotation matrix from Eulers" << std::endl;

		//*********************Test the lens distortion functions************************************************
		std::string setup[4];
		setup[0] = "Checkerboard 8 parm lens dist";
		setup[1] = "Checkerboard 3 parm lens dist";
		setup[2] = "Dots 8 parm lens dist";
		setup[3] = "Dots 3 parm lens dist";

		all_passed = all_passed && passed;
		for (int_t m = 0; m < 4; m++) {
			passed = true;
			if (m == 0) {
				// Checkerboard 8 parm lens dist
				intrinsic1[CX] = 3.138500331437E+02;
				intrinsic1[CY] = 2.423001711052E+02;
				intrinsic1[FX] = 5.341694772442E+02;
				intrinsic1[FY] = 5.288189029375E+02;
				intrinsic1[FS] = 0.0;
				intrinsic1[K1] = -23.96269259;
				intrinsic1[K2] = 144.3689661;
				intrinsic1[K3] = -9.005811529;
				intrinsic1[K4] = -23.68880258;
				intrinsic1[K5] = 137.8148457;
				intrinsic1[K6] = 30.21994582;
				intrinsic1[S1] = 0;
				intrinsic1[S2] = 0;
				intrinsic1[S3] = 0;
				intrinsic1[S4] = 0;
				intrinsic1[P1] = 0.001778201;
				intrinsic1[P2] = -0.000292407;
				intrinsic1[T1] = 0;
				intrinsic1[T2] = 0;
				intrinsic1[LD_MODEL] = OPENCV_DIS;

				image_height = 480;
				image_width = 640;

			}
			else if (m == 1) {

				// Checkerboard 3 parm lens dist
				intrinsic1[CX] = 3.138500331437E+02;
				intrinsic1[CY] = 2.423001711052E+02;
				intrinsic1[FX] = 5.341694772442E+02;
				intrinsic1[FY] = 5.288189029375E+02;
				intrinsic1[FS] = 0.0;
				intrinsic1[K1] = -0.268055154;
				intrinsic1[K2] = -0.020449625;
				intrinsic1[K3] = 0.201676357;
				intrinsic1[K4] = 0;
				intrinsic1[K5] = 0;
				intrinsic1[K6] = 0;
				intrinsic1[S1] = 0;
				intrinsic1[S2] = 0;
				intrinsic1[S3] = 0;
				intrinsic1[S4] = 0;
				intrinsic1[P1] = 0;
				intrinsic1[P2] = 0;
				intrinsic1[T1] = 0;
				intrinsic1[T2] = 0;
				intrinsic1[LD_MODEL] = OPENCV_DIS;

				image_height = 480;
				image_width = 640;
			}
			else if (m == 2){
				//dot pattern 8 parm lens dist
				intrinsic1[CX] = 1.224066475835E+03;
				intrinsic1[CY] = 1.024007702260E+03;
				intrinsic1[FX] = 1.275326315913E+04;
				intrinsic1[FY] = 1.271582348356E+04;
				intrinsic1[FS] = 0.0;
				intrinsic1[K1] = 11.02145155;
				intrinsic1[K2] = 41.23525914;
				intrinsic1[K3] = 0.340681332;
				intrinsic1[K4] = 11.29201846;
				intrinsic1[K5] = -41.25264705;
				intrinsic1[K6] = -0.340766754;
				intrinsic1[S1] = 0;
				intrinsic1[S2] = 0;
				intrinsic1[S3] = 0;
				intrinsic1[S4] = 0;
				intrinsic1[P1] = 0.004164821;
				intrinsic1[P2] = -0.00707348;
				intrinsic1[T1] = 0;
				intrinsic1[T2] = 0;
				intrinsic1[LD_MODEL] = OPENCV_DIS;

				image_height = 2000;
				image_width = 2400;				
			}
			else {

				//dot pattern 3 parm lens dist
				intrinsic1[CX] = 1.224066475835E+03;
				intrinsic1[CY] = 1.024007702260E+03;
				intrinsic1[FX] = 1.275326315913E+04;
				intrinsic1[FY] = 1.271582348356E+04;
				intrinsic1[FS] = 0.0;
				intrinsic1[K1] = -0.330765074;
				intrinsic1[K2] = 78.54478098;
				intrinsic1[K3] = 0.512417786;
				intrinsic1[K4] = 0;
				intrinsic1[K5] = 0;
				intrinsic1[K6] = 0;
				intrinsic1[S1] = 0;
				intrinsic1[S2] = 0;
				intrinsic1[S3] = 0;
				intrinsic1[S4] = 0;
				intrinsic1[P1] = 0;
				intrinsic1[P2] = 0;
				intrinsic1[T1] = 0;
				intrinsic1[T2] = 0;
				intrinsic1[LD_MODEL] = OPENCV_DIS;

				extrinsic1[ALPHA] = 0;
				extrinsic1[BETA] = 0;
				extrinsic1[GAMMA] = 0;
				extrinsic1[TX] = -8.823158862228E+01;
				extrinsic1[TY] = 5.771721469879E-01;
				extrinsic1[TZ] = 2.396269011734E+01;

				rotate_3x3[0][0] = 8.838522041011E-01;
				rotate_3x3[0][1] = 1.380068199293E-02;
				rotate_3x3[0][2] = 4.675626401693E-01;
				rotate_3x3[1][0] = -1.330392252901E-02;
				rotate_3x3[1][1] = 9.999019740475E-01;
				rotate_3x3[1][2] = -4.364394718824E-03;
				rotate_3x3[2][0] = -4.675770385198E-01;
				rotate_3x3[2][1] = -2.362937250472E-03;
				rotate_3x3[2][2] = 8.839491668510E-01;

				image_height = 2000;
				image_width = 2400;

			}

			//set the image height and width
			test_cam.clear_camera();

			test_cam.set_Image_Height(image_height);
			test_cam.set_Image_Width(image_width);

			//fill the intrinsic and extrinsic values in the camera
			test_cam.set_Intrinsics(intrinsic1);
			test_cam.set_Extrinsics(extrinsic1);

			//prep the camera
			test_cam.prep_camera();

			//assume the max distortion would be at point (0,0)
			img_x.assign(1, 0.0);
			img_y.assign(1, 0.0);
			sen_x.assign(1, 0.0);
			sen_y.assign(1, 0.0);
			ret_x.assign(1, 0.0);
			ret_y.assign(1, 0.0);
			//transform the points from the image to the sensor (inverse distortion)
			test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y);
			//convert that value back to image coordinates without applying distortoin
			img_x[0] = sen_x[0] * intrinsic1[FX] + intrinsic1[CX];
			img_y[0] = sen_y[0] * intrinsic1[FY] + intrinsic1[CY];
			DEBUG_MSG("max distortion movement: " << setup[m] << ":  " << img_x[0] << " " << img_y[0]);

			//testing the distortion/inverted distortion routines
			scalar_t max_deviation = -1;
			//seed the random number generator
			srand((unsigned)time(0));
			//create arrays of 10000 random x,y image points
			int_t num_points = 10000;
			img_x.assign(num_points, 0.0);
			img_y.assign(num_points, 0.0);
			sen_x.assign(num_points, 0.0);
			sen_y.assign(num_points, 0.0);
			cam_x.assign(num_points, 0.0);
			cam_y.assign(num_points, 0.0);
			cam_z.assign(num_points, 0.0);
			wld_x.assign(num_points, 0.0);
			wld_y.assign(num_points, 0.0);
			wld_z.assign(num_points, 0.0);
			ret_x.assign(num_points, 0.0);
			ret_y.assign(num_points, 0.0);
			ret_z.assign(num_points, 0.0);
			for (int_t i = 0; i < num_points; i++) {
				img_x[i] = rand() % image_width;
				img_y[i] = rand() % image_height;
			}
			//transform the points from the image to the sensor (inverse distortion)
			test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y);
			//make sure the projections are doing something
			bool all_match = true;
			for (int_t i = 0; i < num_points; i++) {
				if (0.01 < abs(img_x[i] - sen_x[i])) all_match = false;
			}
			if (all_match) {
				*outStream << "failed: no change in image coordinates" << setup[m] << ":  " << std::endl;
				all_passed = false;
			}

			//convert the sensor locations to image locations
			test_cam.sensor_to_image(sen_x, sen_y, ret_x, ret_y);
			for (int_t i = 0; i < num_points; i++) {
				if (max_deviation < abs(ret_x[i] - img_x[i])) max_deviation = abs(ret_x[i] - img_x[i]);
				if (max_deviation < abs(ret_y[i] - img_y[i])) max_deviation = abs(ret_y[i] - img_y[i]);
			}
			if (max_deviation > 0.001) {
				*outStream << "failed: max integer distortion deviation(10000 rand points) : " << setup[m] << ":  " << max_deviation << std::endl;
				all_passed = false;
			}
			else *outStream << "passed: max integer distortion deviation(10000 rand points) : " << setup[m] << ":  " << max_deviation << std::endl;

			//do the same with non-integer locations
			for (int_t i = 0; i < num_points; i++) {
				img_x[i] = rand() % (image_width);
				img_x[i] = img_x[i] / 100.0;
				img_y[i] = rand() % (image_height * 100);
				img_y[i] = img_y[i] / 100.0;
			}
			test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y, false);
			test_cam.sensor_to_image(sen_x, sen_y, ret_x, ret_y);
			for (int_t i = 0; i < num_points; i++) {
				if (max_deviation < abs(ret_x[i] - img_x[i])) max_deviation = abs(ret_x[i] - img_x[i]);
				if (max_deviation < abs(ret_y[i] - img_y[i])) max_deviation = abs(ret_y[i] - img_y[i]);
			}
			if (max_deviation > 0.001) {
				*outStream << "failed: max non-integer distortion deviation(10000 rand points) : " << setup[m] << ":  " << max_deviation << std::endl;
				all_passed = false;
			}
			else *outStream << "passed: max non-integer distortion deviation(10000 rand points) : " << setup[m] << ":  " << max_deviation << std::endl;
		}

		//*****************************test the other transformation functions****************
		//make sure we are playing with the dot pattern parameters
		//dot pattern 3 parm lens dist
		intrinsic1[CX] = 1.224066475835E+03;
		intrinsic1[CY] = 1.024007702260E+03;
		intrinsic1[FX] = 1.275326315913E+04;
		intrinsic1[FY] = 1.271582348356E+04;
		intrinsic1[FS] = 0.0;
		intrinsic1[K1] = -0.330765074;
		intrinsic1[K2] = 78.54478098;
		intrinsic1[K3] = 0.512417786;
		intrinsic1[K4] = 0;
		intrinsic1[K5] = 0;
		intrinsic1[K6] = 0;
		intrinsic1[S1] = 0;
		intrinsic1[S2] = 0;
		intrinsic1[S3] = 0;
		intrinsic1[S4] = 0;
		intrinsic1[P1] = 0;
		intrinsic1[P2] = 0;
		intrinsic1[T1] = 0;
		intrinsic1[T2] = 0;
		intrinsic1[LD_MODEL] = OPENCV_DIS;

		extrinsic1[ALPHA] = 0;
		extrinsic1[BETA] = 0;
		extrinsic1[GAMMA] = 0;
		extrinsic1[TX] = -8.823158862228E+01;
		extrinsic1[TY] = 5.771721469879E-01;
		extrinsic1[TZ] = 2.396269011734E+01;

		rotate_3x3[0][0] = 8.838522041011E-01;
		rotate_3x3[0][1] = 1.380068199293E-02;
		rotate_3x3[0][2] = 4.675626401693E-01;
		rotate_3x3[1][0] = -1.330392252901E-02;
		rotate_3x3[1][1] = 9.999019740475E-01;
		rotate_3x3[1][2] = -4.364394718824E-03;
		rotate_3x3[2][0] = -4.675770385198E-01;
		rotate_3x3[2][1] = -2.362937250472E-03;
		rotate_3x3[2][2] = 8.839491668510E-01;

		image_height = 2000;
		image_width = 2400;

		//continue using the dot calibration values
		//set the image height and width
		test_cam.clear_camera();

		test_cam.set_Image_Height(image_height);
		test_cam.set_Image_Width(image_width);

		//fill the intrinsic and extrinsic values in the camera
		test_cam.set_Intrinsics(intrinsic1);
		test_cam.set_Extrinsics(extrinsic1);

		//fill the rotation matrix
		test_cam.set_3x3_Rotation_Matrix(rotate_3x3);

		//prep the camera
		test_cam.prep_camera();

		//use reasonalable projection parameters for the setup
		proj_params[ZP] = 188;
		proj_params[THETA] = 1.32;
		proj_params[PHI] = 1.5;

		proj_params2[ZP] = proj_params[ZP];
		proj_params2[THETA] = proj_params[THETA];
		proj_params2[PHI] = proj_params[PHI];

		//testing the distortion/inverted distortion routines
		scalar_t max_deviation = -1;

		//test projection - transformation functions
		srand((unsigned)time(0));
		//create arrays of 10000 random x,y image points
		int_t num_points = 10;

		//initialize the arrays of position values
		img_x.assign(num_points, 0.0);
		img_y.assign(num_points, 0.0);
		sen_x.assign(num_points, 0.0);
		sen_y.assign(num_points, 0.0);
		cam_x.assign(num_points, 0.0);
		cam_y.assign(num_points, 0.0);
		cam_z.assign(num_points, 0.0);
		wld_x.assign(num_points, 0.0);
		wld_y.assign(num_points, 0.0);
		wld_z.assign(num_points, 0.0);
		ret_x.assign(num_points, 0.0);
		ret_y.assign(num_points, 0.0);
		ret_z.assign(num_points, 0.0);
		ret1_x.assign(num_points, 0.0);
		ret1_y.assign(num_points, 0.0);
		ret1_z.assign(num_points, 0.0);		
		ret2_x.assign(num_points, 0.0);
		ret2_y.assign(num_points, 0.0);
		ret2_z.assign(num_points, 0.0);
		//fill the image locations with random numbers
		for (int_t i = 0; i < num_points; i++) {
			img_x[i] = rand() % image_width;
			img_y[i] = rand() % image_height;
		}
		//convert to the camera coordiante and then back to image coordiates
		test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y);
		test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params);
		test_cam.cam_to_sensor(cam_x, cam_y, cam_z, sen_x, sen_y);
		test_cam.sensor_to_image(sen_x, sen_y, ret_x, ret_y);
		//the miage coordinates should be the same
		for (int_t i = 0; i < num_points; i++) {
			if (max_deviation < abs(ret_x[i] - img_x[i])) max_deviation = abs(ret_x[i] - img_x[i]);
			if (max_deviation < abs(ret_y[i] - img_y[i])) max_deviation = abs(ret_y[i] - img_y[i]);
		}
		if (max_deviation > 0.001) {
			*outStream << "failed: sensor->cam, cam->sensor maximum deviation: " << max_deviation << " pixels on image" << std::endl;
			all_passed = false;
		}
		else *outStream << "passed: sensor->cam, cam->sensor maximum deviation: " << max_deviation << " pixels on image" << std::endl;

		//repeat with the world coordinate transform
		test_cam.cam_to_world(cam_x, cam_y, cam_z, wld_x, wld_y, wld_z);
		test_cam.world_to_cam(wld_x, wld_y, wld_z, cam_x, cam_y, cam_z);
		test_cam.cam_to_sensor(cam_x, cam_y, cam_z, sen_x, sen_y);
		test_cam.sensor_to_image(sen_x, sen_y, ret_x, ret_y);

		for (int_t i = 0; i < num_points; i++) {
			if (max_deviation < abs(ret_x[i] - img_x[i])) max_deviation = abs(ret_x[i] - img_x[i]);
			if (max_deviation < abs(ret_y[i] - img_y[i])) max_deviation = abs(ret_y[i] - img_y[i]);
		}
		if (max_deviation > 0.001) {
			*outStream << "failed: cam->world, world->cam maximum deviation: " << max_deviation << " pixels on image" << std::endl;
			all_passed = false;
		}
		else *outStream << "passed: cam->world, world->cam maximum deviation: " << max_deviation << " pixels on image" << std::endl;

		//***********************Test the derivitives****************************************
		d1_dx.clear();
		d1_dx.resize(3);
		d1_dy.clear();
		d1_dy.resize(3);
		d1_dz.clear();
		d1_dz.resize(3);
		d2_dx.clear();
		d2_dx.resize(3);
		d2_dy.clear();
		d2_dy.resize(3);
		d2_dz.clear();
		d2_dz.resize(3);
		dn_dx.clear();
		dn_dx.resize(3);
		dn_dy.clear();
		dn_dy.resize(3);
		dn_dz.clear();
		dn_dz.resize(3);
		der_dels.clear();
		der_dels.resize(3);
		der_aves.clear();
		der_aves.resize(3);


		for (int_t i = 0; i < 3; i++) {
			d1_dx[i].assign(num_points, 0.0);
			d1_dy[i].assign(num_points, 0.0);
			d1_dz[i].assign(num_points, 0.0);
			d2_dx[i].assign(num_points, 0.0);
			d2_dy[i].assign(num_points, 0.0);
			d2_dz[i].assign(num_points, 0.0);
			dn_dx[i].assign(num_points, 0.0);
			dn_dy[i].assign(num_points, 0.0);
			dn_dz[i].assign(num_points, 0.0);
			der_dels[i].assign(3, 0.0);
			der_aves[i].assign(3, 0.0);
		}

		DEBUG_MSG(" *************Sensor to cam derivitives *****************************************");
		//first derivitives of the projection function alone
		//calculate numberical first derivitives
		test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y);
		test_cam.sensor_to_cam(sen_x, sen_y, ret_x, ret_y, ret_z, proj_params);
		proj_params2[ZP] = proj_params[ZP] + params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, ret2_x, ret2_y, ret2_z, proj_params2);
		proj_params2[ZP] = proj_params[ZP] - params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, ret1_x, ret1_y, ret1_z, proj_params2);
		for (int_t i = 0; i < num_points; i++) {
			dn_dx[ZP][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
			dn_dy[ZP][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
			dn_dz[ZP][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
		}

		proj_params2[ZP] = proj_params[ZP];
		proj_params2[THETA] = proj_params[THETA] + params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, ret2_x, ret2_y, ret2_z, proj_params2);
		proj_params2[THETA] = proj_params[THETA] - params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, ret1_x, ret1_y, ret1_z, proj_params2);
		for (int_t i = 0; i < num_points; i++) {
			dn_dx[THETA][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
			dn_dy[THETA][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
			dn_dz[THETA][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
		}

		proj_params2[THETA] = proj_params[THETA];
		proj_params2[PHI] = proj_params[PHI] + params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, ret2_x, ret2_y, ret2_z, proj_params2);
		proj_params2[PHI] = proj_params[PHI] - params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, ret1_x, ret1_y, ret1_z, proj_params2);
		for (int_t i = 0; i < num_points; i++) {
			dn_dx[PHI][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
			dn_dy[PHI][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
			dn_dz[PHI][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
		}
		//compair the numberical derivitives to the calculated ones
		//print out additional information for the first ten points
		int_t disp_pnts = 10;
		if (disp_pnts > num_points) disp_pnts = num_points;
		test_cam.sensor_to_cam(sen_x, sen_y, ret2_x, ret2_y, ret2_z, proj_params, d1_dx, d1_dy, d1_dz);
		for (int_t i = 0; i < disp_pnts; i++) {
			DEBUG_MSG(" ");
			DEBUG_MSG("   dZP(x y z): X: " << dn_dx[ZP][i] << " " << d1_dx[ZP][i] << " Y: " << dn_dy[ZP][i] << " " << d1_dy[ZP][i] << " Z: " << dn_dz[ZP][i] << " " << d1_dz[ZP][i]);
			DEBUG_MSG("dTHETA(x y z): X: " << dn_dx[THETA][i] << " " << d1_dx[THETA][i] << " Y: " << dn_dy[THETA][i] << " " << d1_dy[THETA][i] << " Z: " << dn_dz[THETA][i] << " " << d1_dz[THETA][i]);
			DEBUG_MSG("  dPHI(x y z): X: " << dn_dx[PHI][i] << " " << d1_dx[PHI][i] << " Y: " << dn_dy[PHI][i] << " " << d1_dy[PHI][i] << " Z: " << dn_dz[PHI][i] << " " << d1_dz[PHI][i]);
		}

		//calculate the average derivitive value and the maximum deviation between the numerical and analyitical derivitives
		for (int_t i = 0; i < 3; i++) {
			der_dels[i].assign(3, 0.0);
			der_aves[i].assign(3, 0.0);
		}

		for (int_t j = 0; j < 3; j++){
			for (int_t i = 0; i < num_points; i++) {
				der_aves[j][0] += abs(d1_dx[j][i]);
				der_aves[j][1] += abs(d1_dy[j][i]);
				der_aves[j][2] += abs(d1_dz[j][i]);

				if (der_dels[j][0] < abs(dn_dx[j][i] - d1_dx[j][i])) der_dels[j][0] = abs(dn_dx[j][i] - d1_dx[j][i]);
				if (der_dels[j][1] < abs(dn_dy[j][i] - d1_dy[j][i])) der_dels[j][1] = abs(dn_dy[j][i] - d1_dy[j][i]);
				if (der_dels[j][2] < abs(dn_dz[j][i] - d1_dz[j][i])) der_dels[j][2] = abs(dn_dz[j][i] - d1_dz[j][i]);
			}
			der_aves[j][0] = der_aves[j][0] / num_points;
			der_aves[j][1] = der_aves[j][1] / num_points;
			der_aves[j][2] = der_aves[j][2] / num_points;
		}
		DEBUG_MSG(" ");
		DEBUG_MSG("   dZP(x y z) (ave,max dev): X: (" << der_aves[ZP][0] << ", " << der_dels[ZP][0] << ") Y: (" << der_aves[ZP][1] << ", " << der_dels[ZP][1] 
			<< ") Z: (" << der_aves[ZP][2] << ", " << der_dels[ZP][2]<<")" << std::endl);
		DEBUG_MSG("dTheta(x y z) (ave,max dev): X: (" << der_aves[THETA][0] << ", " << der_dels[THETA][0] << ") Y: (" << der_aves[THETA][1] << ", " << der_dels[THETA][1] 
			<< ") Z: (" << der_aves[THETA][2] << ", " << der_dels[THETA][2] << ")" << std::endl);
		DEBUG_MSG("  dPhi(x y z) (ave,max dev): X: (" << der_aves[PHI][0] << ", " << der_dels[PHI][0] << ") Y: (" << der_aves[PHI][1] << ", " << der_dels[PHI][1] 
			<< ") Z: (" << der_aves[PHI][2] << ", " << der_dels[PHI][2] << ")" << std::endl);



		DEBUG_MSG(" ");
		DEBUG_MSG(" *************sensor to world derivitives *****************************************");
		//calculate and print out the values including the transformation to world coordinates.
		test_cam.image_to_sensor(img_x, img_y, sen_x, sen_y);
		test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params);
		test_cam.cam_to_world(cam_x, cam_y, cam_z, ret_x, ret_y, ret_z);

		proj_params2[ZP] = proj_params[ZP] + params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
		test_cam.cam_to_world(cam_x, cam_y, cam_z, ret2_x, ret2_y, ret2_z);
		proj_params2[ZP] = proj_params[ZP] - params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
		test_cam.cam_to_world(cam_x, cam_y, cam_z, ret1_x, ret1_y, ret1_z);
		for (int_t i = 0; i < num_points; i++) {
			dn_dx[ZP][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
			dn_dy[ZP][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
			dn_dz[ZP][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
		}

		proj_params2[ZP] = proj_params[ZP];
		proj_params2[THETA] = proj_params[THETA] + params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
		test_cam.cam_to_world(cam_x, cam_y, cam_z, ret2_x, ret2_y, ret2_z);
		proj_params2[THETA] = proj_params[THETA] - params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
		test_cam.cam_to_world(cam_x, cam_y, cam_z, ret1_x, ret1_y, ret1_z);
		for (int_t i = 0; i < num_points; i++) {
			dn_dx[THETA][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
			dn_dy[THETA][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
			dn_dz[THETA][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
		}

		proj_params2[THETA] = proj_params[THETA];
		proj_params2[PHI] = proj_params[PHI] + params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
		test_cam.cam_to_world(cam_x, cam_y, cam_z, ret2_x, ret2_y, ret2_z);
		proj_params2[PHI] = proj_params[PHI] - params_delta;
		test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params2);
		test_cam.cam_to_world(cam_x, cam_y, cam_z, ret1_x, ret1_y, ret1_z);
		for (int_t i = 0; i < num_points; i++) {
			dn_dx[PHI][i] = (ret2_x[i] - ret1_x[i]) / (2 * params_delta);
			dn_dy[PHI][i] = (ret2_y[i] - ret1_y[i]) / (2 * params_delta);
			dn_dz[PHI][i] = (ret2_z[i] - ret1_z[i]) / (2 * params_delta);
		}

		//print out additional information for the first ten points
		if (disp_pnts > num_points) disp_pnts = num_points;
		test_cam.sensor_to_cam(sen_x, sen_y, cam_x, cam_y, cam_z, proj_params, d1_dx, d1_dy, d1_dz);
		test_cam.cam_to_world(cam_x, cam_y, cam_z, ret1_x, ret1_y, ret1_z, d1_dx, d1_dy, d1_dz, d2_dx, d2_dy, d2_dz);
		for (int_t i = 0; i < disp_pnts; i++) {
			DEBUG_MSG(" ");
			DEBUG_MSG("   dZP(x y z): X: " << dn_dx[ZP][i] << " " << d2_dx[ZP][i] << " Y: " << dn_dy[ZP][i] << " " << d2_dy[ZP][i] << " Z: " << dn_dz[ZP][i] << " " << d2_dz[ZP][i]);
			DEBUG_MSG("dTHETA(x y z): X: " << dn_dx[THETA][i] << " " << d2_dx[THETA][i] << " Y: " << dn_dy[THETA][i] << " " << d2_dy[THETA][i] << " Z: " << dn_dz[THETA][i] << " " << d2_dz[THETA][i]);
			DEBUG_MSG("  dPHI(x y z): X: " << dn_dx[PHI][i] << " " << d2_dx[PHI][i] << " Y: " << dn_dy[PHI][i] << " " << d2_dy[PHI][i] << " Z: " << dn_dz[PHI][i] << " " << d2_dz[PHI][i]);
		}

		//calculate the averages and deviations
		for (int_t i = 0; i < 3; i++) {
			der_dels[i].assign(3, 0.0);
			der_aves[i].assign(3, 0.0);
		}

		for (int_t j = 0; j < 3; j++) {
			for (int_t i = 0; i < num_points; i++) {
				der_aves[j][0] += abs(d2_dx[j][i]);
				der_aves[j][1] += abs(d2_dy[j][i]);
				der_aves[j][2] += abs(d2_dz[j][i]);

				if (der_dels[j][0] < abs(dn_dx[j][i] - d2_dx[j][i])) der_dels[j][0] = abs(dn_dx[j][i] - d2_dx[j][i]);
				if (der_dels[j][1] < abs(dn_dy[j][i] - d2_dy[j][i])) der_dels[j][1] = abs(dn_dy[j][i] - d2_dy[j][i]);
				if (der_dels[j][2] < abs(dn_dz[j][i] - d2_dz[j][i])) der_dels[j][2] = abs(dn_dz[j][i] - d2_dz[j][i]);
			}
			der_aves[j][0] = der_aves[j][0] / num_points;
			der_aves[j][1] = der_aves[j][1] / num_points;
			der_aves[j][2] = der_aves[j][2] / num_points;
		}
		DEBUG_MSG(" ");
		DEBUG_MSG("   dZP(x y z) (ave,max dev): X: (" << der_aves[ZP][0] << ", " << der_dels[ZP][0] << ") Y: (" << der_aves[ZP][1] << ", " << der_dels[ZP][1]
			<< ") Z: (" << der_aves[ZP][2] << ", " << der_dels[ZP][2] << ")" << std::endl);
		DEBUG_MSG("dTheta(x y z) (ave,max dev): X: (" << der_aves[THETA][0] << ", " << der_dels[THETA][0] << ") Y: (" << der_aves[THETA][1] << ", " << der_dels[THETA][1]
			<< ") Z: (" << der_aves[THETA][2] << ", " << der_dels[THETA][2] << ")" << std::endl);
		DEBUG_MSG("  dPhi(x y z) (ave,max dev): X: (" << der_aves[PHI][0] << ", " << der_dels[PHI][0] << ") Y: (" << der_aves[PHI][1] << ", " << der_dels[PHI][1]
			<< ") Z: (" << der_aves[PHI][2] << ", " << der_dels[PHI][2] << ")" << std::endl);


	}

	catch (std::exception & e) {
		std::cout << e.what() << std::endl;
		return 1;
	}
	DICe::finalize();
	return all_passed;

}

