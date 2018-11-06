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


using namespace DICe;

int main(int argc, char *argv[]) {

	DICe::initialize(argc, argv);

	try {
		DEBUG_MSG("Starting test of the camera class");
		bool passed = true;
		DICe::Camera test_cam;
		//arrays to hold intrinsic and extrinsic values
		std::vector<scalar_t>intrinsic1(MAX_CAM_INTRINSIC_PARAMS, 3.21);
		std::vector<scalar_t>extrinsic1(MAX_CAM_EXTRINSIC_PARAMS, 4.321);
		std::vector<scalar_t>intrinsic2(MAX_CAM_INTRINSIC_PARAMS, 0.0);
		std::vector<scalar_t>extrinsic2(MAX_CAM_EXTRINSIC_PARAMS, 0.0);
		std::vector<std::vector<scalar_t>> rotate;
//		std::vector<std::vector<scalar_t>> projection;
		double projection[3][4];
		std::string test_str;
		std::string return_str;
		int_t test_val;
		int_t return_val;
		int_t image_height;
		int_t image_width;


		test_str = "testing";
		test_val = 555;

		//clear and prep the rotation matrix
		rotate.clear();
//		projection.clear();
		for (int_t i = 0; i < 3; ++i) {
			rotate.push_back(std::vector<scalar_t>(3, 0.0));
//			projection.push_back(std::vector<scalar_t>(4, 0.0));
		}

		//fill the intrinsic and extrinsic values in the camera
		test_cam.set_Intrinsics(intrinsic1);
		test_cam.set_Extrinsics(extrinsic1);
		//get the values into the second set of arrays
		test_cam.get_Intrinsics(intrinsic2);
		test_cam.get_Extrinsics(extrinsic2);

		//Test setting getting and clearing the intrinsic and extrinsic parameters
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
		test_str = "";
		test_val = 0;

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

		if (passed) DEBUG_MSG("passed: parameter set, get and clear");
		else DEBUG_MSG("failed: parameter set, get and clear");

		//values from the checkerboard calibration test
		//Fill parameters with reasonable values for testing
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

		rotate[0][0] = 9.999741331604E-01;
		rotate[0][1] = 4.700768261799E-03;
		rotate[0][2] = 5.443876178856E-03;
		rotate[1][0] = -4.764218549160E-03;
		rotate[1][1] = 9.999201838000E-01;
		rotate[1][2] = 1.170163454318E-02;
		rotate[2][0] = -5.388434997075E-03;
		rotate[2][1] = -1.172726767475E-02;
		rotate[2][2] = 9.999167145123E-01;


		
		//create the projection matrix
		projection[0][0] = intrinsic1[FX] * rotate[0][0] + intrinsic1[CX] * rotate[2][0];
		projection[1][0] = intrinsic1[FY] * rotate[1][0] + intrinsic1[CY] * rotate[2][0];
		projection[2][0] = rotate[2][0];

		projection[0][1] = intrinsic1[FX] * rotate[0][1] + intrinsic1[CX] * rotate[2][1];
		projection[1][1] = intrinsic1[FY] * rotate[1][1] + intrinsic1[CY] * rotate[2][1];
		projection[2][1] = rotate[2][1];

		projection[0][2] = intrinsic1[FX] * rotate[0][2] + intrinsic1[CX] * rotate[2][2];
		projection[1][2] = intrinsic1[FY] * rotate[1][2] + intrinsic1[CY] * rotate[2][2];
		projection[2][2] = rotate[2][2];

		projection[0][3] = intrinsic1[FX] * extrinsic1[TX] + intrinsic1[CX] * extrinsic1[TZ];
		projection[1][3] = intrinsic1[FY] * extrinsic1[TY] + intrinsic1[CY] * extrinsic1[TZ];
		projection[2][3] = extrinsic1[TZ];


//		cv::Mat proj(3, 4, cv::DataType<double>::type);
		cv::Mat proj = cv::Mat(3, 4, cv::DataType<double>::type,projection);


		cv::Mat cam, rot, trans, rotx, roty, rotz, euler;
		


		cv::decomposeProjectionMatrix(proj, cam,
			rot, trans,
			rotx,
			roty,
			rotz,
			euler);

		DEBUG_MSG("cam" << cam);
		DEBUG_MSG("rot" << rot);
		DEBUG_MSG("trans" << trans);
		DEBUG_MSG("rotx" << rotx);
		DEBUG_MSG("roty" << roty);
		DEBUG_MSG("rotz" << rotz);
		DEBUG_MSG("euler" << euler);

		//set the image height and width
		test_cam.clear_camera();
		image_height = 480;
		image_width = 640;
		test_cam.set_Image_Height(image_height);
		test_cam.set_Image_Width(image_width);

		//fill the intrinsic and extrinsic values in the camera
		test_cam.set_Intrinsics(intrinsic1);
		test_cam.set_Extrinsics(extrinsic1);

		test_cam.prep_camera();


		scalar_t max_deviation = -1;
		//seed the random number generator
		srand((unsigned)time(0));
		//create arrays of 1000 random x,y image points
		int_t num_points = 10000;
		std::vector<scalar_t> image_x(num_points, 0.0);
		std::vector<scalar_t> image_y(num_points, 0.0);
		std::vector<scalar_t> sen_x(num_points, 0.0);
		std::vector<scalar_t> sen_y(num_points, 0.0);
		std::vector<scalar_t> ret_x(num_points, 0.0);
		std::vector<scalar_t> ret_y(num_points, 0.0);
		for (int_t i = 0; i < num_points; i++) {
			image_x[i] = rand() % image_width;
			image_y[i] = rand() % image_height;
		}
		DEBUG_MSG("image to sensor");
		test_cam.image_to_sensor(image_x, image_y, sen_x, sen_y);
		DEBUG_MSG("sensor to image");
		test_cam.sensor_to_image(sen_x, sen_y, ret_x, ret_y);
		for (int_t i = 0; i < num_points; i++) {
			if (max_deviation < abs(ret_x[i] - image_x[i])) max_deviation = abs(ret_x[i] - image_x[i]);
			if (max_deviation < abs(ret_y[i] - image_y[i])) max_deviation = abs(ret_y[i] - image_y[i]);
		}
		DEBUG_MSG("max integer distortion deviation (10000 rand points): " << max_deviation);


		for (int_t i = 0; i < num_points; i++) {
			image_x[i] = rand() % (image_width );
			image_x[i] = image_x[i] / 100.0;
			image_y[i] = rand() % (image_height * 100);
			image_y[i] = image_y[i] / 100.0;
		}

		DEBUG_MSG("image to sensor");
		test_cam.image_to_sensor(image_x, image_y, sen_x, sen_y, false);
		DEBUG_MSG("sensor to image");
		test_cam.sensor_to_image(sen_x, sen_y, ret_x, ret_y);
		for (int_t i = 0; i < num_points; i++) {
			if (max_deviation < abs(ret_x[i] - image_x[i])) max_deviation = abs(ret_x[i] - image_x[i]);
			if (max_deviation < abs(ret_y[i] - image_y[i])) max_deviation = abs(ret_y[i] - image_y[i]);
		}
		DEBUG_MSG("max non-integer distortion deviation (10000 rand points): " << max_deviation);

	}
	catch (std::exception & e) {
		std::cout << e.what() << std::endl;
		return 1;
	}
	DICe::finalize();
	return 0;

}

