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

#include <DICe.h>
#include <DICe_Camera.h>
#include <DICe_CameraSystem.h>
#include <DICe_Image.h>
#include <DICe_LocalShapeFunction.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

using namespace DICe;

const scalar_t period = 20.0;
const scalar_t freq = DICE_TWOPI/period;
const scalar_t world_def_x = 1.0; // constant displacement in world coordinates
const scalar_t world_def_y = -2.0;
const int_t image_w = 1920;
const int_t image_h = 1200;

#define PERIOD 20.0

//  The world object is a plate with a pattern of a two dimensional sign wave with amplitude of 100.0 and period of 10mm
scalar_t world_ref_intensity_value(const scalar_t & world_x,
  const scalar_t & world_y){
  return 125.0 + 100.0*std::sin(freq*world_x)*std::sin(freq*world_y);
}

// in world coordinates the object is moved by the world_def variables above
scalar_t world_def_intensity_value(const scalar_t & world_x,
  const scalar_t & world_y){
  return 125.0 + 100.0*std::sin(freq*(world_x-world_def_x))*std::sin(freq*(world_y-world_def_y));
}

// the origin of the coordinates system corresponds to centroid of the pixel (0,0) in the image

int main(int argc, char *argv[]) {


  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  scalar_t error_tol = 1.0;
  scalar_t diff_error_tol = 2.0;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // create a camera with the calibration parameters above:

  DICe::Camera_System cam_sys("./cal/image_to_world_2d.xml");
  *outStream << cam_sys << std::endl;

  Teuchos::RCP<DICe::Camera> cam = cam_sys.camera(0);

  // create the reference image:
  std::vector<scalar_t> cam_x(image_w*image_h,0.0);
  std::vector<scalar_t> cam_y(image_w*image_h,0.0);
  std::vector<scalar_t> cam_z(image_w*image_h,0.0);
  std::vector<scalar_t> world_x(image_w*image_h,0.0);
  std::vector<scalar_t> world_y(image_w*image_h,0.0);
  std::vector<scalar_t> world_z(image_w*image_h,0.0);
  std::vector<scalar_t> sensor_x(image_w*image_h,0.0);
  std::vector<scalar_t> sensor_y(image_w*image_h,0.0);
  std::vector<scalar_t> image_x(image_w*image_h,0.0);
  std::vector<scalar_t> image_y(image_w*image_h,0.0);
  for(int_t y=0;y<image_h;++y){
    for(int_t x=0;x<image_w;++x){
      image_x[y*image_w+x] = x;
      image_y[y*image_w+x] = y;
    }
  }
  std::vector<scalar_t> params(3,0.0); // sensor is perpendicular to the camera frame of reference:
  params[Projection_Shape_Function::ZP] = cam->tz();
  params[Projection_Shape_Function::THETA] = 1.57079633;
  params[Projection_Shape_Function::PHI] = 1.57079633;

  // for each pixel convert from image coordinates to world coordinates and get the intensity value:
  cam->image_to_sensor(image_x,image_y,sensor_x,sensor_y);
  cam->sensor_to_cam(sensor_x,sensor_y,cam_x,cam_y,cam_z,params);
  cam->cam_to_world(cam_x,cam_y,cam_z,world_x,world_y,world_z);

  Teuchos::ArrayRCP<intensity_t> def_values(image_h*image_w,0.0);
  Teuchos::ArrayRCP<intensity_t> ref_values(image_h*image_w,0.0);
  for(int_t y=0;y<image_h;++y){
    for(int_t x=0;x<image_w;++x){
      ref_values[y*image_w+x] = world_ref_intensity_value(world_x[y*image_w+x],world_y[y*image_w+x]);
      def_values[y*image_w+x] = world_def_intensity_value(world_x[y*image_w+x],world_y[y*image_w+x]);
    }
  }
  DICe::Image ref_img(image_w,image_h,ref_values);
  DICe::Image def_img(image_w,image_h,def_values);
  //ref_img.write("ITWreference.tif");
  //def_img.write("ITWdeformed.tif");

  std::vector<scalar_t> def_image_x(image_w*image_h,0.0);
  std::vector<scalar_t> def_image_y(image_w*image_h,0.0);
  for(int_t y=0;y<image_h;++y){
    for(int_t x=0;x<image_w;++x){
      world_x[y*image_w+x] = world_x[y*image_w+x] + world_def_x;
      world_y[y*image_w+x] = world_y[y*image_w+x] + world_def_y;
    }
  }
  cam->world_to_cam(world_x,world_y,world_z,cam_x,cam_y,cam_z);
  cam->cam_to_sensor(cam_x,cam_y,cam_z,sensor_x,sensor_y);
  cam->sensor_to_image(sensor_x,sensor_y,def_image_x,def_image_y);

  scalar_t diff = 0.0;
  scalar_t mid_diff = 0.0;
  for(int_t y=100;y<image_h-100;++y){
    for(int_t x=100;x<image_w-100;++x){
      mid_diff = std::abs(ref_img.interpolate_keys_fourth(image_x[y*image_w+x],image_y[y*image_w+x]) - def_img.interpolate_keys_fourth(def_image_x[y*image_w+x],def_image_y[y*image_w+x]));
      diff += mid_diff;
    }
  }
  diff = diff / (image_h*image_w);
  *outStream << "diff: " << diff << std::endl;
  if(diff > diff_error_tol){
    *outStream << "error, image diff error too high" << std::endl;
    errorFlag++;
  }

  // test some sample points with known locations in both image and world coordinates

  std::vector<scalar_t> trial_cam_x(4,0.0);
  std::vector<scalar_t> trial_cam_y(4,0.0);
  std::vector<scalar_t> trial_cam_z(4,0.0);
  std::vector<scalar_t> trial_world_x(4,0.0);
  std::vector<scalar_t> trial_world_y(4,0.0);
  std::vector<scalar_t> trial_world_z(4,0.0);
  std::vector<scalar_t> trial_sensor_x(4,0.0);
  std::vector<scalar_t> trial_sensor_y(4,0.0);
  std::vector<scalar_t> trial_image_x(4,0.0);
  std::vector<scalar_t> trial_image_y(4,0.0);

  trial_world_x[0] = 0;   trial_world_y[0] = 0;   trial_world_z[0] = 0.0;
  trial_world_x[1] = 55;  trial_world_y[1] = 0;   trial_world_z[1] = 0.0;
  trial_world_x[2] = 55;  trial_world_y[2] = 35;  trial_world_z[2] = 0.0;
  trial_world_x[3] = 0;   trial_world_y[3] = 35;  trial_world_z[3] = 0.0;
  cam->world_to_cam(trial_world_x,trial_world_y,trial_world_z,trial_cam_x,trial_cam_y,trial_cam_z);
  cam->cam_to_sensor(trial_cam_x,trial_cam_y,trial_cam_z,trial_sensor_x,trial_sensor_y);
  cam->sensor_to_image(trial_sensor_x,trial_sensor_y,trial_image_x,trial_image_y);

  std::vector<scalar_t> sol_x{314,1225,1324,421};
  std::vector<scalar_t> sol_y{720,857,263,147};

  for(size_t i=0;i<trial_image_x.size();++i){
    if(std::abs(trial_image_x[i] - sol_x[i])>error_tol||std::abs(trial_image_y[i] - sol_y[i])>error_tol){
      *outStream << "error, image coodinates incorrect" << std::endl;
      *outStream << " should be " << sol_x[i] << " " << sol_y[i] << " is " << trial_image_x[i] << " " << trial_image_y[i] << std::endl;
      errorFlag++;
    }
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

