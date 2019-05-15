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

//  The world object is a plate with a pattern of a two dimensional sign wave with amplitude of 100.0 and period of 10mm
scalar_t world_ref_intensity_value(const scalar_t & world_x,
  const scalar_t & world_y,
  const scalar_t & world_z){
  assert(std::abs(world_z)<1.0E-8);
  return 125.0 + 100.0*std::sin(freq*world_x)*std::sin(freq*world_y);
}

// in world coordinates the object is moved by the world_def variables above
scalar_t world_def_intensity_value(const scalar_t & world_x,
  const scalar_t & world_y,
  const scalar_t & world_z){
  assert(std::abs(world_z)<1.0E-8);
  return 125.0 + 100.0*std::sin(freq*(world_x-world_def_x))*std::sin(freq*(world_y-world_def_y));
}

// the origin of the coordinates system corresponds to centroid of the pixel (0,0) in the image

int main(int argc, char *argv[]) {


  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  scalar_t tol = 1.0E-3;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  *outStream << "testing a simple projection example for one point" << std::endl;

  DICe::Camera_System cam_sys_simple("./cal/image_to_world_2d_simple.xml");
  //*outStream << cam_sys_simple << std::endl;
  //outStream->precision(8);
 //*outStream << std::scientific;

  Teuchos::RCP<DICe::Camera> cam_simple = cam_sys_simple.camera(0);
  std::vector<scalar_t> facet_params_simple = cam_simple->get_facet_params();

  std::vector<scalar_t> cam_x_utest(1,0.0);
  std::vector<scalar_t> cam_y_utest(1,0.0);
  std::vector<scalar_t> cam_z_utest(1,0.0);
  std::vector<scalar_t> world_x_utest(1,0.0);
  std::vector<scalar_t> world_y_utest(1,0.0);
  std::vector<scalar_t> world_z_utest(1,0.0);
  std::vector<scalar_t> sensor_x_utest(1,0.0);
  std::vector<scalar_t> sensor_y_utest(1,0.0);
  std::vector<scalar_t> image_x_utest(1,0.0);
  std::vector<scalar_t> image_y_utest(1,0.0);
  image_x_utest[0] = 1060; image_y_utest[0] = 700;
  cam_simple->image_to_sensor(image_x_utest,image_y_utest,sensor_x_utest,sensor_y_utest);
  *outStream << " img_x " << image_x_utest[0] << " img_y " << image_y_utest[0] <<
      " sen_x " << sensor_x_utest[0] << " sen_y " << sensor_y_utest[0] << std::endl;
  // check that sensor x and y are correct
  const scalar_t sen_x_gold = 0.015227;
  const scalar_t sen_y_gold = 0.015227;
  if(std::abs(sen_x_gold - sensor_x_utest[0])>tol){
    *outStream << "error, unit test for sensor x position failed, diff: " << std::abs(sen_x_gold - sensor_x_utest[0]) << std::endl;
    errorFlag++;
  }
  if(std::abs(sen_y_gold - sensor_y_utest[0])>tol){
    *outStream << "error, unit test for sensor y position failed, diff: " << std::abs(sen_y_gold - sensor_y_utest[0]) << std::endl;
    errorFlag++;
  }
  cam_simple->sensor_to_cam(sensor_x_utest,sensor_y_utest,cam_x_utest,cam_y_utest,cam_z_utest,facet_params_simple);
  *outStream << " img_x " << image_x_utest[0] << " img_y " << image_y_utest[0] <<
      " cam_x " << cam_x_utest[0] << " cam_y " << cam_y_utest[0] << " cam_z " << cam_z_utest[0] << std::endl;
  const scalar_t cam_x_gold = 35.792346;
  const scalar_t cam_y_gold = 35.792346;
  const scalar_t cam_z_gold = 2350.54215;
  if(std::abs(cam_x_gold - cam_x_utest[0])>tol){
    *outStream << "error, unit test for camera x position failed, diff: " << std::abs(cam_x_gold - cam_x_utest[0]) << std::endl;
    errorFlag++;
  }
  if(std::abs(cam_y_gold - cam_y_utest[0])>tol){
    *outStream << "error, unit test for camera y position failed, diff: " << std::abs(cam_y_gold - cam_y_utest[0]) << std::endl;
    errorFlag++;
  }
  if(std::abs(cam_z_gold - cam_z_utest[0])>tol){
    *outStream << "error, unit test for camera z position failed, diff: " << std::abs(cam_z_gold - cam_z_utest[0]) << std::endl;
    errorFlag++;
  }
  cam_simple->cam_to_world(cam_x_utest,cam_y_utest,cam_z_utest,world_x_utest,world_y_utest,world_z_utest);
  *outStream << " img_x " << image_x_utest[0] << " img_y " << image_y_utest[0] << " w_x " << world_x_utest[0] << " w_y " << world_y_utest[0] << " w_z " << world_z_utest[0] << std::endl;
  const scalar_t w_x_gold = 35.792346;
  const scalar_t w_y_gold = 2050.85451;
  const scalar_t w_z_gold = 0.0;
  if(std::abs(w_x_gold - world_x_utest[0])>tol){
    *outStream << "error, unit test for world x position failed, diff: " << std::abs(w_x_gold - world_x_utest[0]) << std::endl;
    errorFlag++;
  }
  if(std::abs(w_y_gold - world_y_utest[0])>tol){
    *outStream << "error, unit test for world y position failed, diff: " << std::abs(w_y_gold - world_y_utest[0]) << std::endl;
    errorFlag++;
  }
  if(std::abs(w_z_gold - world_z_utest[0])>tol){
    *outStream << "error, unit test for world z position failed, diff: " << std::abs(w_z_gold - world_z_utest[0]) <<std::endl;
    errorFlag++;
  }

  DICe::Camera_System cam_sys_simple2("./cal/image_to_world_2d_simple2.xml");
  //*outStream << cam_sys_simple2 << std::endl;

  *outStream << "testing another simple projection example for one point" << std::endl;

  Teuchos::RCP<DICe::Camera> cam_simple2 = cam_sys_simple2.camera(0);
  std::vector<scalar_t> facet_params_simple2 = cam_simple2->get_facet_params();
  //outStream->precision(8);
  //*outStream << std::scientific;

  cam_simple2->image_to_sensor(image_x_utest,image_y_utest,sensor_x_utest,sensor_y_utest);
  *outStream << " img_x " << image_x_utest[0] << " img_y " << image_y_utest[0] <<
      " sen_x " << sensor_x_utest[0] << " sen_y " << sensor_y_utest[0] << std::endl;
  // check that sensor x and y are correct
  if(std::abs(sen_x_gold - sensor_x_utest[0])>tol){
    *outStream << "error, unit test for sensor x position failed (simple 2), diff: " <<  std::abs(sen_x_gold - sensor_x_utest[0]) <<  std::endl;
    errorFlag++;
  }
  if(std::abs(sen_y_gold - sensor_y_utest[0])>tol){
    *outStream << "error, unit test for sensor y position failed (simple 2), diff: " <<  std::abs(sen_x_gold - sensor_x_utest[0]) << std::endl;
    errorFlag++;
  }
  cam_simple2->sensor_to_cam(sensor_x_utest,sensor_y_utest,cam_x_utest,cam_y_utest,cam_z_utest,facet_params_simple2);
  *outStream << " img_x " << image_x_utest[0] << " img_y " << image_y_utest[0] <<
      " cam_x " << cam_x_utest[0] << " cam_y " << cam_y_utest[0] << " cam_z " << cam_z_utest[0] << std::endl;
  const scalar_t cam_x_gold2 = 2.4397;
  const scalar_t cam_y_gold2 = 2.4397;
  const scalar_t cam_z_gold2 = 160.224;
  if(std::abs(cam_x_gold2 - cam_x_utest[0])>tol){
    *outStream << "error, unit test for camera x position failed (simple 2), diff: " << std::abs(cam_x_gold2 - cam_x_utest[0]) << std::endl;
    errorFlag++;
  }
  if(std::abs(cam_y_gold2 - cam_y_utest[0])>tol){
    *outStream << "error, unit test for camera y position failed (simple 2), diff: " << std::abs(cam_y_gold2 - cam_y_utest[0]) << std::endl;
    errorFlag++;
  }
  if(std::abs(cam_z_gold2 - cam_z_utest[0])>tol){
    *outStream << "error, unit test for camera z position failed (simple 2), diff: " << std::abs(cam_z_gold2 - cam_z_utest[0]) << std::endl;
    errorFlag++;
  }
  cam_simple2->cam_to_world(cam_x_utest,cam_y_utest,cam_z_utest,world_x_utest,world_y_utest,world_z_utest);
  *outStream << " img_x " << image_x_utest[0] << " img_y " << image_y_utest[0] << " w_x " << world_x_utest[0] << " w_y " << world_y_utest[0] << " w_z " << world_z_utest[0] << std::endl;
  const scalar_t w_x_gold2 = 2.439786;
  const scalar_t w_y_gold2 = -139.796539;
  const scalar_t w_z_gold2 = 0.0;
  if(std::abs(w_x_gold2 - world_x_utest[0])>tol){
    *outStream << "error, unit test for world x position failed (simple 2), diff: " << std::abs(w_x_gold2 - world_x_utest[0])  << std::endl;
    errorFlag++;
  }
  if(std::abs(w_y_gold2 - world_y_utest[0])>tol){
    *outStream << "error, unit test for world y position failed (simple 2), diff: " << std::abs(w_y_gold2 - world_y_utest[0]) << std::endl;
    errorFlag++;
  }
  if(std::abs(w_z_gold2 - world_z_utest[0])>tol){
    *outStream << "error, unit test for world z position failed (simple 2), diff: " << std::abs(w_z_gold2 - world_z_utest[0]) << std::endl;
    errorFlag++;
  }

  *outStream << "testing full image example with synthetic solution for the motion and image intensities" << std::endl;

  DICe::Camera_System cam_sys("./cal/image_to_world_2d.xml");
  //*outStream << cam_sys << std::endl;

  Teuchos::RCP<DICe::Camera> cam = cam_sys.camera(0);
  std::vector<scalar_t> facet_params = cam->get_facet_params();

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
  // for each pixel convert from image coordinates to world coordinates and get the intensity value:
  cam->image_to_sensor(image_x,image_y,sensor_x,sensor_y);
  cam->sensor_to_cam(sensor_x,sensor_y,cam_x,cam_y,cam_z,facet_params);
  cam->cam_to_world(cam_x,cam_y,cam_z,world_x,world_y,world_z);
  //for(size_t i=0;i<image_x.size();++i){
  //  std::cout << i << " img_x " << image_x[i] << " img_y " << image_y[i] << " w_x " << world_x[i] << " w_y " << world_y[i] << " w_z " << world_z[i] << std::endl;
  //}
  Teuchos::ArrayRCP<intensity_t> def_values(image_h*image_w,0.0);
  Teuchos::ArrayRCP<intensity_t> ref_values(image_h*image_w,0.0);
  for(int_t y=0;y<image_h;++y){
    for(int_t x=0;x<image_w;++x){
      ref_values[y*image_w+x] = world_ref_intensity_value(world_x[y*image_w+x],world_y[y*image_w+x],world_z[y*image_w+x]);
      def_values[y*image_w+x] = world_def_intensity_value(world_x[y*image_w+x],world_y[y*image_w+x],world_z[y*image_w+x]);
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
  if(diff > tol){
    *outStream << "error, image diff error too high" << std::endl;
    errorFlag++;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

